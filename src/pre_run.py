import logging
import polars as pl
import os 
from tqdm import tqdm
import json
import functools

# TODO: cleanify
# Analyze column types and dump schema summary
def audit_column_types(file_path: str, log_path: str, schema_json_path: str, sample_rows: int = 10000, int_threshold: float = 0.8):
    df_sample = pl.read_csv(file_path, infer_schema_length=sample_rows)

    log_lines = []
    schema = {}

    for col in tqdm(df_sample.columns, desc="Auditing column types"):
        values = df_sample[col].drop_nulls().unique().to_list()
        n = len(values)
        num_int = 0
        for v in values:
            try:
                int(v)
                num_int += 1
            except:
                continue

        int_ratio = num_int / n if n else 0
        dtype = "Int64" if int_ratio >= int_threshold else "Utf8"
        log_lines.append(f"{col}\t{int_ratio:.2f}\t{dtype}")
        schema[col] = dtype

    # Save audit log
    with open(log_path, "w") as f:
        f.write("column\tint_ratio\ttype_inferred\n")
        f.write("\n".join(log_lines))

    # Save schema as JSON
    with open(schema_json_path, "w") as f:
        json.dump(schema, f, indent=2)



class Runner:
    def __init__(self, sigs_path, log_dir="."):
        self.sigs_path = sigs_path
        self.schema_log_path = f"{log_dir}/PRE.types-audit.log"
        self.schema_json_path = f"{log_dir}/s_frame.schema.json"
        
        self.logger = self._setup_logging(log_dir)
        
        # Load data and process immediately upon instantiation
        audit_column_types(self.sigs_path, self.schema_log_path, self.schema_json_path)
        self._load_data()
        self._process_data()

    def _setup_logging(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        full_handler = logging.FileHandler(f"{log_dir}/PRE.processing.log", mode='w')
        full_handler.setFormatter(formatter)
        logger.addHandler(full_handler)
        return logger

    def _load_data(self):
        """Load data with all columns forced to Utf8 to ensure resilience."""
        schema = pl.read_csv(self.sigs_path, infer_schema_length=10000).schema
        utf8_schema = {col: pl.Utf8 for col in schema} # overriding all types!
        # self.s = pl.read_csv(self.sigs_path, schema_overrides=utf8_schema)
        self.s = pl.read_csv(
            self.sigs_path,
            schema_overrides=utf8_schema,
            null_values=["null", "NA", "NUB", "", "-"]
        )
        self.logger.info("Data loaded with Utf8 schema.")

    def _process_data(self):
        """Process the data (clean NaN and handle duplicates)."""

        sigs_anatomy_essentials = [
            'variant_cui', 
            'allDays',
            "cycle_length_lb",
            "cycle_length_ub",   
            "cycle_length_unit",
            "timing_sequence"
        ]

        group_keys=["condition_cui", "regimen_cui", "variant_cui"]
        
        # Nan condition_cui fallback
        self.s = self.s.with_columns(
            pl.col("condition_cui").fill_null("undefined")
        )

        # Clean NaNs from group keys - safe
        self.s = self.s.drop_nulls(group_keys) 

        self.log_regimen_level_stats()
        self.log_cycle_length_unit(group_keys)
        self.log_variants_level_stats(group_keys)


        self.handle_processing(group_keys, sigs_anatomy_essentials)
        
        self.logger.info(f"Data pre-processed. Shape: {self.s.shape}")

    def log_cycle_length_unit(self, group_keys=["condition_cui", "regimen_cui", "variant_cui"]):
        """checks cycle_length for indeterminate"""

        cycle_length_unit_indefinite = (
            self.s
            .group_by(group_keys)
            .agg([
                (pl.col("cycle_length_unit") == "indeterminate").any().alias("has_indeterminate")
            ])
            .filter(pl.col("has_indeterminate"))
            .shape[0]
        )

        cycle_length_unit_indefinite_regimens_unique = (
             self.s
            .filter(pl.col("cycle_length_unit") == "indeterminate")
            .group_by("regimen_cui")
            .agg([
                pl.col("variant_cui").n_unique().alias("n_variant")
            ])
            .select(pl.col("n_variant").sum())
            .item()
        )

        self.logger.info(f"[Lookup] cycle_length_unit / indeterminate variants: {cycle_length_unit_indefinite} ({cycle_length_unit_indefinite_regimens_unique} unique regimen variants)")

    def log_regimen_level_stats(self):
        """Note: Needs to be called early on"""

        unique_regimens = (
            self.s.select("regimen_cui").n_unique()
        )

        self.logger.info(f"[Lookup] Total regimens (unique): {unique_regimens}")


        unique_regimens_per_conditions = (
            self.s.group_by("condition_cui")
            .agg(pl.col("regimen_cui").n_unique().alias("n_regimens"))
            .select(pl.col("n_regimens")).sum()
            .item()
        )

        self.logger.info(f"[Lookup] Total regimens per condition (unique): {unique_regimens_per_conditions}")
    
    def log_variants_level_stats(self,group_keys):
        """"""
        all_keys = self.s.select(group_keys).unique()

        # Check for NaNs/nulls in any group key column
        nan_group_keys = all_keys.filter(
            pl.fold(
                acc=pl.lit(False),
                function=lambda acc, x: acc | x.is_null(),
                exprs=[pl.col(c) for c in group_keys]
            )
        )

        if nan_group_keys.height > 0:
            self.logger.error(f"[GROUP_KEYS] Found {nan_group_keys.height} groups with NaNs:")
            self.logger.debug(nan_group_keys)
            raise ValueError("Invalid group_keys: NaNs present in condition/regimen/variant key set")

        all_keys_regimens_unique = (
            all_keys
            .select(["regimen_cui", "variant_cui"])
            .unique()
            .group_by("regimen_cui")
            .agg(pl.col("variant_cui").n_unique().alias("n_variant"))
            .select(pl.col("n_variant").sum())
            .item()
        )

        self.logger.info(f"[Lookup] Total variant: {all_keys.shape[0]} ({all_keys_regimens_unique} unique regimen variants)")

        s_with_dup = self.s.with_columns([
            pl.col("component")
            .is_duplicated()
            .over(group_keys)
            .alias("is_duplicated")
        ])

        #
        #   Multi-parted variants grouping
        #
        mask_renamed = (
            s_with_dup           # Clean + duplicate tags
            .group_by(group_keys)
            .agg([
                pl.col("is_duplicated").any().alias("is_multipart")
            ]) # Creates struct data type, column w multi-fileds object
            .with_columns(      # set value names
            pl.when(pl.col("is_multipart"))
            .then(pl.lit("Multi-part Sig")).otherwise(pl.lit("Single-part Sig"))
            .alias("sig_type")
            )
        )
        
        multi_count = (
            mask_renamed
            .group_by("sig_type")
            .agg(pl.count())  # This counts unique group_keys per sig_type (since mask already grouped by group_keys)
            .filter(pl.col("sig_type") == "Multi-part Sig").select("count").item()
        )

        single_count = (
            mask_renamed
            .group_by("sig_type")
            .agg(pl.count())  # This counts unique group_keys per sig_type (since mask already grouped by group_keys)
            .filter(pl.col("sig_type") == "Single-part Sig").select("count").item()
        )

        regimen_variant_sums = (
            mask_renamed
            .group_by(["sig_type", "regimen_cui"])
            .agg(pl.col("variant_cui").n_unique().alias("n_variants"))
            .group_by("sig_type")
            .agg(pl.col("n_variants").sum().alias("total_unique_variants"))
        )

        multi_variant_count = (
            regimen_variant_sums
            .filter(pl.col("sig_type") == "Multi-part Sig")
            .select("total_unique_variants")
            .item()
        )

        single_variant_count = (
            regimen_variant_sums
            .filter(pl.col("sig_type") == "Single-part Sig")
            .select("total_unique_variants")
            .item()
        )

        self.logger.info(f"[Lookup] Multi-part / Single-part variants: {multi_count} ({multi_variant_count} unique regimen variants) / {single_count} ({single_variant_count} unique regimen variants)")

        assert single_variant_count + multi_variant_count == all_keys_regimens_unique

    def handle_processing(self, group_keys, sigs_anatomy_essentials):
        """
            Find Multi-part sig group keys. Split Multi part and single part sigs.
            Process single part as standard. Combine filtered single part and multi part as funny
            Log stats for funny and standard regimnes.
        """

        all_keys = self.s.select(group_keys).unique()
        
        all_keys_regimens_unique = (
            all_keys
            .select(["regimen_cui", "variant_cui"])
            .unique()
            .group_by("regimen_cui")
            .agg(pl.col("variant_cui").n_unique().alias("n_variant"))
            .select(pl.col("n_variant").sum())
            .item()
        )

        s_with_dup = self.s.with_columns([
            pl.col("component")
            .is_duplicated()
            .over(group_keys)
            .alias("is_duplicated")
        ])

        multipart_keys = (
            s_with_dup           
            .group_by(group_keys)
            .agg([
                pl.col("is_duplicated").any().alias("is_multipart")
            ])
            .filter(pl.col("is_multipart"))
            .select(group_keys)
        )
        
        multipart_df = self.s.join(multipart_keys, on=group_keys, how="inner")
        
        singlepart_keys = all_keys.join(
            multipart_keys,
            on=group_keys,
            how="anti"
        )

        singlepart_df = self.s.join(
            singlepart_keys,
            on=group_keys,
            how="inner"
        )

        ######################### Cleaning single-parted sigs ###################################
        tracked_all_days_pattern = r"-\d+|\d+\|\d+|\d+~\d+|\(.*?\)"

        # has Nans in non-nan mandatory fields
        singlepart_df = singlepart_df.with_columns([
            pl.fold(
                acc=pl.lit(False),
                function=lambda acc, x: acc | x.is_null(),
                exprs=[pl.col(c) for c in sigs_anatomy_essentials],
            ).alias("has_null_in_sig")
        ])

        valid_group_ids = (
            singlepart_df
            .group_by(*group_keys)
            .agg([
                
                (~pl.col("has_null_in_sig")).all().alias("non_null_fields"),

                ( # has unhandled pattern in allDays
                    ~pl.col("allDays")
                    .cast(pl.Utf8)
                    .str.contains(tracked_all_days_pattern, literal=False)
                ).all().alias("no_allDays_pattern"),

            ])
            .filter( 
                pl.col("non_null_fields")
                & pl.col("no_allDays_pattern")
            )
            .select(group_keys)
        )

        
        # Valid groups output
        self.s = singlepart_df.join(valid_group_ids, on=group_keys, how="inner").drop("has_null_in_sig")

        # Double check the leaks...
        leaks = self.s.filter(
            pl.col("allDays").cast(pl.Utf8).str.contains(tracked_all_days_pattern, literal=False)
        )

        if leaks.height > 0:
            self.logger.error(f"[LEAK] {leaks.height} rows still match invalid allDays pattern!")
            # self.logger.debug(leaks.select(group_keys + ["allDays"]))
            raise RuntimeError("Group filter failed — bad allDays pattern leaked post-filter.")
        
        ######################### Logging ###################################
        standard = self.s.select(group_keys).unique()
        funny_singlepart = singlepart_df.join(standard, on=group_keys, how='anti').drop("has_null_in_sig")
        funny = pl.concat([funny_singlepart, multipart_df]) # defining filtered and multiparted sigs as funny
        funny_unique = funny.select(group_keys).n_unique()

        standard_regimens_unique = ( 
            standard
            .select(["regimen_cui", "variant_cui"])
            .unique()  # filters duplicated reg - var
            .group_by("regimen_cui")
            .agg(pl.col("variant_cui").n_unique().alias("n_variant"))
            .select(pl.col("n_variant").sum())
            .item()
        )

        funny_regimens_unique = ( 
            funny
            .select(["regimen_cui", "variant_cui"])
            .unique()
            .group_by("regimen_cui")
            .agg(pl.col("variant_cui").n_unique().alias("n_variant"))
            .select(pl.col("n_variant").sum())
            .item()
        )

        assert all_keys_regimens_unique == standard_regimens_unique + funny_regimens_unique, "Mismatch in group splits!"
        
        self.logger.info(f"[Lookup] Standrad variants: {standard.shape[0]} ({standard_regimens_unique} unique regimen variants)")
        self.logger.info(f"[Lookup] Funny variants: {funny_unique} ({funny_regimens_unique} unique regimen variants)")

       


def pre_run(
    input_files_name=".",
    output_dir="workdir",
    log_dir= "log_dir"
):
    sigs_path = f"{input_files_name}"

    print("[INFO] Starting preprocessing run...")
    r = Runner(
        sigs_path=sigs_path,
        log_dir=log_dir,
    )
    dp = r.s[:]

    dp.write_parquet(f"{output_dir}/s_frame.parquet") 

    print("[INFO] Output files written.")

if __name__ == "__main__":
    pre_run()