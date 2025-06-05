import logging
import polars as pl
import os 
from tqdm import tqdm
import json
import re

# TODO: cleanify
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    full_handler = logging.FileHandler(f"{log_dir}/PRE.processing.log", mode='w')
    full_handler.setFormatter(formatter)
    logger.addHandler(full_handler)
    return logger

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
        self.logger = setup_logging(log_dir)
        self.schema_log_path = f"{log_dir}/PRE.types-audit.log"
        self.schema_json_path = f"{log_dir}/s_frame.schema.json"
        
        # Load data and process immediately upon instantiation
        audit_column_types(self.sigs_path, self.schema_log_path, self.schema_json_path)
        self._load_data()
        self._process_data()

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

    def clean_nan_columns(self, df, drop_na_cols=None):
        """Clean NaN values in specified columns."""
        assert drop_na_cols is not None
        assert all(col in df.columns for col in drop_na_cols)

        self.logger.info("---- CLEANING NAN RECORDS ----")

        for col in drop_na_cols:
            before = df.shape

            # Group by variant keys to detect partial nulls
            cols = ['regimen_cui', 'variant_cui', col] if col != "variant_cui" else ['regimen_cui', "variant_cui"]
            group = df.select(cols)

            # Count total rows and null rows per variant group
            stats = (
                group
                .group_by(["regimen_cui", "variant_cui"])
                .agg([
                    pl.count().alias("total"),
                    pl.col(col).is_null().sum().alias("null_count")
                ])
                .with_columns([
                    (pl.col("null_count") > 0).alias("has_null"),
                    (pl.col("null_count") < pl.col("total")).alias("has_non_null")
                ])
            )

            # Detect cases where column is partially null (within group)
            partial_drop = stats.filter(pl.col("has_null") & pl.col("has_non_null")).shape[0]
            total_groups = stats.shape[0]

            self.logger.warning(f"[CLEAN] Column '{col}': dropping rows from {partial_drop} partially-null variants (out of {total_groups} total variants).")

            # Drop actual nulls
            df = df.filter(pl.col(col).is_not_null())
            after = df.shape
            self.logger.info(f"[CLEAN] Dropped nulls in '{col}': {before} → {after}")
            
            print(f"[CLEAN] Dropped nulls in '{col}': {before} → {after}")

        return df

    def clean_group_duplicated_components_and_log(self, df):
        df = df.with_row_count("row_idx")  # Add index for safe row deletion
        rows_to_remove = []
        # Iterate through groups of regimen_cui + variant_cui
        for group_id, group_df in tqdm(df.group_by(['regimen_cui', 'variant_cui', 'condition_cui']), desc="Cleaning duplicate components"):
            # Check for duplicates within the group by 'component'
            regimen_cui, variant_cui, condition_cui = group_id
            duplicate_mask = group_df.select('component').to_series().is_duplicated()
            if not duplicate_mask.any():
                continue

            dups = group_df.filter(duplicate_mask)
            for component, comp_df in dups.group_by('component'):
                if comp_df.height <= 1:
                    continue
                
                self.logger.info("---------------------------------------------------")
                # change since 2024!!! multiple regimen names for the same reg_cui 
                # self.logger.warning(f"Regimen: {group_df.select('regimen').unique().item()}. Variant: {group_df.select('variant').unique().item()}")
                self.logger.warning(
                    f"Regimen: {'|'.join(group_df.select('regimen').unique().to_series().to_list())}. "
                    f"Variant: {'|'.join(group_df.select('variant').unique().to_series().to_list())}"
                )
                self.logger.warning(f"RegimenCUI: {regimen_cui}. VariantCUI: {variant_cui}. ConditionCUI: {condition_cui}")
                self.logger.warning(f"Component: {component}. # Duplicates: {comp_df.shape[0]}")

                base_row = comp_df[0]
                for i in range(1, comp_df.height):
                    diffs = []
                    for col in comp_df.columns:
                        if col == "row_idx":
                            continue
                        base_val = base_row[col].item()
                        curr_val = comp_df[i][col].item()
                        if base_val != curr_val:
                            diffs.append((col, base_val, curr_val))

                    for col, prev, curr in diffs:
                        self.logger.warning(f"Duplicate_{i+1} different in '{col}': {prev} != {curr}")


                # Track row indices to remove
                rows_to_remove.extend(comp_df["row_idx"].to_list())

        # Remove duplicated rows by index
        df_cleaned = df.filter(~pl.col("row_idx").is_in(rows_to_remove)).drop("row_idx")
        print(f"[CLEAN] Removed duplicated components (final shape): {df_cleaned.shape}")
        return df_cleaned


    def clean_sre_anatomy_records(self, df):
        """Each is a sep case 
        allDays 		    -\d, d1 | d2 , \d ~ \d  #TODO: (c+)...skipped/logged at the moment - uncommon
        """
        # timing_sequence	    (\d) (+) (c+)           # TODO: this will not work...

        self.logger.info("---- CLEANING SRE ANATOMY RECORDS ----")

        tracked_all_days_pattern = r"-\d+|\d+\|\d+|\d+~\d+|\(.*?\)"

        # tracked_timing_sequence_pattern = r"\(.*?\)"

        # find offending variant-level pairs instead
        unhandled_pairs = (
            df.filter(
                pl.col("allDays").str.contains(tracked_all_days_pattern, literal=False)
                # | pl.col("timing_sequence").str.contains(tracked_timing_sequence_pattern, literal=False)
            )
            .select("regimen_cui")
            .unique()
        )

        self.logger.warning(f"Dropping unhandled regimen CUIs #: {unhandled_pairs.shape[0]}")

        df_cleaned = df.join(unhandled_pairs, on=["regimen_cui"], how="anti")

        unq_variants = (
            df_cleaned
            .group_by("regimen_cui")
            .agg(pl.col("variant_cui").n_unique().alias("n_unique"))
            .select(pl.col("n_unique").sum())
        )
        
        self.logger.info(f"[CLEAN] No missing SRE anatomy records (final shape): {df_cleaned.shape}")
        self.logger.info(f"[CLEAN] Total regimen variants: {unq_variants}")
        return df_cleaned


    def mock_nan_conditions(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.when(pl.col("condition").is_null())
            .then(pl.lit("condition*") + pl.col("regimen").cast(str))
            .otherwise(pl.col("condition"))
            .alias("condition")
        )


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

        self.s = (
            self.clean_nan_columns(self.s, sigs_anatomy_essentials)         # nan records for important fields droped
                .pipe(self.clean_group_duplicated_components_and_log)       # multi-part sigs droped
                .pipe(self.clean_sre_anatomy_records)                       # custom edge cases droped
                .pipe(self.mock_nan_conditions)                             # mock conditions
        )
        self.logger.info(f"Data processed. Shape: {self.s.shape}")


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

    # dp.to_pandas().to_csv(f"{output_dir}/s_frame.tsv", sep="\t", index=False)
    dp.write_parquet(f"{output_dir}/s_frame.parquet") # safeguarding mixed fields

    print("[INFO] Output files written.")

if __name__ == "__main__":
    pre_run()