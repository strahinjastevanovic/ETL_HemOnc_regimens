import logging
import polars as pl
import os 
from tqdm import tqdm
from itertools import combinations
import json
import re

class Logger:
    def __init__(self, log_dir, filename="PRE.processing.log", level="DEBUG"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, filename)

        self.logger = logging.getLogger(filename)
        self.logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
        self.logger.propagate = False

        if not self.logger.handlers:
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler = logging.FileHandler(self.log_path, mode="w")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, msg): self.logger.info(msg)
    def debug(self, msg): self.logger.debug(msg)
    def warning(self, msg): self.logger.warning(msg)
    def error(self, msg): self.logger.error(msg)
    def critical(self, msg): self.logger.critical(msg)

class AuditColumnTypes:
    def __init__(self, log_dir, filename):
        self.logger = Logger(log_dir, filename)
        self.schema_json_path = f"{log_dir}/s_frame.schema.json"

    def audit(self, data_path: str, sample_rows: int = 10000, int_threshold: float = 0.8):
        df_sample = pl.read_csv(data_path, infer_schema_length=sample_rows)
        field_dtypes = []
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
            field_dtypes.append(f"{col}\t{int_ratio:.2f}\t{dtype}")
            schema[col] = dtype

        # Log types
        self.logger.info("column\tint_ratio\ttype_inferred\n")
        self.logger.info("\n".join(field_dtypes))
        # Save schema
        with open(self.schema_json_path, "w") as f:
            json.dump(schema, f, indent=2)

class Frame:
    def load_data(self, sigs_path):
        """
        Load data with all columns forced to Utf8 to ensure resilience.
        Audit types immediately upon instantiation.
        """
        schema = pl.read_csv(sigs_path, infer_schema_length=10000).schema
        utf8_schema = {col: pl.Utf8 for col in schema} # overriding all types!

        frame = pl.read_csv(
            sigs_path,
            schema_overrides=utf8_schema,
            null_values=["null", "NA", "NUB", "", "-"]
        )
        return frame
 
class Reporter:
    def __init__(self, output):
        self.output = output
        os.makedirs(self.output, exist_ok=True)
    def to_tsv(self, frame, file_name):
        frame.write_csv(f"{self.output}/{file_name}.tsv", separator="\t")

class NullValueHandlers:
    def __init__(self, logger:object, reporter:object):
        self.logger = logger
        self.reporter = reporter
            
    def handle_nan_in_group_keys(self, frame, group_keys): 
        """Clean NaNs in group keys, removes groups with NaN keys"""

        pre_drop_groups  = frame.select(group_keys + ["regimen"]).unique()
        frame            = frame.drop_nulls(group_keys)  
        post_drop_groups = frame.select(group_keys).unique()

        # find dropped groups 
        dropped_groups_df = pre_drop_groups.join(
            post_drop_groups, on=group_keys, how="anti")
        
        num_dropped_groups = dropped_groups_df.height
        
        self.logger.info(f"[REPORT] Removed {num_dropped_groups} groups due to nulls in group keys: \n{group_keys}\n")
        self.reporter.to_tsv(dropped_groups_df, "null_group_keys")
        return frame

    def handle_nan_in_condition(self):
        # TODO: Nan condition_cui fallback 
        # self.s = self.s.with_columns(
        #     pl.col("condition_cui").fill_null("undefined"),
        #     pl.col("condition").fill_null("undefined")
        # )
        pass
   
    def handle_null_in_sigs(self, frame, fields, group_keys):
        """Filter out entire variants where any record has nulls in essential fields"""

        # Tag rows with null
        frame = frame.with_columns([
            pl.fold(
                acc=pl.lit(False),
                function=lambda acc, x: acc | x.is_null(),
                exprs=[pl.col(c) for c in fields],
            ).alias("has_null_in_sig")
        ])

        # Find (regimen_cui, variant_cui) groups with any nulls
        variant_with_nulls = (
            frame
            .group_by(group_keys)
            .agg(pl.col("has_null_in_sig").any().alias("group_has_null"))
            .filter(pl.col("group_has_null"))
            .select(group_keys)
        )

        # Join back to get full rows of affected variants
        null_sig_df = frame.join(variant_with_nulls, on=group_keys, how="inner")

        self.logger.info(f"[REPORT] Variants with NULLs in essentials — count: {variant_with_nulls.height}")
        self.reporter.to_tsv(null_sig_df, "null_sig")

        # Drop all rows from affected variants
        frame = frame.join(variant_with_nulls, on=group_keys, how="anti").drop("has_null_in_sig")

        return frame


class RegimenHandler:
    def __init__(self, logger:object, reporter:object):
        self.logger = logger
        self.reporter = reporter
            
    def log_regimen_level_stats(self, frame):
        """Log stats on full unfiltered data"""

        unique_regimens = frame.select("regimen_cui").n_unique()
        self.logger.info(f"[REPORT] Total regimens (unique) - before filtering: {unique_regimens}")

        unique_regimens_per_conditions = (
            frame.group_by("condition_cui")
            .agg(pl.col("regimen_cui").n_unique().alias("n_regimens"))
            .select(pl.col("n_regimens")).sum().item()
        )
        self.logger.info(f"[REPORT] Total regimens per condition (unique): {unique_regimens_per_conditions}")

        group_keys_w_cui = ["condition", "condition_cui", "regimen", "regimen_cui", "variant", "variant_cui"]
        unique_groups = (
            frame.select(group_keys_w_cui)
                .unique()
        )
        self.reporter.to_tsv(unique_groups, "unique_groups")

    def filter_rt(self, frame):
        """Handle RadioTherapy-containing regimens"""

        rt_pattern = r"(?:^|[\s,/])\(?(?:RT|SCRT|CSRT|WBRT|WB-XRT)\)?(?:[\s,)\-]|$)"
        filtered_rt = (
            frame.filter(
                pl.col("regimen").cast(pl.Utf8).str.contains(rt_pattern, literal=False) |
                (pl.col("regimen") == "Whole brain irradiation")
            )
        )
        rt_match = (
            filtered_rt
            .select("regimen")
            .unique()
        )
        frame = frame.filter(
            ~pl.col("regimen").cast(pl.Utf8).str.contains(rt_pattern, literal=False)
        )
        
        # Log RT-containing stats
        rt_count    = rt_match.height
        self.logger.info(f"[REPORT] Regimens containing RT (spaced/parenthesized): {rt_count}")
        self.reporter.to_tsv(filtered_rt, "radiotherapy_containing")
        
        return frame
    
    def filter_imbalanced(self, frame: pl.DataFrame, group_keys, report_name="regimen_components_nonequal"):
        """
        Logs out groups where components are not equal across variants within the group.
        Logs and reports the dropped groups regimens.
        """
        group_keys_regimen = group_keys[:2]

        bad_groups = (
            frame.group_by(group_keys)
            .agg(pl.col("component").n_unique().alias("component_count"))
            .group_by(group_keys_regimen)
            .agg(pl.col("component_count").n_unique().alias("count_of_component_counts"))
            .filter(pl.col("count_of_component_counts") > 1)
            .select(group_keys_regimen)
        )

        frame_bad = frame.join(bad_groups, on=group_keys_regimen, how="inner")
        frame_good = frame.join(bad_groups, on=group_keys_regimen, how="anti")

        frame_bad_height = frame_bad.select("regimen").unique().height

        self.logger.info(f"[REPORT] Removed regimens due to inconsistent components number cross variants: {frame_bad_height}")
        self.reporter.to_tsv(frame_bad, report_name)
        print(frame_bad.columns)

        variant_component_counts = (
            frame_bad.group_by(group_keys)
            .agg([
                pl.col("component").n_unique().alias("component_count_in_this_variant"),
                pl.col("component").unique().alias("components_unsorted")
            ])
            .with_columns(
                pl.col('components_unsorted').list.sort().alias("components")
            )
            .drop("components_unsorted")
        )

        variant_component_report = (
            variant_component_counts.join(
                frame_bad.select(group_keys + ['regimen']).unique(),
                on=group_keys,
                how="left"
            )
            .select([
                "regimen", 
                "variant_cui", 
                "component_count_in_this_variant", 
                "components"
            ])
            .with_columns(
                pl.col("components").list.join(", ").alias("components")  # Stringify 
            )
            .unique(subset=["regimen", "variant_cui", "components"])  # Dedup
            .sort(["regimen", "variant_cui"])
        )
        self.reporter.to_tsv(variant_component_report, f"{report_name}_variants")
        return frame_good


class VariantHandler:
    def __init__(self, logger:object, reporter:object):
        self.logger = logger
        self.reporter = reporter

    def create_checkopoint_frame(self, frame, group_keys):
        """Create checkpoint of variant groups (to verify nothing is lost)"""
        return frame.select(group_keys).unique()

    def handle_partial_variants(self, frame, group_keys):
        """Logs variant and parts status"""

        # Total variants and unique regimen-variant pairs
        uniq = frame.select(group_keys).unique()
        n_variants = uniq.height
        regimen_variants = (
            uniq.select(["regimen_cui", "variant_cui"])
                .unique()
                .group_by("regimen_cui")
                .agg(pl.col("variant_cui").n_unique().alias("n_variant"))
        )

        self.reporter.to_tsv(regimen_variants, "regimen_variants_n_unique")

        n_regimen_variants = (
            regimen_variants
                .select(pl.col("n_variant").sum())
                .item()
        )
        self.logger.info(f"[REPORT] Total number of variants: {n_variants} ({n_regimen_variants} unique)")

        # Detect multipart
        sig_types = (
            frame.with_columns(pl.col("component").is_duplicated().over(group_keys).alias("dup"))
                .group_by(group_keys)
                .agg(pl.col("dup").any().alias("is_multi"))
                .with_columns(pl.when("is_multi").then(pl.lit("Multi-part Sig")).otherwise(pl.lit("Single-part Sig")).alias("sig_type"))
        )

        s = frame.join(sig_types, on=group_keys, how="left")

        # Counts
        counts = (
            s.group_by("sig_type")
            .agg([
                pl.count().alias("count"),
                pl.col("regimen").n_unique().alias("n_regimens"),
                pl.col("variant_cui").n_unique().alias("n_variants"),
                pl.col("regimen").unique().alias("regimens")
            ])
        )

        for row in counts.iter_rows(named=True):
            self.logger.info(f"[REPORT] Number of total {row['sig_type']} variants: {row['count']} ({row['n_variants']} unique)")

        assert counts["n_variants"].sum() == n_regimen_variants

        multi_part_df = s.filter(pl.col("sig_type") == "Multi-part Sig") 
        single_part_df = s.filter(pl.col("sig_type") == "Single-part Sig")

        self.reporter.to_tsv(multi_part_df, "multi_part_sigs")
        self.reporter.to_tsv(single_part_df, "single_part_sigs")

        # cleaning
        single_part_df = single_part_df.drop("sig_type")  
        multi_part_df = multi_part_df.drop("sig_type")
        return single_part_df, multi_part_df

class PatternHandlers:
    def __init__(self, logger:object, reporter:object):
        self.logger = logger 
        self.reporter = reporter
    
    def log_indefinite_cycle_length(self, frame, group_keys):
        """checks cycle_length for indeterminate - indefinite value handler"""

        groups_with_indeterminate = (
            frame
            .group_by(group_keys)
            .agg([
                (pl.col("cycle_length_unit") == "indeterminate").any().alias("has_indeterminate")
            ])
            .filter(pl.col("has_indeterminate"))
        )

        cycle_length_unit_indefinite = (
            groups_with_indeterminate
            .shape[0]
        )

        cycle_length_unit_indefinite_df = frame.join(groups_with_indeterminate, on=group_keys, how="inner")
        self.reporter.to_tsv(cycle_length_unit_indefinite_df, "cycle_length_unit_indefinite")

        cycle_length_unit_indefinite_regimens_unique = (
            frame
            .filter(pl.col("cycle_length_unit") == "indeterminate")
            .group_by("regimen_cui")
            .agg([
                pl.col("variant_cui").n_unique().alias("n_variant")
            ])
            .select(pl.col("n_variant").sum())
            .item()
        )

        self.logger.info(f"[REPORT] Number of variants with indefinite cycles: {cycle_length_unit_indefinite} ({cycle_length_unit_indefinite_regimens_unique} unique)")

    def log_cycle_length_indefinite(self,fields): # TODO
        """Logs cases where (+c or c) exist in cycle_length_ub or cycle_length_lb"""
        pass 
    
    def log_field_indefinite(self, fields): # TODO
        """Logs cases where allDays (+c, +n)  or timing_sequence (+c, +n, +1, +2) exist in <fields>"""
        pass
    
    def filter_optional_from_fields(self,frame, group_keys):
        """Log cases where (\d) exist in timing_sequence or allDays
        cleans inplace
        """ 
        tracked_pattern = r"\(.*?\)"
        valid_group_ids = (
            frame.group_by(*group_keys)
                .agg((~pl.col("allDays").cast(pl.Utf8)
                    .str.contains(tracked_pattern, literal=False)
                ).all().alias("no_pattern"))
                .filter(pl.col("no_pattern"))
                .select(group_keys)
        )
        # Extract unique regimen after patterns filtering
        # Anti-join to get invalid groups (with patterns in Field)
        valid_groups = frame.join(valid_group_ids, on=group_keys, how="inner")
        invalid_groups = frame.join(valid_group_ids, on=group_keys, how="anti")
        invalid_groups_keys = invalid_groups.select(group_keys).unique().height

        self.logger.info(f"[REPORT] Groups WITH Optional — groups: {invalid_groups_keys}")
        self.reporter.to_tsv(invalid_groups, "with_optional_pattern")
    
        # Double check the leaks...
        leaks = valid_groups.filter(
            pl.col("allDays").cast(pl.Utf8).str.contains(tracked_pattern, literal=False)
        )

        if leaks.height > 0:
            self.logger.error(f"[LEAK] {leaks.height} rows still match invalid pattern!")
            raise RuntimeError("Group filter failed — bad field pattern leaked post-filter.")
        
        return valid_groups, invalid_groups

    # Filters by variant
    def all_days_pattern_handler(self, frame, group_keys): 
        """Handled pattern in allDays field""" 
        tracked_all_days_pattern = r"-\d+|\d+\|\d+|\d+~\d+|\(.*?\)|0"
        valid_group_ids = (
            frame.group_by(*group_keys)
                .agg((~pl.col("allDays").cast(pl.Utf8)
                    .str.contains(tracked_all_days_pattern, literal=False)
                ).all().alias("no_pattern"))
                .filter(pl.col("no_pattern"))
                .select(group_keys)
        )
        
        valid_groups = frame.join(valid_group_ids, on=group_keys, how="inner")
        invalid_groups = frame.join(valid_group_ids, on=group_keys, how="anti")
        invalid_groups_keys = invalid_groups.select(group_keys).unique().height

        self.logger.info(f"[REPORT] Number of variants with unhandled allDays pattern: {invalid_groups_keys}")
        self.reporter.to_tsv(invalid_groups, "with_allDays_pattern")
       
        # Safeguard
        leaks = valid_groups.filter(
            pl.col("allDays").cast(pl.Utf8).str.contains(tracked_all_days_pattern, literal=False)
        )
        if leaks.height > 0:
            self.logger.error(f"[LEAK] {leaks.height} rows still match invalid allDays pattern!")
            raise RuntimeError("Group filter failed — bad allDays pattern leaked post-filter.")
        
        return valid_groups, invalid_groups
    
    def from_to_by_pattern_handler(self, frame, group_keys, fields=["timing_sequence",'allDays']): 
        """Handled pattern in custom field""" 
        tracked_pattern = r"\[.*\]"

        condition = pl.fold(
            acc=pl.lit(True),
            function=lambda acc, col: acc & (~col.cast(pl.Utf8).str.contains(tracked_pattern, literal=False)),
            exprs=[pl.col(f) for f in fields]
        )
        valid_group_ids = (
            frame.group_by(*group_keys)
            .agg(condition.all().alias("no_pattern"))
            .filter(pl.col("no_pattern"))
            .select(group_keys)
        )
        
        valid_groups = frame.join(valid_group_ids, on=group_keys, how="inner")
        invalid_groups = frame.join(valid_group_ids, on=group_keys, how="anti")
        invalid_groups_count = invalid_groups.select(group_keys).unique().height

        self.logger.info(f"[REPORT] Number of variants with unhandled from_to_by pattern: {invalid_groups_count}")
        self.reporter.to_tsv(invalid_groups, "with_FromToBy_pattern")

        # Step 3: Safeguard: double-check that no valid_groups rows leak the pattern in any field
        leaks = pl.concat([
            valid_groups.filter(pl.col(f).cast(pl.Utf8).str.contains(tracked_pattern, literal=False))
            for f in fields
        ])

        if leaks.height > 0:
            self.logger.error(f"[LEAK] {leaks.height} rows still match invalid {fields} pattern!")
            raise RuntimeError("Group filter failed — bad pattern leaked post-filter.")

        return valid_groups, invalid_groups

class SupplementaryHandler:
    def __init__(self, logger:object, reporter:object):
        self.logger = logger
        self.reporter = reporter

    @staticmethod
    def clean_text(text: str) -> str:
        return re.sub(r"[^\w\s]", "", str(text)).strip().lower()

    @staticmethod
    def proc_blist_naive(blist: list) -> list:
        blist = [l for subli in [s.split("(") for s in blist] for l in subli]
        return [s.strip(")").strip().lower() for s in blist]

    def clean_components(self, frame, supplementary=None):
        """
        Removes blacklisted components; 
        Will drop variant_cui if all its components are blacklisted.
        Return invalid - variant_cui droped, valid - passed after cleanup

        Creates metaCondition - side effect TODO...
        """

        self.logger.info(f"Input shape: {frame.shape}")
        frame = frame.with_columns(
            pl.col("component").cast(str).map_elements(self.clean_text).alias("meta_component")
        )

        blacklist_set = set()
        if supplementary:
            bl_json = json.load(open(supplementary))
            blacklist_set = set(self.proc_blist_naive(bl_json.get("custom_sigs_curated", [])))

        frame = frame.with_columns(
            pl.col("meta_component").is_in(blacklist_set).alias("is_blacklisted")
        )

        # Count how many components are blacklisted
        if blacklist_set:
            n_drop = frame.filter("is_blacklisted").height
            dropped = frame.filter("is_blacklisted").select("component").unique().to_series().to_list()
            self.logger.info(f"[REPORT] Dropped {n_drop} components from blacklist ({len(blacklist_set)} items).")
            for comp in dropped:
                self.logger.info(f"[Lookup] DROPPED: {comp}")
        else:
            self.logger.info("No blacklist applied.")

        # Drop blacklisted components
        frame_clean = frame.filter(~pl.col("is_blacklisted"))

        # Determine dropped variant_cui group keys
        valid_keys = frame_clean.select(["regimen_cui", "variant_cui"]).unique()
        valid = frame_clean.drop(["is_blacklisted", "meta_component"])
        invalid = frame.join(valid_keys, on=["regimen_cui", "variant_cui"], how="anti").drop(["is_blacklisted", "meta_component"])
        
        self.logger.info(f"Output shape: {valid.shape}")
        self.reporter.to_tsv(invalid, "supplementary_dropped")
        
        n_dropped = invalid.select(["regimen_cui", "variant_cui"]).unique().height
        self.logger.info(f"[REPORT] Removed variants as supplementary - {n_dropped}")
        
        return valid, invalid

    def clean_by_role(self, frame, field="component_role"):
        dropped = frame.filter(pl.col(field).is_in(['secondary systemic', 'locoregional']))
        dropped_components_count = dropped.select("component").n_unique()
        self.reporter.to_tsv(dropped, "component_role_secondary")
        self.logger.info(f"[REPORT] Removed supplementary records kept variants in groups: {round(dropped.shape[0] / frame.shape[0], 2)}% - Components loss number: {dropped_components_count}")
        filtered = frame.join(dropped, on=frame.columns, how="anti")
        return filtered

class Sumstats:
    def __init__(self, logger:object):
        self.logger = logger

    def concat_with_overlap_diagonstics(self, subsets:list, group_keys):
        overlap = lambda *args: (
            lambda a, b, name: (
                lambda o: self.logger.debug(f"[OVERLAP] {o.height} overlapping group_keys in {name}")
                if o.height > 0 else None
            )(a.select(group_keys).unique().join(b.select(group_keys).unique(), on=group_keys, how="inner"))
        )(*args)

        if len(subsets) > 1:
            pairs = [((a[0], a[1]), (b[0], b[1])) for a, b in combinations(subsets, 2)]
            for pair in pairs:
                overlap(pair[0][1], pair[1][1], f"{pair[0][0]} vs {pair[1][0]}")
        return pl.concat([subset[1] for subset in subsets])
    
    def log_summary(self, standard, funny, all_keys, group_keys):
        all_keys_regimens_unique = (
            all_keys
            .select(["regimen_cui", "variant_cui"])
            .unique()  # filters duplicated variants
            .group_by("regimen_cui")
            .agg(pl.col("variant_cui").n_unique().alias("n_variant"))
            .select(pl.col("n_variant").sum())
            .item()
        )

        standard_regimens_unique = ( 
            standard
            .select(["regimen_cui", "variant_cui"])
            .unique() 
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
        
        funny_unique = funny.select(group_keys).n_unique()
        
        assert all_keys_regimens_unique == standard_regimens_unique + funny_regimens_unique, f"Mismatch in group splits! {all_keys_regimens_unique} != {standard_regimens_unique} + {funny_regimens_unique}"
        
        self.logger.info(f"[REPORT] Number of vanilla variants: {standard.shape[0]} ({standard_regimens_unique} unique)")
        self.logger.info(f"[REPORT] Number of funny variants: {funny_unique} ({funny_regimens_unique} unique)")



class Preprocessor:
    def __init__(self, sigs_path, output_dir, log_dir=".", supplementary_file=None,):
       
        self.logger     = Logger(log_dir, )
        self.audits     = AuditColumnTypes(log_dir, "PRE.audit.log")
        self.audits.audit(sigs_path)
        self.reporter   = Reporter(f"{output_dir}/report_tables") 

        self.s          = Frame().load_data(sigs_path)
        self.sf         = supplementary_file
        self.group_keys = ["condition_cui", "regimen_cui", "variant_cui"]
        self.sigs_anatomy_essentials = [
            'variant_cui', 
            'allDays',
            "cycle_length_lb",
            "cycle_length_ub",   
            "cycle_length_unit",
            "timing_sequence"
        ]

    def initialize(self, ):
        self.null_handlers       = NullValueHandlers(self.logger, self.reporter)
        self.regimen_handler     = RegimenHandler(self.logger, self.reporter)
        self.variant_handler     = VariantHandler(self.logger, self.reporter)
        self.pattern_handlers    = PatternHandlers(self.logger, self.reporter)
        self.supp_handler        = SupplementaryHandler(self.logger, self.reporter)
        self.sumstats            = Sumstats(self.logger)
        return self # enables chaining


    def run(self):
        """Process the data (clean NaN and handle duplicates)."""

        frame = self.s.clone()
        group_keys = self.group_keys
        fields = self.sigs_anatomy_essentials
        # supplementary_file = self.sf

        # ----------- 1 -- 1st level subset block -component level dropouts, variants kept ------------
        frame = self.supp_handler.clean_by_role(frame) 

        # ----------- 2 2nd  level subset block -regimen level dropouts ------------
        frame = self.null_handlers.handle_nan_in_group_keys(frame, group_keys)
        frame = self.null_handlers.handle_null_in_sigs(frame, fields, group_keys)
        self.regimen_handler.log_regimen_level_stats(frame)
        frame = self.regimen_handler.filter_imbalanced(frame, group_keys)
        frame = self.regimen_handler.filter_rt(frame)
        self.pattern_handlers.log_indefinite_cycle_length(frame, fields)
        # ----------- 3 - 3nd level subset block -variant level drouputs ------------
        checkpoint_df = self.variant_handler.create_checkopoint_frame(frame, group_keys) # safeguard
        single_df, multi_df = self.variant_handler.handle_partial_variants(frame, group_keys)
        valid_df, invalid_df_1 = self.pattern_handlers.all_days_pattern_handler(single_df, group_keys)
        cleaned_df, invalid_df_2 = self.pattern_handlers.from_to_by_pattern_handler(valid_df, group_keys)
        
        # # ----------- 4 - 3rd level subset block -component level dropouts, variants kept ------------
        # cleaned_df = self.supp_handler.clean_by_role(valid_df) 
        # # ----------- 5 - rejoin filtered -----------
        funny_df = self.sumstats.concat_with_overlap_diagonstics(subsets=[("invalid_1", invalid_df_1), ("invalid_2", invalid_df_2), ("multi", multi_df)], group_keys=group_keys)
        # ------------ 6 - logs + reports -----------
        self.sumstats.log_summary(cleaned_df, funny_df, checkpoint_df, group_keys)
        self.reporter.to_tsv(cleaned_df, "preproc_cleaned")
        # ---- << final frame >> ----
        self.processed = cleaned_df.clone()
        return self

    def get_processed(self):
        """Returns the final cleaned frame."""
        if not hasattr(self, "processed"):
            raise RuntimeError("Data has not been processed yet.")
        return self.processed
    
def preprocessing(
    sigs_file=".",
    output_dir="workdir",
    log_dir= "log_dir",
    supplementary_file="."
):

    print("[INFO] Starting preprocessing run...")
    proc = Preprocessor(
        sigs_path=sigs_file,
        output_dir=output_dir,
        log_dir=log_dir,
        supplementary_file=supplementary_file,
    ).initialize().run()

    dp = proc.get_processed()

    dp.write_parquet(f"{output_dir}/s_frame.parquet") 
    dp.write_csv(f"{output_dir}/s_frame.tsv", separator="\t") 

    print("[INFO] Output files written.")

if __name__ == "__main__":
    preprocessing()