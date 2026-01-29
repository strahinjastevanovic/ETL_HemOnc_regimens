import polars as pl
import json
import re


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
            null_values=["null", "NA", "", "-"]
        )
        return frame

class NullValueHandlers:
    def __init__(self, logger:object, reporter:object):
        self.logger = logger
        self.reporter = reporter
            
    def handle_nan_in_group_keys(self, frame, group_keys): 
        """Clean NaNs in group keys, removes groups with NaN keys"""

        pre_drop_groups  = frame.select(group_keys + ["regimen"]).unique() # Integrify of the mapping rule reg_cui -> regimen
        frame            = frame.drop_nulls(group_keys)  
        post_drop_groups = frame.select(group_keys).unique()

        # find dropped groups 
        dropped_groups_df = pre_drop_groups.join(
            post_drop_groups, on=group_keys, how="anti")
        
        num_dropped_groups = dropped_groups_df.height
        
        self.logger.info(f"[REPORT] Removed {num_dropped_groups} groups due to nulls in group keys: \n{group_keys}\n")
        self.reporter.to_tsv(dropped_groups_df, "null_group_keys")
        return frame

    def handle_nan_in_condition(self, frame):
        null_rows = frame.filter(
            pl.col("condition_cui").is_null() | pl.col("condition").is_null()
        )
        num_affected = null_rows.height
        num_unique_regimens = null_rows.select("regimen").unique().height
        unique_regimens = null_rows.select("regimen").unique()
        
        frame = frame.with_columns(
            pl.col("condition_cui").fill_null("undefined"),
            pl.col("condition").fill_null("undefined")
        )
        self.logger.info(f"[REPORT] Filled {num_affected} (unique regimens: {num_unique_regimens} ) records with nulls in condition/condition_cui.")
        self.reporter.report(null_rows, "null_condition", pattern="Missing {c} or out of sync with AthenaDB.", field="{cc}, {c}", status="P")
        self.reporter.to_tsv(unique_regimens, "null_condition_regimens_unique")
        return frame
        
   
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

        # Find (regimen_cui, variant_key) groups with any nulls
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
        self.reporter.report(null_sig_df, "null_sig", pattern="sig-related fields can not be empty. Thes fields are {allDays}, {lb}, {ub}, {unit}, {ts}", field="{allDays}, {lb}, {ub}, {unit}, {ts}", status="N")

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

        group_keys_w_cui = ["condition", "condition_cui", "regimen", "regimen_cui", "variant", "variant_key"]
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
        self.reporter.report(filtered_rt, "radiotherapy_containing", "Radiotherapy used in {r}.", "{r}", "N")
        
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
        self.reporter.report(frame_bad, report_name, "Variants missmatch in component number.", "{c} {cc} {vc}", "N")

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
                "variant_key", 
                "component_count_in_this_variant", 
                "components"
            ])
            .with_columns(
                pl.col("components").list.join(", ").alias("components")  # Stringify 
            )
            .unique(subset=["regimen", "variant_key", "components"])  # Dedup
            .sort(["regimen", "variant_key"])
        )
        self.reporter.to_tsv(variant_component_report, f"{report_name}_variants")
        return frame_good

class VariantHandler:
    def __init__(self, logger:object, reporter:object):
        self.logger = logger
        self.reporter = reporter

    def save_checkpoint(self, frame, group_keys):
        """Create checkpoint of variant groups (to verify nothing is lost)"""
        return frame.select(group_keys).unique()

    def handle_partial(self, frame, group_keys):
        """Logs variant and parts status"""

        # Total variants and unique regimen-variant pairs
        uniq = frame.select(group_keys).unique()
        n_variants = uniq.height
        regimen_variants = (
            uniq.select(["regimen_cui", "variant_key"])
                .unique()
                .group_by("regimen_cui")
                .agg(pl.col("variant_key").n_unique().alias("n_variant"))
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
                pl.col("variant_key").n_unique().alias("n_variants"),
                pl.col("regimen").unique().alias("regimens")
            ])
        )

        for row in counts.iter_rows(named=True):
            self.logger.info(f"[REPORT] Number of total {row['sig_type']} variants: {row['count']} ({row['n_variants']} unique)")

        assert counts["n_variants"].sum() == n_regimen_variants

        multi_part_df = s.filter(pl.col("sig_type") == "Multi-part Sig") 
        single_part_df = s.filter(pl.col("sig_type") == "Single-part Sig")

        self.reporter.report(multi_part_df, "multi_part_sigs", pattern="Sig entry contains multi step variants", field="step_number", status="N")
        self.reporter.report(single_part_df, "single_part_sigs", pattern="Sig with only single step variants. Not idiosyncratic", field="step_number", status="H")

        # cleaning
        single_part_df = single_part_df.drop("sig_type")  
        multi_part_df = multi_part_df.drop("sig_type")
        return single_part_df, multi_part_df
    
class PatternHandlers:
    def __init__(self, logger:object, reporter:object):
        self.logger = logger 
        self.reporter = reporter
    
    def timing_sequence():
        pass

    def cycle_length_indeterminate(self, frame, group_keys):
        """Checks for indeterminate cycle length units and non-numeric bounds"""

        # Detect group keys that contain indeterminate cycle units OR non-numeric cycle length bounds
        pattern = r"[^\d\.]"  # Anything that's not a digit or dot

        # Boolean mask for bad bounds (non-numeric strings)
        bad_bounds_mask = (
            pl.col("cycle_length_lb").str.contains(pattern)
            | pl.col("cycle_length_ub").str.contains(pattern)
        )

        # Main condition: either indeterminate unit or bad bounds
        indefinite_mask = (
            (pl.col("cycle_length_unit") == "indeterminate")
            | bad_bounds_mask
        )

        # Identify affected groups
        groups_with_indeterminate = (
            frame
            .with_columns([
                indefinite_mask.alias("has_indefinite")
            ])
            .group_by(group_keys)
            .agg([
                pl.col("has_indefinite").any().alias("has_indefinite")
            ])
            .filter(pl.col("has_indefinite"))
        )

        # Count of affected groups
        indefinite = groups_with_indeterminate.height

        # Join to get full rows
        indefinite_df = frame.join(groups_with_indeterminate, on=group_keys, how="inner")
        non_indefinite_df = frame.join(groups_with_indeterminate, on=group_keys, how="anti")

        # Reporting
        self.reporter.report(
            indefinite_df,
            "cycle_length_unit_indefinite",
            pattern="Cycle length unit is indeterminate or cycle length bounds are non-numeric.",
            field="{unit}",
            status="P"
        )

        # Count unique regimens with indeterminate or invalid data
        indefinite_regimens_unique = (
            indefinite_df
            .group_by("regimen_cui")
            .agg([
                pl.col("variant_key").n_unique().alias("n_variant")
            ])
            .select(pl.col("n_variant").sum())
            .item()
        )

        # Ensure output has original column structure
        indefinite_df = indefinite_df.select(frame.columns)
        non_indefinite_df = non_indefinite_df.select(frame.columns)

        self.logger.info(
            f"[REPORT] Number of variants with indefinite cycles: {indefinite} ({indefinite_regimens_unique} unique)"
        )

        return indefinite_df, non_indefinite_df

    # Filters by variant
    def all_days_pattern_handler(self, frame, group_keys): 
        """Handled pattern in allDays field""" 
        tracked_all_days_pattern = r"-\d+|\d+\|\d+|\d+~\d+|\(.*?\)|\b0\b"
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
        self.reporter.report(invalid_groups, "with_allDays_pattern",pattern="Negative units, optional elements, starting on 0th day", field="{ad}", status="N")
       
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

    def cyclen_mismatch_regimen_handler(df: pl.DataFrame) -> int:
        numeric_pattern = r"^\d+(\.\d+)?$"

        return pl.concat(
            [
                df.filter(
                    df["cycle_length_lb"].str.contains(numeric_pattern, literal=False)
                    & df["cycle_length_ub"].str.contains(numeric_pattern, literal=False)
                )
                .with_columns(
                    [
                        pl.col("cycle_length_lb").cast(pl.Float64).alias("lb"),
                        pl.col("cycle_length_ub").cast(pl.Float64).alias("ub"),
                    ]
                )
                .filter(pl.col("lb") != pl.col("ub")),
                df.filter(
                    ~df["cycle_length_lb"].str.contains(numeric_pattern, literal=False)
                    & ~df["cycle_length_ub"].str.contains(numeric_pattern, literal=False)
                    & df["cycle_length_lb"].is_not_null()
                    & df["cycle_length_ub"].is_not_null()
                    & (pl.col("cycle_length_lb") != pl.col("cycle_length_ub"))
                ),
            ],
            how="vertical_relaxed",
        )["regimen"].unique().height

        
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

    def clean_by_blacklist(self, frame, supplementary=None, group_keys=None):
        """
        Removes blacklisted components.
        Drops entire variant groups if any of their components are blacklisted.
        Returns: valid (cleaned) frame.
        """
        group_keys_loc = group_keys[1:]

        self.logger.info(f"Input shape: {frame.shape}")

        frame = frame.with_columns(
            pl.col("component").cast(str).map_elements(self.clean_text).alias("meta_component")
        )
        
        # Apply blacklist
        blacklist_set = set()
        if supplementary:
            bl_json = json.load(open(supplementary))
            blacklist_set = set(self.proc_blist_naive(bl_json.get("custom", [])))

        frame = frame.with_columns(
            pl.col("meta_component").is_in(blacklist_set).alias("is_blacklisted")
        )

        # CComponent-level logging
        if blacklist_set:
            n_drop = frame.filter(pl.col("is_blacklisted")).height
            dropped = frame.filter(pl.col("is_blacklisted")).select("component").unique().to_series().to_list()
            # Report blacklisted component-level hits
            self.logger.info(f"[REPORT] Dropped {n_drop} components from a blacklist with ({len(blacklist_set)} items).")
            for comp in dropped:
                self.logger.info(f"[Lookup] DROPPED: {comp}")
        else:
            self.logger.info("No blacklist applied.")

        # Identify variant groups containing at least one blacklisted component
        invalid_keys = (
            frame.filter(pl.col("is_blacklisted"))
                .select(group_keys_loc)
                .unique()
        )

        # Drop entire variant groups that have any blacklisted components
        invalid = frame.join(invalid_keys, on=group_keys_loc, how="inner")
        valid = frame.join(invalid_keys, on=group_keys_loc, how="anti")

        # Reporting
        self.logger.info(f"Output shape: {valid.shape}")
        self.reporter.report(
            invalid,
            "supplementary_dropped",
            pattern="Filtered item from a blacklist (variant contained blacklisted component).",
            field="{c}",
            status="N"
        )

        n_dropped = invalid_keys.height
        self.logger.info(f"[REPORT] Removed variants as supplementary - {n_dropped}")

        # Cleanified
        return valid.drop("is_blacklisted", "meta_component")

    def clean_by_blacklist_regimen(self, frame: pl.DataFrame, supplementary: str = None, group_keys: list[str] = None) -> pl.DataFrame:
        """
        Removes blacklisted components. If any are found in the 'regimen' field (title-lowercased),
        drops the entire group (defined by group_keys) if any component matches a blacklist regex.
        Regex is checked using Polars str.search() operations.
        Returns cleaned frame.
        """
        self.logger.info(f"Input shape: {frame.shape}")

        # Normalize 'regimen' field: lowercase → titlecase
        frame = frame.with_columns(
            pl.col("regimen")
            .cast(str)
            .str.to_lowercase()
            .alias("meta_regimen")
        )

        # Apply blacklist
        blacklist_set = set()
        if supplementary:
            bl_json = json.load(open(supplementary))
            blacklist_set = set(self.proc_blist_naive(bl_json.get("custom", [])))
            blacklist_set = set([x.lower() for x in list(blacklist_set)])

        if not blacklist_set:
            self.logger.info("No blacklist patterns provided — skipping filtering.")
            return frame.drop("meta_regimen")

        # Build Polars-native OR-ed regex condition
        condition = pl.lit(False)
        for pat in blacklist_set:
            word_boundary_pat = f"\\b{pat}\\b"
            condition = condition | pl.col("meta_regimen").str.contains(word_boundary_pat, literal=False)

        # Add blacklist flag
        frame = frame.with_columns(
            condition.alias("is_blacklisted")
        )

        # Drop groups if any item in them is blacklisted
        if group_keys:
            invalid_keys = (
                frame.filter(pl.col("is_blacklisted"))
                    .select(group_keys)
                    .unique()
            )
            valid = frame.join(invalid_keys, on=group_keys, how="anti")
            invalid = frame.join(invalid_keys, on=group_keys, how="inner")
        else:
            invalid = frame.filter(pl.col("is_blacklisted"))
            valid = frame.filter(~pl.col("is_blacklisted"))
            invalid_keys = invalid.select(group_keys or []).unique() if group_keys else invalid

        # Reporting
        self.logger.info(f"Output shape: {valid.shape}")
        self.reporter.report(
            invalid,
            "supplementary_dropped_regimen_title",
            pattern="Regimen name contains blacklisted component (regex matched).",
            field="{r}",
            status="N"
        )

        self.logger.info(f"[REPORT] Removed blacklisted groups: {invalid_keys.height}")

        # Cleanified
        return valid.drop("is_blacklisted", "meta_regimen")


    def clean_by_role(self, frame: pl.DataFrame, group_keys: list[str], field: str = "component_role", ) -> pl.DataFrame:
        dropped_groups = []
        dropped_info = []

        for value, name in [
            ("secondary systemic", "component_role_secondary_systemic"),
            ("locoregional", "component_role_locoregional")
        ]:
            # Find all variants containing this role
            subset = frame.filter(pl.col(field) == value)

            # Only keys (for grouping/removal)
            subset_keys = subset.select(group_keys).unique()
            dropped_groups.append(subset_keys)

            # Reporting for this specific role
            self.reporter.report(subset, name, pattern=f"Filtered component by role - {value}.", field="{co}, {cr}", status="N")
            # Collect for logging
            dropped_info.append((value, subset.height))

        # Merge all dropped groups
        all_dropped = pl.concat(dropped_groups) if dropped_groups else pl.DataFrame(schema={k: pl.Utf8 for k in group_keys})
        all_dropped = all_dropped.unique()

        # Calculate summary stats
        total_variants = frame.select(group_keys).unique().height
        dropped_variants = all_dropped.height
        ratio = round((dropped_variants / total_variants) * 100, 2) if total_variants else 0.0

        # Log per-role and total
        for value, count in dropped_info:
            self.logger.info(f"[REPORT] Variants with '{value}' role: {count}")
        self.logger.info(
            f"[REPORT] Total dropped variants: {dropped_variants}/{total_variants} ({ratio}%) "
            f"due to disallowed component roles."
        )

        # Exclude those variant groups
        filtered = frame.join(all_dropped, on=group_keys, how="anti")
        return filtered
