import polars as pl
import re

class Resolver:
    def __init__(self, logger:object, reporter:object):
        self.logger = logger
        self.reporter = reporter
        self.f_label  = "resolved"

class ResolverParted(Resolver):
    def __init__(self, logger: object, reporter: object):
        super().__init__(logger, reporter)

    @staticmethod
    def with_variant_key(df: pl.DataFrame) -> pl.DataFrame:
        if ("variant_cui" not in df.columns) and ("variant" not in df.columns):
            raise ValueError("Need at least one of: variant_cui or variant to build variant_key")

        return df.with_columns(
            pl.when(
                pl.col("variant_cui").is_not_null()
                & (pl.col("variant_cui").cast(pl.Utf8).str.strip_chars() != "")
            )
            .then(pl.concat_str([pl.lit("cui:"), pl.col("variant_cui").cast(pl.Utf8)], separator=""))
            .otherwise(
                pl.concat_str(
                    [
                        pl.lit("txt:"),
                        pl.col("variant")
                        .cast(pl.Utf8)
                        .fill_null("")
                        .str.strip_chars()
                        .str.to_lowercase()
                        .str.replace_all(r"\s+", " "),
                    ],
                    separator="",
                )
            )
            .alias("variant_key")
        )

    def resolve_partial(self, frame: pl.DataFrame, group_keys) -> pl.DataFrame:
        frame = self.with_variant_key(frame)

        identity_cols = [
            "component_cui",
            "timing_sequence",
            "step_number",
            "allDays",
            "cycle_length_lb",
            "cycle_length_ub",
            "cycle_length_unit",
        ]

        missing = [c for c in identity_cols + group_keys if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns for resolver: {missing}")

        frame = frame.with_row_count("_rid")

        resolved_groups = []
        dropped_groups = []

        for gkey, group_df in frame.group_by(group_keys, maintain_order=True):
            before = group_df.height

            group_resolved = group_df.unique(
                subset=identity_cols,
                keep="first",
                maintain_order=True,
            )

            after = group_resolved.height
            if after != before:
                self.logger.info(f"[RESOLVE] Dropped {before - after} duplicate rows in group={gkey}")

                dropped = group_df.join(
                    group_resolved.select(["_rid"]),
                    on="_rid",
                    how="anti",
                )
                dropped_groups.append(dropped)

            resolved_groups.append(group_resolved)

        resolved = pl.concat(resolved_groups, how="vertical").drop("_rid")
        self.reporter.to_tsv(resolved, f"multi_part_sigs.{self.f_label}")

        if dropped_groups:
            dropped_all = pl.concat(dropped_groups, how="vertical").drop("_rid")
            self.reporter.to_tsv(dropped_all, f"multi_part_sigs.dropped.{self.f_label}")

        n_in = frame.select(group_keys).unique().height
        n_out = resolved.select(group_keys).unique().height
        self.logger.info(f"[RESOLVED] Multi-parted variants fixed: {n_out} / {n_in}")

        return resolved

    @staticmethod
    def combine(table_list):
        return pl.concat(table_list, how="vertical")

class ResolverIndefinite(Resolver):
    def __init__(self, logger:object, reporter:object):
        super().__init__(logger, reporter)

    def timing_sequence(
            self,
            frame: pl.DataFrame, 
            group_keys=['condition_cui', 'regimen_cui', 'variant'],
        ):
        group_keys = group_keys[:-1] + ['variant']

        resolved_groups = []
        log_chunks = []
        
        for i, group_df in frame.group_by(group_keys, maintain_order=True):
            group_p = self.indefinite_timing(group_df)
            resolved_groups.append(group_p)

        resolved = pl.concat(resolved_groups, how='vertical')

        self.reporter.to_tsv(resolved, f"cycle_length_unit_indefinite.{self.f_label}")

        n_rows = resolved.height
        self.logger.info(f"[INFO] : {n_rows} rows cleaned from optional (.*) notation.")
      
        return resolved
        
    # TODO: variants explosion
    # Unhandled 2,(+2) at the moment
    # Unhandled (+) might match group timing_sequence pattern
    @staticmethod
    def indefinite_timing(group: pl.DataFrame) -> pl.DataFrame:
        """
        timing_sequence will be changed if contains (.*) pattern, other unaffected
        - For values like '(3),(4)' alone -> remove parentheses only
        - For mixed values like '1,(+c),12' -> remove all (...) blocks
        - For values in brackets and not, remove the optional part (brackets)
        """
        group = group.with_columns([
            pl.when(
                pl.col("timing_sequence").str.contains(r"^(\(.*?\),?)+$", literal=False)  # simulates full_match
            )
            .then(
                pl.col("timing_sequence").str.replace_all(r"[()]", "", literal=False)
            )
            .when(
                pl.col("timing_sequence").str.contains(r"\(.*?\)", literal=False)
            )
            .then(
                pl.col("timing_sequence")
                .str.replace_all(r"\(.*?\)", "", literal=False)
                .str.replace_all(r",+", ",")  # clean up double commas
                .str.strip_chars(",")         # trim edge commas
            )
            .otherwise(pl.col("timing_sequence"))
            .alias("timing_sequence")
        ])
        return group
    
    def cycle_bounds(
            self,
            frame: pl.DataFrame, 
            group_keys=['condition_cui', 'regimen_cui', 'variant'],
        ):
        group_keys = group_keys[:-1] + ['variant']

        resolved_groups = []
        log_chunks = []
        
        for i, group_df in frame.group_by(group_keys, maintain_order=True):
            group_p, log_chunk = self.patch_indeterminate_cycles(group_df)
            resolved_groups.append(group_p)
            if log_chunk:
                log_chunks.append(log_chunk)

        resolved = pl.concat(resolved_groups, how='vertical')

        self.reporter.to_tsv(resolved, f"indeterminate_cycle_lengths.{self.f_label}")

        if log_chunks:
            self.logger.info(f"[RESOLVED] Indeterminate cycle patches applied:\n" + "\n".join(log_chunks))
      
        return resolved

    @staticmethod
    def patch_indeterminate_cycles(group: pl.DataFrame) -> pl.DataFrame:
        """
        Patches '(+c)' or 'NUB' in cycle_length_lb / cycle_length_ub with '1'
        if either the cycle_length_unit is 'indeterminate' OR the values are bad.
        """

        log_chunk = ""
        bad_vals = ["(+c)", "NUB"]

        # Detect bad cycle length values (independent of unit)
        bad_bounds_mask = (
            pl.col("cycle_length_lb").is_in(bad_vals)
            | pl.col("cycle_length_ub").is_in(bad_vals)
        )

        # Detect indeterminate unit
        indeterminate_mask = (pl.col("cycle_length_unit") == "indeterminate")

        # Patch if either condition is true
        patch_mask = indeterminate_mask | bad_bounds_mask

        matching_rows = group.filter(patch_mask)

        if matching_rows.height > 0:
            regimens = matching_rows.select(pl.col("regimen").unique()).to_series().to_list()
            regimen_str = ", ".join(regimens)

            variants = matching_rows.select(pl.col("variant").unique()).to_series().to_list()
            variant_str = ", ".join(variants)

            log_chunk += (
                f"regimen: {regimen_str}, variant: {variant_str} - {matching_rows.height} rows"
            )

        patched_group = group.with_columns([
            pl.when(patch_mask).then(pl.lit("1")).otherwise(pl.col("cycle_length_lb")).alias("cycle_length_lb"),
            pl.when(patch_mask).then(pl.lit("1")).otherwise(pl.col("cycle_length_ub")).alias("cycle_length_ub"),
        ])

        return patched_group, log_chunk



class ResolverAllDays(Resolver):
    """
        Resolve patterns in the `allDays` field.

        Regular patterns:     \d+
        Irregular patterns:   -\d+ | \d+\|\d+ | \d+~\d+ | \(.*?\) | \b0\b
                                NZ      SR       SR         BR      NZ
        Input:
            - frame from handler (may contain irregular / misaligned allDays values)
        Output:
            - list of patched group DataFrames, each with normalized allDays

        Notes:
            Pediatric regimens may begin at day 0.
            Transplant/biologic regimens may begin at negative days.
            We ignore transplant day and normalize everything relative to regimen start.

            Example:
                input:  [-4, -1, 0, 1, 4], [0], []
                output: [1, 4, 5, 6, 9]

            BUT:
            allDays values across variants are considered relative.
            Example:
                variant_1 = [0, 2, 4]
                variant_2 = [-3, 1, 2]
                Resulting span = -3 to 4
                    regimen timing_unit     component allDays
                0  7+3d and Glasdegib      Course    Cytarabine       1
                1  7+3d and Glasdegib      Course  Daunorubicin   1,2,3
                2  7+3d and Glasdegib      Course     Glasdegib      -3


            We will use global span (min, max).
    """
    def __init__(self, logger:object, reporter:object):
        super().__init__(logger, reporter)

    # ----------------------------
    # PARSING UTILITIES
    # ----------------------------
    @staticmethod
    def remove_bracket_content(ds: str) -> str:
        """
        Remove all parenthesized content from a raw allDays string.
        Example: '1,2,(3),(4)' → '1,2'
        """
        # Remove (anything)
        cleaned = re.sub(r"\([^)]*\)", "", ds)

        # Remove extra commas/spaces after deletion
        cleaned = re.sub(r"\s*,\s*", ",", cleaned)  # normalize comma spacing
        cleaned = re.sub(r",+", ",", cleaned)       # collapse duplicate commas
        cleaned = cleaned.strip("," )               # strip leading/trailing commas

        return cleaned

    @staticmethod
    def parse_days(ds: str) -> list[int]:
        """Parse a single allDays string into a flat list of ints."""
        # Range support: "3~5" or "3|5" means take lower bound
        if "~" in ds or "|" in ds:
            parts = re.split(r"[~|]", ds)
            return [int(parts[0])]

        # Normal comma-separated case
        return list(map(int, ds.split(",")))

    @staticmethod
    def collapse_zero(days: list[int]) -> list[int]:
        """If all zero → [1]."""
        return [1] if all(d == 0 for d in days) else days

    @staticmethod
    def scale_days(days: list[int], global_min: int) -> list[int]:
        """Shift so global_min → 1."""
        return [(d - global_min) + 1 for d in days]

    # ----------------------------
    # MAIN NORMALIZATION
    # ----------------------------
    def resolve_group(self, group_df: pl.DataFrame) -> pl.DataFrame:
        raw_strings = group_df["allDays"].to_list()

        # 1. remove parenthesized content
        cleaned_strings = [self.remove_bracket_content(s) for s in raw_strings]

        # 2. parse + zero collapse
        parsed = [self.collapse_zero(self.parse_days(s)) for s in cleaned_strings]

        # 3. flatten, compute global min, scale, etc.
        flat = [d for lst in parsed for d in lst]
        if not flat:
            return group_df

        global_min = min(flat)

        scaled = [self.scale_days(lst, global_min) for lst in parsed]
        scaled_strings = [",".join(map(str, lst)) for lst in scaled]

        return group_df.with_columns(
            pl.Series("allDays", scaled_strings)
        )

    # ----------------------------
    # ENTRY POINT
    # ----------------------------
    def resolve_ex(self, frame: pl.DataFrame, group_keys):
        patched = []

        for _, group_df in frame.group_by(group_keys, maintain_order=True):
            patched.append(self.resolve_group(group_df))

        return pl.concat(patched, how="vertical")

class ResolverKey:
    @staticmethod
    def with_variant_key(df: pl.DataFrame) -> pl.DataFrame:
        if ("variant_cui" not in df.columns) and ("variant" not in df.columns):
            raise ValueError("Need at least one of: variant_cui or variant to build variant_key")

        return df.with_columns(
            pl.when(
                pl.col("variant_cui").is_not_null()
                & (pl.col("variant_cui").cast(pl.Utf8).str.strip_chars() != "")
            )
            .then(pl.concat_str([pl.lit("cui:"), pl.col("variant_cui").cast(pl.Utf8)], separator=""))
            .otherwise(
                pl.concat_str(
                    [
                        pl.lit("txt:"),
                        pl.col("variant")
                        .cast(pl.Utf8)
                        .fill_null("")
                        .str.strip_chars()
                        .str.to_lowercase()
                        .str.replace_all(r"\s+", " "),
                    ],
                    separator="",
                )
            )
            .alias("variant_key")
        )