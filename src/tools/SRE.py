import polars as pl
from tqdm import tqdm
import time
from pathlib import Path 
import sys 

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from sre_tools import (
    get_last_cycle,
    convert_to_days,
    get_idays,
    build_component_vector,
    collapse_event_matrix_wrapper as create_reg_string
)

import logging

class Handlers:
    def __init__(self):
        pass 
    
    @staticmethod
    def handle_timing_sequence(group: pl.DataFrame) -> pl.DataFrame:
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
    
    @staticmethod
    def patch_indeterminate_cycles(group: pl.DataFrame) -> pl.DataFrame:
        """
        Patch '(+c)' with '1' in cycle_length_lb or cycle_length_ub
        only when cycle_length_unit == 'indeterminate'.
        """

        log_chunk = ""
       
        patch_mask = (
            (pl.col("cycle_length_unit") == "indeterminate") &
            (pl.col("cycle_length_lb") == "(+c)") |
            (pl.col("cycle_length_ub") == "(+c)")
        )

        matching_rows = group.filter(patch_mask)

        if matching_rows.height > 0:
            log_chunk+=f"Applying patch to {matching_rows.height} rows with '(+c)' under cycle_length lb or ub"

        return group.with_columns([
            pl.when(patch_mask).then(pl.lit("1")).otherwise(pl.col("cycle_length_lb")).alias("cycle_length_lb"),
            pl.when(patch_mask).then(pl.lit("1")).otherwise(pl.col("cycle_length_ub")).alias("cycle_length_ub"),
        ]), log_chunk

class RegStringHandler:
    def __init__(self, frame_path: str, log_dir: str):
        self.frame = pl.read_parquet(frame_path)
        self.logger = self._setup_logging(log_dir)
        print("[INFO] Loaded schema:", self.frame.schema)
        self.logger.info(f"Loaded schema:\n {self.frame.schema}")


    def _setup_logging(self, log_dir): # TODO: cleanify
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        info_handler = logging.FileHandler(f"{log_dir}/SRE.process.log", mode='w')
        info_handler.setLevel(logging.INFO)
        info_handler.addFilter(lambda record: record.levelno == logging.INFO)
        info_handler.setFormatter(formatter)

        # All else (debug, warning, error, critical) → output.log
        output_handler = logging.FileHandler(f"{log_dir}/SRE.output.log", mode='w')
        output_handler.setLevel(logging.DEBUG)
        output_handler.addFilter(lambda record: record.levelno != logging.INFO)
        output_handler.setFormatter(formatter)

        logger.addHandler(info_handler)
        logger.addHandler(output_handler)

        return logger


    def _infer_block_tseq(self, block: pl.DataFrame) -> int:
        max_day_from_allDays = 0
        max_day_from_meta = 0
        for row in block.iter_rows(named=True):
            idays = get_idays(row["allDays"])
            if idays:
                max_day_from_allDays = max(max_day_from_allDays, max(idays))
        try:
            lb = float(row["cycle_length_lb"])
            ub = float(row["cycle_length_ub"])
            unit = row["cycle_length_unit"]
            meta_days = max(
                convert_to_days(lb, unit, idays),
                convert_to_days(ub, unit, idays),
            )
            max_day_from_meta = max(max_day_from_meta, meta_days)

        except Exception:
            self.logger.warning(
            f"Failed to convert cycle lengths to days for row: \n{row}"
            )

        tseq = max(max_day_from_allDays, max_day_from_meta)

        if tseq <= 0:
            raise ValueError("Cannot infer tseq for timing_sequence block")

        return int(tseq)
    
    def _process_group(self, group: pl.DataFrame ) -> pl.DataFrame:
        """
        
        Input: regimen_cui .. variant_cui .. condition_cui group

        # Boiled-down hierarchy
                   regimen -> variant > <portion?> -> components/cui -> step_number
                   step is repeated once or more -> 
                   sig anatomy (cycle_length_lb, cycle_length_ub, cycle_length_unit, timing_sequence) ->
                   allDays [-> freq/cui # dose-stuff

        """ 

        group = Handlers.handle_timing_sequence(group)
        group, log_chunk = Handlers.patch_indeterminate_cycles(group)

        if log_chunk != "":
            self.logger.error(f"[PATCHED] Detected unhandled case - {log_chunk}")

        try:
            total_vector_len = get_last_cycle(group.select("timing_sequence").unique().to_series().to_list())
        except:
            print(group.select("timing_sequence").unique().to_series().to_list())
            raise ValueError("Processing total vector length is non-standard.")

        component_groups = group.group_by("component")
        counter_mix = 0
        for g_drug, df in component_groups:
            if df.height > 1:
                counter_mix+=1
                self.logger.debug(f"Component '{g_drug}' has multiple entries ({df.height} rows).")
        if counter_mix == 0:
            self.logger.debug(f"No duplicate components per group. Safe for processing w/o components groups")

        component_vectors = {}
        component_error = False  # Track failure
        days_error = False  # Track failure
        for row in group.iter_rows(named=True):
            
            drug = str(row['component']).strip().replace(" ", "").lower().capitalize()  
            timing_sequence = row['timing_sequence']
            allDays = row['allDays']
            cycle_length_lb = row['cycle_length_lb']
            cycle_length_ub = row['cycle_length_ub']
            cycle_length_unit = row['cycle_length_unit']
            
            idays = get_idays(allDays)
            
            
            cycle_lengths = set(map(float, [cycle_length_lb, cycle_length_ub]))
            
            # Logs
            self.logger.info(f"-----Component: {drug}------")
            self.logger.info(f"cycle size: {cycle_length_lb} {cycle_length_ub} {cycle_length_unit}")
            self.logger.info(f"days within a cycle (parsed): {idays}")
            self.logger.info(f"this component is given in: {timing_sequence} of {total_vector_len}-cycle long regimen.")
                
            # main SRE block
            for length in cycle_lengths:
                try:
                    length_in_days = convert_to_days(
                        length, 
                        cycle_length_unit, 
                        allDays # handling indeterminate cases
                        )
                except Exception as e:
                    group_id = group.select(['condition', 'regimen', 'variant']).to_dicts()
                    self.logger.error(f"[SKIPPED days] Length:{length} @ CycLenUnit:{cycle_length_unit} @ allDays={allDays} : [ERR] {e}")
                    days_error = True
                    break

                try:
                    component_vector = build_component_vector(idays, length_in_days)
                except Exception as e:
                    group_id = group.select(['condition', 'regimen', 'variant']).to_dicts()
                    self.logger.error(f"[SKIPPED COMPONENT] Cycle: {length} @ Unit: {cycle_length_unit} @ AllDays {allDays} @ i-AllDays {idays} @ Length in Days - {length_in_days}: [ERR] {e}")
                    component_error = True
                    break
                
                if component_error or days_error:
                    break  # break out of rows
                component_vectors.setdefault(drug, []).append((timing_sequence, component_vector))

        try:
            group_reg_string = create_reg_string(component_vectors)
        except Exception as e:
            group_id = group.select(['condition', 'regimen', 'variant']).to_dicts()
            self.logger.error(f"[SKIPPED GROUP] Failed to create reg string {group_id} - [ERR] {e}")
            null_col = pl.Series("regString", [None] * group.height)
            null_cycle = pl.Series("cycleLength", [None] * group.height)
            return group.with_columns([null_col, null_cycle])

        n_strings = len(group_reg_string)

        if n_strings == 0:
            null_col = pl.Series("regString", [None] * group.height)
            null_cycle = pl.Series("cycleLength", [None] * group.height)
            return group.with_columns([null_col, null_cycle])
        
        if n_strings > 1:
            group_id = group.select(['condition', 'regimen', 'variant']).to_numpy()[0, :]
            self.logger.debug(f"N_STRINGS={n_strings} @ {group_id}")
            self.logger.debug(group_reg_string)

        group_repeated = pl.concat([group] * n_strings, how="vertical")
        reg_string_col = pl.Series("regString", group_reg_string).repeat_by(group.height).explode()
        all_tseqs = [self._infer_block_tseq(block) for _, block in group.group_by("timing_sequence", maintain_order=True)]
        cycle_length_value = max(all_tseqs) if all_tseqs else 1
        
        cycle_length_col = (
            pl.Series("cycleLength", [cycle_length_value] * len(group_reg_string))
            .repeat_by(group.height)
            .explode()
            )
        group_with_regstrings = group_repeated.with_columns([
            reg_string_col,
            cycle_length_col
            ])

        return group_with_regstrings
      

    def get_cycle_length_mismatch_regimens(df: pl.DataFrame) -> int:
        numeric_pattern = r"^\d+(\.\d+)?$"

        return pl.concat([
            df.filter(
                df["cycle_length_lb"].str.contains(numeric_pattern, literal=False) &
                df["cycle_length_ub"].str.contains(numeric_pattern, literal=False)
            )
            .with_columns([
                pl.col("cycle_length_lb").cast(pl.Float64).alias("lb"),
                pl.col("cycle_length_ub").cast(pl.Float64).alias("ub")
            ])
            .filter(pl.col("lb") != pl.col("ub")),

            df.filter(
                ~df["cycle_length_lb"].str.contains(numeric_pattern, literal=False) &
                ~df["cycle_length_ub"].str.contains(numeric_pattern, literal=False) &
                df["cycle_length_lb"].is_not_null() &
                df["cycle_length_ub"].is_not_null() &
                (pl.col("cycle_length_lb") != pl.col("cycle_length_ub"))
            )
        ], how="vertical_relaxed")["regimen"].unique().height
    
    def log_timing_sequence_regimen_categories(self, df: pl.DataFrame):
        condition_only_brackets = df["timing_sequence"].str.contains(r"^(\(.*?\),?)+$", literal=False)
        condition_mixed_brackets = df["timing_sequence"].str.contains(r"\(.*?\)", literal=False)

        mask_only = condition_only_brackets
        mask_mixed = ~condition_only_brackets & condition_mixed_brackets
        mask_other = ~condition_only_brackets & ~condition_mixed_brackets

        patch_mask = (
            (df["cycle_length_unit"] == "indeterminate") &
            ((df["cycle_length_lb"] == "(+c)") | (df["cycle_length_ub"] == "(+c)"))
        )

        lb_not_maching_ub = self.get_cycle_length_mismatch_regimens(df)
        self.logger.info(f"[REPORT] Unique regimens (only brackets): {df.filter(mask_only)['regimen'].unique().height}")
        self.logger.info(f"[REPORT] Unique regimens (mixed brackets): {df.filter(mask_mixed)['regimen'].unique().height}")
        self.logger.info(f"[REPORT] Unique regimens (other): {df.filter(mask_other)['regimen'].unique().height}")
        self.logger.info(f"[REPORT] Unique regimens (patch_mask): {df.filter(patch_mask)['regimen'].unique().height}")
        self.logger.info(f"[REPORT] Unique regimens (ub_not_lb): {lb_not_maching_ub}")

    def process(self):
        print(f"SRE - Frame size: {self.frame.shape}")
        group_cols = ["regimen_cui", "variant_cui", "condition_cui"]
        group_names = ["regimen", "variant", "condition"]
        assert all(col in self.frame.columns for col in group_cols), "⚠️ Missing group column(s)"

        n_groups = self.frame.select(group_cols).unique().height

        tracker = {"Total": n_groups, "Skipped_groups": 0}

        progress = tqdm(total=n_groups, desc="Processing groups", dynamic_ncols=True)
        results = []

        for group_key, group_df in self.frame.group_by(group_cols, maintain_order=True):
            if group_key == (None, None, None):
                print("⚠️ Skipping group with key (None, None, None)")
                tracker['Skipped_groups'] += 1
                continue
            
            start_time = time.time()
            processed = self._process_group(group_df)
            results.append(processed)
            progress.update(1)
            duration = time.time() - start_time
            if duration > 5:
                print(f"[WARN] Slow group {group_key} took {duration:.2f}s — breaking for debug.")
                break

        progress.close()

        tracker_summary = "\n".join([f"{k}: {v}" for k, v in tracker.items()]) # not needed probably
        self.logger.info("--- Tracker Summary: ---\n" + tracker_summary)

        if results:
            self.frame = pl.concat(results)
            self.frame = self.frame.filter(pl.col("regString").is_not_null())
            self.frame = self.frame.to_pandas()


