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
    
    # TODO: variants explosion
    # Unhandled 2,(+2) at the moment
    # Unhandled (+) migth match group timing_sequence pattern
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

        # Info-only handler → process.log
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


    def _process_group(self, group: pl.DataFrame ) -> pl.DataFrame:
        """
        Dose unique group processing
        Note: all columns must not be NaN ! WIP
        Input: regimen_cui .. variant_cui .. condition_cui

        # Boildown hierarchy
                   regimen -> variant > <portion?> -> components/cui -> step_number
                   step is repeated once or more -> 
                   sig anatomy (cycle_length_lb, cycle_length_ub, cycle_length_unit, timing_sequence) ->
                   allDays [-> freq/cui # dose-stuff

        """ 

        group = Handlers.handle_timing_sequence(group)

        # needed for matrix cration endpoint only!
        total_vector_len = get_last_cycle(group.select("timing_sequence").unique().to_series().to_list())

        # Is there any group size greater then 1?
        # NOTE: current implementation does not support duplicate components per MAIN group
        # Will be changes once variant_id allows multipart Sigs... TODO
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
            
            # extract all days 
            # TODO: these are cleaned at the moment no (n+, c+) or (optional)... 
            # Need to handle all cases
            idays = get_idays(allDays)
            
            
            # TODO: This will create 2 subvariants shortStrings at the moment !
            # Decided to keep it this way for now instead of range or single value
            # Also in matrix, we are not mixing lb and ub at the moment!... ether all lb or ub...
            # It uses set not to repeat indeterminate cases, since both are = 1
            cycle_lengths = set([cycle_length_lb, cycle_length_ub])
            
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
                    self.logger.error(f"[SKIPPED days] {length} @ {cycle_length_unit} @ {allDays} : [ERR] {e}")
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
            # Tag the group rows with null regString (preserving row count)
            null_col = pl.Series("regString", [None] * group.height)
            return group.with_columns(null_col)

        # How many regStrings are we generating?
        n_strings = len(group_reg_string)

        # Duplicate the full group N times
        group_repeated = pl.concat([group] * n_strings, how="vertical")

        # Attach the regString column — one for each duplicate block
        reg_string_col = pl.Series("regString", group_reg_string).repeat_by(group.height).explode()

        # Final merged frame
        group_with_regstrings = group_repeated.with_columns(reg_string_col)

    
        return group_with_regstrings
      

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

            self.logger.info(f"Processing group: {group_df.select(group_names).unique().to_numpy()[0]}")

            dm_list = [  2554,  29570,   7333, 116702, 121537,  59546,  31986,  30172,
        50101,  59563,   2610]
            
            dm_list = list(map(str, dm_list))
            
            filtered = group_df.filter(pl.col("regimen_cui").is_in(dm_list))
            if filtered.height > 0:
                self.logger.debug(filtered.select([
                    "component", "variant", "allDays", "timing_sequence", "cycle_length_unit"
                ]))
            processed = self._process_group(group_df)
            results.append(processed)
         
            progress.update(1)

            duration = time.time() - start_time
            if duration > 5:
                print(f"[WARN] Slow group {group_key} took {duration:.2f}s — breaking for debug.")
                break

        progress.close()

        # Log tracker summary to INFO log
        tracker_summary = "\n".join([f"{k}: {v}" for k, v in tracker.items()]) # not needed probably
        self.logger.info("--- Tracker Summary: ---\n" + tracker_summary)

        if results:
            self.frame = pl.concat(results)
            self.frame = self.frame.filter(pl.col("regString").is_not_null())
            self.frame = self.frame.to_pandas()


