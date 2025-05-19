import polars as pl
from tqdm import tqdm
import time
from collections import defaultdict

from pathlib import Path 
import sys 
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from sre_tools import (
    extract_number,
    get_idays,
    build_component_vector,
    collapse_event_matrix_wrapper,
)

import logging


class RegStringHandler:
    def __init__(self, merged_df_path: str, log_dir: str):
        with open(merged_df_path, "r", encoding="utf-8") as f:
            header_line = f.readline().strip()
        column_names = header_line.split('\t')

        schema_overrides = {col: pl.Utf8 for col in column_names}

        self.frame = pl.read_csv(
            merged_df_path,
            separator='\t',
            dtypes=schema_overrides
        )
    
        print("[INFO] Loaded schema:", self.frame.schema)

        self.logger = self._setup_logging(log_dir)


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

    def _process_group(self, group: pl.DataFrame, condition: str = None) -> pl.DataFrame:
        
        component_variants = {}
        group = group.unique(subset=["component", "allDays", "cyclesigs", "variant", "regimen_cui"])

        for row in group.iter_rows(named=True):
            
            drug = str(row["component"]).strip().replace(" ", "").lower().capitalize()
            cycsigs_int, unit_or_reason = extract_number(row["cyclesigs"])

            self.logger.info(f"cycsigs: {cycsigs_int} - {unit_or_reason}")

            if row['allDays']:
                idays = get_idays(row['allDays'])
            else:
                idays = [None]
                self.logger.warning(f" ❌ Missing: {drug}, {cycsigs_int}, {idays}")

            self.logger.debug(f"✅ PassedAs: {drug}, {cycsigs_int}, {idays}")

            vector = build_component_vector(idays, drug, cycsigs_int, debug=False)
            built_component = vector.get(drug, None)
            # built_tracker = vector.get("tracker", "Correct")

            if built_component:
                component_variants.setdefault(drug, []).append(built_component)

        if not component_variants:
            self.logger.warning(f"⚠️ No usable variants in group:\n{group.select(['regimen_cui', 'variant_cui', 'component_cui', 'cyclesigs','allDays','condition'])}")
            return pl.DataFrame(schema={**group.schema, "regString": pl.Utf8})

        all_reg_strings = []

        #
        # Group components by vector length 
        #
        length_groups = defaultdict(dict)
        for drug, variants in component_variants.items():
            for variant in variants:
                if variant is not None:
                    length_groups[len(variant)][drug] = variant

        for length, group_dict in length_groups.items():
            reg_strings, _ = collapse_event_matrix_wrapper(group_dict)
            all_reg_strings.extend([s for s in reg_strings if s])

        if not all_reg_strings:
            self.logger.warning(f"⚠️ No regimen strings generated for group:\n{group}")
            schema = group.schema
            if "regString" not in schema:
                schema["regString"] = pl.Utf8
            return pl.DataFrame(schema=schema)

        if len(all_reg_strings) == 1:
            return group.with_columns([
                pl.lit(all_reg_strings[0]).alias("regString")
            ])
        else:
            return pl.concat([
                group.clone().with_columns([
                    pl.lit(rs).alias("regString")
                ])
                for rs in all_reg_strings
            ])

    def process(self):
        print(f"SRE - Frame size: {self.frame.shape}")
        group_cols = ["regimen_cui", "variant"]
        assert all(col in self.frame.columns for col in group_cols), "⚠️ Missing group column(s)"

        n_groups = self.frame.select(group_cols).unique().height

        tracker = {"Total": n_groups, "Skipped_groups": 0}

        progress = tqdm(total=n_groups, desc="Processing groups", dynamic_ncols=True)
        results = []

        for group_key, group_df in self.frame.group_by(group_cols, maintain_order=True):
            if group_key == (None, None):
                print("⚠️ Skipping group with key (None, None)")
                tracker['Skipped_groups'] += 1
                continue

            start_time = time.time()

            #### Condition impuation - dirty fix
            ### TODO: Isolate for optional use...
            condition_series = group_df.select("condition")
            condition_values = condition_series.to_series().to_list()
            named_conditions = sorted({c for c in condition_values if c is not None})

            # Case 1: All nulls
            if not named_conditions:
                tracker[(group_key, None)] = len(group_df)
                try:
                    processed = self._process_group(group_df, None)
                    results.append(processed)
                except Exception as e:
                    self.logger.warning(f"⚠️ Error processing group {group_key} | None: {e}")
                progress.update(1)
                continue

            # Case 2: Named + nulls → expand nulls into all named
            for cond in named_conditions:
                # Subset where condition == cond
                cond_mask = group_df["condition"] == cond
                cond_rows = group_df.filter(cond_mask)

                # Subset where condition is null
                null_rows = group_df.filter(group_df["condition"].is_null())
                if null_rows.height > 0:
                    null_rows = null_rows.with_columns(pl.lit(cond).alias("condition"))

                # Combine named and filled nulls
                expanded_group = pl.concat([cond_rows, null_rows])

                tracker[(group_key, cond)] = len(expanded_group)

                try:
                    processed = self._process_group(expanded_group, cond)
                    results.append(processed)
                except Exception as e:
                    self.logger.warning(f"⚠️ Error processing group {group_key} | {cond}: {e}")

                progress.update(1)

            duration = time.time() - start_time
            if duration > 5:
                print(f"[WARN] Slow group {group_key} took {duration:.2f}s — breaking for debug.")
                break


        progress.close()

        # Log tracker summary to INFO log
        tracker_summary = "\n".join([f"{k}: {v}" for k, v in tracker.items()])
        self.logger.info("--- Tracker Summary: ---\n" + tracker_summary)

        if results:
            self.frame = pl.concat(results)
            self.frame = self.frame.to_pandas()


