import polars as pl
from tqdm import tqdm
import logging
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Set, Dict, List, Tuple, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from sre_tools import (
    get_last_cycle,
    convert_to_days,
    get_idays,
    build_component_vector,
    collapse_event_matrix_wrapper as create_reg_string,
)

GROUP_KEYS_LOG = ["condition", "regimen", "variant"]
GROUP_KEYS = ["condition_cui", "regimen_cui", "variant_key"]


@dataclass(frozen=True)
class SREState:
    drug: str
    timing_sequence: int
    all_days: str
    idays: List[int]
    cycle_length_lb: float
    cycle_length_ub: float
    cycle_length_unit: str

    @property
    def cycle_lengths(self) -> Set[float]:
        return {float(self.cycle_length_lb), float(self.cycle_length_ub)}


class SREModule:
    def __init__(self, frame_path: str, log_dir: str):
        self.frame = pl.read_parquet(frame_path)
        self.logger = self._setup_logging(log_dir)

        print("[INFO] Loaded schema:", self.frame.schema)
        self.logger.info(f"Loaded schema:\n{self.frame.schema}")

    def _setup_logging(self, log_dir: str) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Consolidated log file: SRE.log contains all DEBUG and INFO messages
        log_handler = logging.FileHandler(f"{log_dir}/SRE.log", mode="w")
        log_handler.setLevel(logging.DEBUG)
        log_handler.setFormatter(formatter)

        logger.addHandler(log_handler)

        return logger

    def _log_reg_counts(self, group: pl.DataFrame, reg_strings: List[str]) -> None:
        if len(reg_strings) > 1:
            group_id = group.select(GROUP_KEYS_LOG).to_numpy()[0]
            self.logger.debug(f"N_STRINGS={len(reg_strings)} @ {group_id}")
            self.logger.debug(reg_strings)

    def _infer_block_tseq(self, block: pl.DataFrame) -> int:
        """
        Infer authoritative block length (last_day) for a timing_sequence block.
        
        This represents the last day in the timeline when all drugs in this timing_sequence
        block are stacked into a single matrix. It must be large enough to accommodate:
        1. All actual drug administration days (allDays parsed to idays)
        2. All cycle length metadata (lb/ub converted to days)
        
        Example:
        - timing_seq="1,2,3" with DrugA allDays=[1,2,3] and DrugB allDays=[10,11]
        - DrugA needs last_day >= 3, DrugB needs last_day >= 11
        - Block's last_day = max(3, 11) = 11
        - Both vectors padded to length 11 for matrix stacking
        
        tseq (total sequence length) = max(
            max(allDays across all drugs),
            max(cycle_length_lb, cycle_length_ub) converted to days across all drugs
        )
        """
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

    def _build_vector_from_idays(
        self,
        idays: List[int],
        tseq: int,
        group_id=None,
        drug=None,
        timing_sequence=None,
    ) -> np.ndarray:
        """
        Build a binary vector of length tseq using absolute idays positions.
        """
        vec = np.zeros(tseq, dtype=int)
        
        if not idays:
            self.logger.error(f"[SRE ERR] Empty idays for drug={drug} timing_sequence={timing_sequence}")
            # raise ValueError(f"Empty idays for drug={drug} timing_sequence={timing_sequence}")
            return vec 
            
        for d in idays:
            if d < 1 or d > tseq:
                raise ValueError(
                    f"[SRE] day {d} out of range tseq={tseq} "
                    f"drug={drug} timing_sequence={timing_sequence} group={group_id}"
                )
            vec[d - 1] = 1

        return vec

    def _process_group(self, group: pl.DataFrame) -> pl.DataFrame:
        """
        Process a group (regimen_cui, variant_cui, condition_cui) by building component vectors
        for each timing_sequence block, generating regimen strings, and mapping cycle lengths.
        
        Key insight: tseq (last_day) is the information needed for stacking drug vectors into
        a single matrix. When drugs in the same timing_sequence have different cycle contexts
        (different allDays or cycle_length ranges), their vectors must all be padded to the
        same length (the block's tseq) to align in the matrix.
        
        Workflow:
        1. For each timing_sequence: Compute tseq = last_day needed to fit ALL drugs in that block
        2. Build vectors of length tseq for all drugs, padding with zeros where needed
        3. Stack vectors into component_vectors Dict[drug] = List[(timing_seq, vec)]
        4. Pass to create_reg_string() which handles:
           - normalize_multicycle_spans: Groups by vector length (handles multi-timing-seq case)
           - validate_and_split_variants: Returns one dict per unique vector length
           - collapse_event_matrix: Generates one regimen string per dict
        5. Map cycle length: Each regimen string gets the tseq that produced its vectors
        6. Repeat group and attach regString and cycleLength columns
        """
        group_id = group.select(GROUP_KEYS_LOG).to_dicts()

        component_vectors: Dict[str, List[Tuple[str, np.ndarray]]] = {}

        # Process by timing_sequence: each block has independent drugs that must align
        for timing_seq, block in group.group_by("timing_sequence", maintain_order=True):

            # timing_seq from group_by is a tuple; extract the string value
            timing_seq_str = str(timing_seq[0]) if isinstance(timing_seq, (tuple, list)) else str(timing_seq)

            # tseq = last_day in timeline for this timing_sequence block
            # Example: timing_seq="1,2,3" (3 cycles) with drugs having allDays up to day 10
            #          → tseq = 10 (all vectors padded to length 10)
            #          timing_seq="4,5" (2 cycles) with drugs having allDays up to day 12
            #          → tseq = 12 (all vectors padded to length 12)
            tseq = self._infer_block_tseq(block)

            # Process each row in the timing_sequence block
            for row in block.iter_rows(named=True):
                drug = (
                    str(row["component"])
                    .strip()
                    .replace(" ", "")
                    .lower()
                    .capitalize()
                )

                idays = get_idays(row["allDays"])

                # Extract cycle_length bounds and create variant set
                # If lb ≠ ub, this creates two separate variants for the same drug
                # (both will be padded to the block's tseq)
                try:
                    cycle_length_lb = float(row["cycle_length_lb"])
                    cycle_length_ub = float(row["cycle_length_ub"])
                    cycle_lengths = sorted({cycle_length_lb, cycle_length_ub})
                except (ValueError, TypeError):
                    cycle_lengths = [1.0]

                # For each cycle_length variant, build a component vector
                # All vectors for this timing_sequence use the block's tseq (last_day)
                # Example with mismatch: DrugA allDays=[1,2,3] vs DrugB allDays=[10,11]
                #   - Block tseq = 11
                #   - DrugA vector: [1,1,1,0,0,0,0,0,0,0,0]
                #   - DrugB vector: [0,0,0,0,0,0,0,0,0,1,1]
                for cycle_len in cycle_lengths:
                    vec = self._build_vector_from_idays(
                        idays=idays,
                        tseq=tseq,
                        group_id=group_id,
                        drug=drug,
                        timing_sequence=timing_seq_str,
                    )

                    # Store vector with timing_seq key for tracking which block it came from
                    component_vectors.setdefault(drug, []).append(
                        (timing_seq_str, vec)
                    )

        # Generate regimen strings from all component vectors
        # normalize_multicycle_spans groups vectors by length and creates @cycleLen{L} keys
        # for vectors of different lengths (from different timing_sequences or cycle variants)
        # validate_and_split_variants returns one dict per unique vector length
        # collapse_event_matrix converts each dict into one regimen string
        reg_strings = create_reg_string(component_vectors, self.logger)
        self._log_reg_counts(group, reg_strings)

        # Repeat the group N times (N = number of regimen strings generated)
        group_repeated = pl.concat([group] * len(reg_strings), how="vertical")

        # Create regString column: repeat each regimen string by group height
        reg_string_col = (
            pl.Series("regString", reg_strings)
            .repeat_by(group.height)
            .explode()
        )

        # Determine cycle length for each regimen string
        # When group spans multiple timing_sequences with different tseqs:
        # - normalize_multicycle_spans creates separate @cycleLen{L} keys
        # - Each key generates one regimen string
        # - That regimen string's cycle length = the tseq of its vector length
        # For now: use max(tseqs) as safe upper bound
        # TODO: Track which tseq produced which regimen string for precise mapping
        all_tseqs = [self._infer_block_tseq(block) for _, block in group.group_by("timing_sequence", maintain_order=True)]
        cycle_length_value = max(all_tseqs) if all_tseqs else 1

        # Create cycleLength column: repeat the cycle length for each regimen string
        cycle_length_col = (
            pl.Series("cycleLength", [cycle_length_value] * len(reg_strings))
            .repeat_by(group.height)
            .explode()
        )

        return group_repeated.with_columns(
            [reg_string_col, cycle_length_col]
        )

    def process(self):
        print(f"SRE - Frame size: {self.frame.shape}")

        assert all(c in self.frame.columns for c in GROUP_KEYS)

        results = []
        progress = tqdm(
            total=self.frame.select(GROUP_KEYS).unique().height,
            desc="Processing groups",
            dynamic_ncols=True,
        )

        for group_key, group_df in self.frame.group_by(GROUP_KEYS, maintain_order=True):
            if group_key == (None, None, None):
                continue

            # Log group key for tracking which group is being processed
            self.logger.info(f"Processing group: regimen_cui={group_key[1]}, variant_key={group_key[2]}, condition_cui={group_key[0]}")
            
            processed = self._process_group(group_df)
            results.append(processed)
            progress.update(1)

        progress.close()

        if results:
            self.frame = (
                pl.concat(results)
                .filter(pl.col("regString").is_not_null()) # what does this serve? if there are regStrings nulls I want to know about
                .to_pandas()
            )
            print(f"SRE - Processed frame size: {self.frame.shape}")
