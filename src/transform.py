from tqdm import tqdm
from tools.SRE import RegStringHandler  
from tools.seq_collapse import collapse_naive, filter_et
from tools.regimen_formatter import build_final_regimens, analyze_shortstring_regimen_mapping
import re
import os
import logging

tqdm.pandas()

class Logger():
    def set_logs_output(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        full_handler = logging.FileHandler(f"{log_dir}/TRANSFORM.processing.log", mode='w')
        full_handler.setFormatter(formatter)
        logger.addHandler(full_handler)
        self.logger = logger

    def log_reports_and_sumstats(self, frame):
        self.logger.info(f"Final frame shape: {frame.shape}")
        self.logger.info(f"Total number of conditions: {frame.condition.nunique()}")

        reg_num =frame.regCode.nunique()
        self.logger.info(f"Total number of regimens: {reg_num}")

        reg_num_cond =frame.groupby('condition').regCode.nunique().sum()
        self.logger.info(f"Total number of regimens per condition: {reg_num_cond}")
        self.logger.info(f"#NOTE# {'Regimens repeat across conditions' if reg_num_cond > reg_num else 'regimens does not repeat across conditions'}")

        unq_var = frame.groupby(['condition','regCode']).variant.nunique().sum()
        self.logger.info(f"Total unique variants by condition and regimen (double-counted across-groups): {unq_var}")
        unq_vars_per_reg = frame[["regCode", "variant"]].drop_duplicates().shape[0]
        self.logger.info(f"Total unique variants: {unq_vars_per_reg}")

        unq_short = frame.groupby(['condition','regCode']).shortString.nunique().sum()
        self.logger.info(f"Total unique short strings by condition and regimen (double-counted across-groups): {unq_short}")
        self.logger.info(
            "#NOTE# " + (
                "Multiple variants produce the same short string" if unq_var > unq_short else
                "Each variant produces a unique short string" if unq_var == unq_short else
                "Multiple short strings produced from a single variant – unlikely! Check for errors"
            )
        )
        if unq_var > unq_short:
            dup_counts = frame.shortString.value_counts()
            self.logger.info(f"Number of short strings with multiple mappings: {len(dup_counts[dup_counts > 1])}")

        self.logger.info(f"Total number of distinct short strings: {frame.shortString.nunique()}")
        
        dup_stats = (
            frame.groupby("shortString")["condition"]
            .nunique()  # unique condition count per shortString
            .describe() # summary statistics
        )
        self.logger.warning(f"\n\n[SUMSTATS - 1] Short Strings shared across Conditions: \n{dup_stats}")

    def short_string_stats(self, frame):
        cleaned = (
            frame
            .sort_values(by=["shortString", "condition"], ascending=[True, False])
            .groupby(["condition", "regCode", "variant"], group_keys=False)
            .apply(lambda g: g.drop_duplicates(subset="shortString", keep="first"))
        )

        counts = cleaned['shortString'].value_counts()
        max_count = counts.max()
        most_common = counts[counts == max_count]

        self.logger.warning(
            f"\n\n[SUMSTATS - 2] Short Strings Total:\n{counts.describe()}\nMost Common: {most_common}\n"
        )


class FrameProcessor:
    def run(self, path_file, logs_dir):
        """SRE endpoint"""
        handler = RegStringHandler(path_file, log_dir=logs_dir)
        handler.process()
        return handler.frame


class FrameSanitizer:
    def make_short_strings(self, df):
        # Remove internal @cycleLen markers before collapsing to shortString.
        # Markers are internal bookkeeping used by SRE; canonical shortString
        # format must be `<day>.<drug>;`.
        def strip_cycle_len(s: str) -> str:
            if not isinstance(s, str):
                return s
            return re.sub(r"@cyclelen\d+", "", s, flags=re.IGNORECASE)
        df['shortString'] = df['regString'].apply(lambda x: collapse_naive(strip_cycle_len(x)))
        return df

    # columns name sync - format output
    regimens2Hemonc = {
        "conditionCode": "condition_cui",
        "regName": "regimen",
        "regCode": "regimen_cui",
        "componentCode": "component_cui",
        "regCodeExt": None,
        "context": None,
        "contextCode": None,
        "day": None,
        "cycleTaken": None,
        "noCycles": None,
        "branchInfo": None,
        "Radio.Therapy.": None,
        "continuous": None,
        "noCycles_Original": None,
    }

    def translate(self, df):
        cols_to_translate = {v: k for k, v in self.regimens2Hemonc.items() if v is not None}
        return df.rename(columns=cols_to_translate)

    def add_metacondition(self, df):
        df["metaCondition"] = "all"
        return df

    def validate_fields(self, df):
        if df.empty:
            raise ValueError("::ERR::`final table` is empty! Something went wrong!")
        if "regString" not in df.columns:
            raise ValueError("No regString created.")
        return df


class Transform:
    def __init__(self, ):
        self.processor = FrameProcessor()
        self.sanitizer = FrameSanitizer()
        self.logger = Logger()
        self.selected_columns = [
            "metaCondition",
            "condition",
            "conditionCode",
            "regName",
            "variant",
            "regCode",
            "component",
            "cycleLength",
            "regString",
            "shortString",
        ]
        self.selected_columns_raw = [
            "metaCondition",
            "condition",
            "conditionCode",
            "regName",
            "variant",
            "regCode",
            "component",
            "componentCode",
            "cycleLength",
            "route",
            "regString",
            "shortString",
        ]

    def run(self, sigs_path="results/s_frame.parquet", output_path="results/regimens_nsclc.tsv", logs_dir="logs", debug=False):
        print("--- Running Transformation Process. Use debug=True to speedup. ---")

        os.makedirs(logs_dir, exist_ok=True)
        workdir = os.path.dirname(output_path)
        self.logger.set_logs_output(logs_dir)

        # ---- SKIP IF OUTPUT EXISTS AND NON-EMPTY ----
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0 and debug:
            self.logger.logger.info(f"[SKIP] Output already exists and is non-empty: {output_path}")
            print("--- Transform skipped (output already exists) ---")
            return 1

        # ---- build base frame ----
        frame = self.processor.run(sigs_path, logs_dir)
        frame = self.sanitizer.validate_fields(frame)
        frame = self.sanitizer.make_short_strings(frame)
        frame = self.sanitizer.translate(frame)
        frame = self.sanitizer.add_metacondition(frame)

        # ---- logging / diagnostics ----
        self.logger.log_reports_and_sumstats(frame)
        self.logger.short_string_stats(frame)

        frame.to_csv(f"{workdir}/frame.checkpoint.tsv", sep='\t')

        # ---- FULL OUTPUT (pre-dedup, all rows, with componentCode + route) ----
        # Written first — regimens_full.tsv is the unfiltered source of truth.
        # No deduplication applied. One row per (condition, regCode, variant, component).
        full_path = output_path.replace(".tsv", "_full.tsv")
        frame[self.selected_columns_raw].to_csv(full_path, sep="\t", index=False)
        self.logger.logger.info(f"[FULL OUTPUT] Saved to {full_path} ({frame.shape[0]} rows)")

        # ---- REGIMENS INDEX (shortString-deduped, authoritative schedule index) ----
        # One row per unique shortString. Full condition×regimen expansion is in
        # regimens_shortStrings.tsv (generated by data_model.generate_shortString_table).
        analyze_shortstring_regimen_mapping(frame, logs_dir)

        final_regimens = build_final_regimens(frame, logs_dir)
        final_regimens[self.selected_columns].to_csv(output_path, sep="\t", index=False)
        self.logger.logger.info(f"[REGIMENS OUTPUT] Schedule index saved to {output_path} ({final_regimens.shape[0]} rows)")

        print("--- Transform Process Completed Successfully! ---")

## Example run 
def test():
    transform = Transform()
    transform.run(logs_dir="logs")   