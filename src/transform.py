from tqdm import tqdm
from tools.SRE import SREModule  
from tools.seq_collapse import collapse_naive, filter_et
from tools.regimen_formatter import build_final_regimens, analyze_shortstring_regimen_mapping
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
        handler = SREModule(path_file, log_dir=logs_dir)
        handler.process()
        return handler.frame


class FrameSanitizer:
    def make_short_strings(self, df):
        # Apply filter_et to regString, then collapse_naive to get shortString
        df['shortString'] = df['regString'].apply(lambda x: collapse_naive(filter_et(x)))
        return df

    def rename_columns(self, df):
        # columns name sync - format output
        return df.rename(columns={
            "regimen": "regName", 
            "regimen_cui":"regCode",
            "condition_cui" : "conditionCode"
        })

    def select_columns(self, df):
        df["metaCondition"] = "all" 
        cols = [ 
            # "regCodeExt",
            "metaCondition",
            "condition",
            "conditionCode",
            # "context",
            # "contextCode",
            "regName",
            "variant",
            "regCode",
            "component",
            # "day",
            # "cycleTaken",
            "cycleLength",
            # "noCycles",
            # "branchInfo",
            # "Radio.Therapy.",
            # "continuous",
            # "noCycles_Original",
            "regString",
            "shortString"
            ]  
        return df[cols]

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

    def run(
        self,
        sigs_path="results/s_frame.parquet",
        output_path="results/regimens_nsclc.tsv",
        logs_dir="logs"
    ):
        print("\n --- Running Transformation Process... --- \n")

        os.makedirs(logs_dir, exist_ok=True)
        workdir = os.path.dirname(output_path)
        self.logger.set_logs_output(logs_dir)

        # ---- SKIP IF OUTPUT EXISTS AND NON-EMPTY ----
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            self.logger.logger.info(
                f"[SKIP] Output already exists and is non-empty: {output_path}"
            )
            print("--- Transform skipped (output already exists) ---")
            return 1

        # ---- build base frame ----
        frame = self.processor.run(sigs_path, logs_dir)
        frame = self.sanitizer.validate_fields(frame)
        frame = self.sanitizer.make_short_strings(frame)
        frame = self.sanitizer.rename_columns(frame)
        frame = self.sanitizer.select_columns(frame)

        # ---- logging / diagnostics ----
        self.logger.log_reports_and_sumstats(frame)
        self.logger.short_string_stats(frame)

        # ---- RAW OUTPUT (unchanged, for audit/debug) ----
        raw_path = f"{workdir}/regimens_raw.tsv"
        frame.to_csv(raw_path, sep="\t", index=False)
        self.logger.logger.info(f"[RAW OUTPUT] Saved to {raw_path} ({frame.shape[0]} rows)")

        # ---- SMART REGIMENS (authoritative) ----
        analyze_shortstring_regimen_mapping(frame, logs_dir)

        final_regimens = build_final_regimens(frame, logs_dir)
        final_regimens.to_csv(output_path, sep="\t", index=False)

        self.logger.logger.info(
            f"[REGIMENS OUTPUT] Smart format saved to {output_path} "
            f"({final_regimens.shape[0]} rows)"
        )

        print("--- Transform Process Completed Successfully! ---")


def interface():
    transform = Transform()
    transform.run(logs_dir="logs")   