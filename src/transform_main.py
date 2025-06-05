import pandas as pd
from tqdm import tqdm 
from tools.adapters import apply_addapters
from tools.SRE import RegStringHandler  
from tools.collapse_seq_naive import collapse
import os
import logging
tqdm.pandas()



class Transform:
    def __init__(self):
        self.logger = None

    def _load(self, csv_path, sep='\t', encoding=None):
        return pd.read_csv(csv_path, sep=sep, encoding=encoding)
    

    def _setup_logging(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        full_handler = logging.FileHandler(f"{log_dir}/TRANSFORM.processing.log", mode='w')
        full_handler.setFormatter(formatter)
        logger.addHandler(full_handler)
        return logger

    def process(self, workdir, logs_dir, supplementary_file_path=None):
        """Process `sigs` by engineering `regString` and `shortString`."""

        self.logger = self._setup_logging(logs_dir)

        # clean
        processed_file_path = apply_addapters(
            frame_path=f"{workdir}/s_frame.parquet", 
            output_dir=workdir,
            supplementary=supplementary_file_path,
            logs_dir=logs_dir
        ) 

        print(f"{processed_file_path=}")
      
        # # SRE endpoint
        obj = RegStringHandler(processed_file_path, log_dir=logs_dir)
        obj.process()
        frame = obj.frame

        # check wheter created
        if "regString" in frame.columns:
           print("--- Regimen Strings created. ---")
        else:
            raise ValueError("No regString created.")
        
        # short string collapse endpoint
        frame['shortString'] = frame['regString'].apply(collapse)

        # columns name sync - format output
        frame.rename(columns={
            "regimen": "regName", "regimen_cui":"regCode",
            "condition_cui" : "conditionCode"
        }, inplace=True)

        final_sorted_cols = [
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
            # "cycleLength",
            # "noCycles",
            # "branchInfo",
            # "Radio.Therapy.",
            # "continuous",
            # "noCycles_Original",
            "regString",
            "shortString",
        ]

        frame = frame[final_sorted_cols]

        # finalize and validate - drop duplicated components to unique shortStrings
        if frame.empty:
            raise ValueError("::ERR::`final table` is empty! Something went wrong!")

        self.logger.info(f"Final frame shape: {frame.shape}")
        self.logger.info(f"Total number of conditions: {frame.condition.nunique()}")
        reg_num =frame.regCode.nunique()
        self.logger.info(f"Total number of regimens: {reg_num}")
        reg_num_cond =frame.groupby('condition').regCode.nunique().sum()
        self.logger.info(f"Sum of regimens per condition: {reg_num_cond}")
        self.logger.info(f"[Explain] - {'regimens repeat across conditions' if reg_num_cond > reg_num else 'regimens does not repeat across conditions'}")
        unq_var = frame.groupby(['condition','regCode']).variant.nunique().sum()
        self.logger.info(f"Total unique variants per condition and regimen: {unq_var}")
        unq_short = frame.groupby(['condition','regCode']).shortString.nunique().sum()
        self.logger.info(f"Total unique short Strings per condition and regimen: {unq_short}")
        self.logger.info(f"[Explain] - {'Variants does not produce same short strings' if unq_var < unq_short else 'Variants produce same short strings'}")
        if unq_var > unq_short:
            dup_counts = frame.shortString.value_counts()
            self.logger.info(f"Number of duplicated short strings: {len(dup_counts[dup_counts > 1])}")

        self.logger.info(f"Total unique short Strings: {frame.shortString.nunique()}")
        
        dup_stats = (
            frame.groupby("shortString")["condition"]
            .nunique()  # unique condition count per shortString
            .describe() # summary statistics
        )
        self.logger.warning(f"\n\n[SUMSTATS - 1] Short Strings shared across Conditions: \n{dup_stats}")

        cleaned_frame = (
            frame
            .sort_values(by=["shortString", "condition"], ascending=[True, False])
            .groupby(["condition", "regCode", "variant"], group_keys=False)
            .apply(lambda g: g.drop_duplicates(subset="shortString", keep="first"))
        )
        counts = cleaned_frame['shortString'].value_counts()
        max_count = counts.max()
        most_common = counts[counts == max_count]
        self.logger.warning(f"\n\n[SUMSTATS - 2] Short Strings Total:\n{counts.describe()}\nMost Common: {most_common}\n")

        frame.to_csv(f"{workdir}/frame.checkpoint.tsv", sep='\t')
        return frame

    def keep_unique_per_condition(self, frame, output_path="path/to/deduped_per_condition.sigs"):
         # keeps shortString per condition
        frame = (
            frame
            .sort_values(by=["shortString", "condition"], ascending=[True, False])
            .groupby("condition", group_keys=False)
            .apply(lambda g: g.drop_duplicates(subset="shortString", keep="first"))
        )

        frame.to_csv(output_path, sep='\t', index=False)

    def keep_unique(self, frame, output_path="path/to/deduped.sigs"):
         # keeps shortString per condition
        frame = (
            frame
            .sort_values(by=["shortString", "condition"], ascending=[True, False])
            .drop_duplicates(subset="shortString", keep="first")
        )

        frame.to_csv(output_path, sep='\t', index=False)


    def run(self, output_path="results/regimens_nsclc.tsv", supplementary_file=None, logs_dir="workdir/logs"):
        """Execute full transform pipeline with structured steps."""
       
        print("\n --- Running Transformation Process... --- \n")

        # logs_dir = f"{workdir}/logs"
        workdir = os.path.dirname(output_path)
        os.makedirs(logs_dir, exist_ok=True)
       
        final = self.process(workdir, logs_dir, supplementary_file)
        self.keep_unique_per_condition(final, output_path.replace(".tsv", "_full.tsv"))
        self.keep_unique(final, output_path)

        print("--- Transform Process Completed Successfully! ---")

## Example run 
def test():
    transform = Transform(log_dir="logs")
    transform.run()   