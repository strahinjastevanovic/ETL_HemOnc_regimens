import pandas as pd
from tqdm import tqdm 
from tools.SRE import RegStringHandler  
from tools.collapse_seq_naive import collapse
from tools.frame_ALL import frame_all
import os

tqdm.pandas()


class ETL:
    def __init__(self):
        """Initialize with a directory containing CSV files and a schema."""
        self.tables = None

    def _load(self, csv_path, sep='\t', encoding=None):
        return pd.read_csv(csv_path, sep=sep, encoding=encoding)

    def process_frame(self, workdir, logs_dir, supplementary_file=None):
        """Process `sigs.csv` by extracting `regString` and `shortString`."""
        print(workdir)

        frame_all(
            indi_path=f"{workdir}/i_frame.tsv", 
            sigs_path=f"{workdir}/s_frame.tsv", 
            output_dir=workdir,
            supplementary=supplementary_file) 


        self.tables = {
            "final" : self.SRE_endpoint(f"{workdir}/merged.tsv", logs_dir),
        }

    def SRE_endpoint(self, merged_tsv, workdir_logs):
        """SRE endpoint and does preprocessing of component to merge into SRE"""
        obj = RegStringHandler(merged_tsv, log_dir=workdir_logs)
        obj.process()
        df = obj.frame

        if "regString" in df.columns:
           print("--- Regimen Strings created. ---")
        else:
            raise ValueError("No regString")
        
        return df

    def short_string_collapse_endpoint(self):
        fin = self.tables['final'].copy()
        fin['shortString'] = fin['regString'].apply(collapse)
        self.tables['final'] = fin
        print("--- Strings Collapsed to Short---")

    def format_output(
            self,     
            ):
        """Columns name sync."""
        
        # TODO: cleanify
        final_sorted_cols = [
            # "regCodeExt",
            "metaCondition",
            "condition",
            # "conditionCode",
            "context",
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
        table_upd = self.tables['final'].rename(columns={
            "regimen": "regName", "regimen_cui":"regCode",
        })

        self.tables["final"] = table_upd[final_sorted_cols]


    def finalize_output(self):
        """Finalize and validate final table."""
        if self.tables["final"].empty:
            raise ValueError("::ERR::`final table` is empty! Something went wrong in merging!")

        self.tables['final'] = (
            self.tables['final']
            .sort_values(
                by=["shortString", "condition", "context"], 
                ascending=[True, False, False]
            )
            # .drop_duplicates(subset="shortString", keep="first") # TODO: cleanify
        )
        

    def run(self, output_path="results/regimens_nsclc.tsv", supplementary_file=None):
        """Execute full ETL pipeline with structured steps."""
       
        print("\n --- Starting ETL Process... --- \n")

        # TODO: cleanify
        workdir = os.path.dirname(output_path)
        logs_dir = f"{workdir}/logs"
        os.makedirs(logs_dir, exist_ok=True)
       
        self.process_frame(workdir, logs_dir, supplementary_file)
       
        self.short_string_collapse_endpoint() 
       
        self.format_output()
       
        self.finalize_output()

        final_out = self.tables['final'].copy()
        
        final_out.to_csv(output_path.replace(".tsv", "_full.tsv"), sep='\t', index=False)

        # TODO: temp resolves unique
        out = (
            final_out.sort_values(by='condition', na_position='last')  # Push None to end
            .drop_duplicates(subset='shortString')                     # Keep first (non-null if present)
        )
        out.to_csv(output_path, sep='\t', index=False)

        print("--- ETL Process Completed Successfully! ---")


# -----------------------
# USAGE EXAMPLE
# -----------------------
def test():
   
    etl = ETL()
    etl.run()
    return etl

if __name__ == "__main__":
    etl = ETL()
    etl.run()   