import pandas as pd 
import polars as pl
from pathlib import Path
import os
import re
import json



### ////////////////////////////////////////////////////////////////////////////////////////
### domain class 

class Reporter:
    def __init__(self, output):
        self.output = output
        self.settings = {
            "field_col" : "Field",
            "pattern_col" : "Pattern"
        }
        self.enc = {
            "r" : "regimen","rc" : "regimenCUI", "vc" : "variantCUI","v" : "variant",
            "c" : "condition","cc" : "conditionCUI", "ad" : "allDays", "lb" : "cycleLengthLB",
             "ub" : "cycleLengthUB", "unit" : "cycleLengthUnit", "ts" : "timingSequence",
             "sn":"stepNumber", 'cr': "componentRole"
            # ... 
        }
        self.decode = lambda x: re.sub(r"\{.*?\}", self._replace_match, x)
        self.add = {
            "H" : "HANDLED",     # kept as is
            "P" : "PATCHED",     # kept with a temporary fix
            "N" : "NOT HANDLED", # dropped
        }

        os.makedirs(self.output, exist_ok=True)

    def _replace_match(self, match):
        key = match.group(0)  
        return self.enc.get(key, key)  

    def to_tsv(self, frame, file_name):
        frame.write_csv(f"{self.output}/{file_name}.tsv", separator="\t")
    
    def resolve(self, frame, file_name, pattern=None, field=None, status=None):
        if pattern and field and status:
            frame = frame.with_columns([
                pl.lit(self.decode(pattern) ).alias("Pattern"),
                pl.lit(self.decode(field)   ).alias("Field"),
                pl.lit(self.add[status]     ).alias("Status")
            ])
        else:
            raise TypeError(f"Please specify a value for the param:\nINPUT: pattern={pattern} ; field={field} ; status={status}. ")

        frame.write_csv(f"{self.output}/{file_name}.resolved.tsv", separator="\t")


### ////////////////////////////////////////////////////////////////////////////////////////
### adapter functions

def group_adapt_pl(frame):
    variant_cui_nunique = frame.select(["regimen_cui", "variant_cui"]).unique().height
    regimen_cui_nunique = frame.select("regimen_cui").unique().height

    return frame.with_columns([
        pl.lit(variant_cui_nunique).alias("variant_cui_nunique"),
        pl.lit(regimen_cui_nunique).alias("regimen_cui_nunique")
    ])

def group_adapt(df):
    variant_cui_nunique = df[["regimen_cui", "variant_cui"]].drop_duplicates().shape[0]
    regimen_cui_nunique = df["regimen_cui"].nunique()

    df["variant_cui_nunique"] = variant_cui_nunique
    df["regimen_cui_nunique"] = regimen_cui_nunique

    return df

def get_sum(df):
    total = len(df)
    print(df.columns)

    status_counts = (
        df["Status"]
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="count")
    )
    status_counts["perc"] = round((status_counts["count"] / total) * 100, 2)
    status_counts["Status_initial"] = status_counts["Status"].str[0]

    # Add columns back to df
    for _, row in status_counts.iterrows():
        col_name = f"Status_{row['Status_initial']}_perc"
        df[col_name] = row["perc"]

    return df

def get_sum_pl(frame):
    total = frame.height

    status_stats = (
        frame.group_by("Status")
             .agg(pl.count().alias("c"))
             .with_columns([
                 (pl.col("count") / total * 100).round(2).alias("perc"),
                 pl.col("Status").str.slice(0, 1).alias("Status_initial")
             ])
    )

    for row in status_stats.iter_rows(named=True):
        col_name = f"Status_{row['Status_initial']}_perc"
        frame = frame.with_columns(
            pl.lit(row['perc']).alias(col_name)
        )

    return frame


### ////////////////////////////////////////////////////////////////////////////////////////
### publisher functions

import pandas as pd
from pathlib import Path

def apply_operations(frame, schema, adapters=[]):
    for adapt in adapters:
        frame = adapt(frame)
    frame.rename(columns=schema, inplace=True)
    return frame[[*schema.values()]]

def load(frame_path):
    return pd.read_csv(frame_path, sep='\t', index_col=False, low_memory=False)

def write_frame_to_excel_sheet(path, frame, sheet_name):
    if os.path.exists(path): # Dev note: default cleanup in case xlsx corrupted 
        os.remove(path)
    try:
        with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            frame.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
    except FileNotFoundError:
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            frame.to_excel(writer, sheet_name=sheet_name, index=False)

def build_reports(sheets, report_tables_path: str, output_dir: str):
    report_tables_path = Path(report_tables_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)
    out_file = output_dir / "reports.xlsx"
    sheets = json.load(open(sheets, "r"))['sheets']
    
    print("[INFO] Sheets processed:")
    for sheet in sheets:
        print(sheet)
        sheet_name = sheet["name"]
        schema = sheet["headers"]
        adapters = [group_adapt, get_sum]

        # write headers first
        headers_df = pd.DataFrame(columns=[*schema.values()])
        write_frame_to_excel_sheet(out_file, headers_df, sheet_name)

        for table in report_tables_path.glob(".resolved.tsv"):
            frame = load(table)
            frame = apply_operations(frame, schema, adapters)
            frame = frame[headers_df.columns].drop_duplicates()
            write_frame_to_excel_sheet(out_file, frame, sheet_name)
     
