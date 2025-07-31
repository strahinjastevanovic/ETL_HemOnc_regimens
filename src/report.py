import pandas as pd 

schema = {
    "Regimen-level" : {
        "Regimen name":[],
        "CUI":[],
        "Variants / regimens":[],
        "Handled in this version (%)":[],
        "Patched (%)":[],
        "Not handled in this version (%)":[]
    },
    "Variant-level" : {
        "Regimen name":[],
        "CUI":[],
        "Variant":[],
        "Status":[],
        "Pattern":[],
        "Field with a pattern": [],
    },
    "Dictionary":{
        "Term":[],
        "Description":[],
    },
    "Sumstats" :{
        "Status":["HANDLED", "PATCHED", "NOT HANDLED"],
        "# variants":[],
    }
}

from pathlib import Path
REPORT_TABLES = "/Users/home/prj/art-ohdsi/ETL_HemOnc_regimens/output_with_reports/report_tables"
tables = Path(REPORT_TABLES)

# skip_table   = ['single_part_sigs', "unique_groups"] 
# handle_table = ['preporc_cleaned']
# special_table = pd.read_csv(tables / "failed.csv")


# for table in tables.glob("*.tsv"):
#     table_name = table.stem

#     if table_name in skip_table:
#         continue 

#     print(f"Reading {table_name}")
#     df = pd.read_csv(table, sep="\t", low_memory=False)
#     print(df.shape)

#     handled     = {}
#     patched     = {}
#     unhandled   = {}

#     group_keys = ["regimen_cui", "variant_cui"]
#     if table_name != "prepoc_cleaned":
#         groups = df.groupby(group_keys).unique()
        
#         for name, group in groups:
#             unhandled.setdefault("Regimen name", group['regimen'] append)
#             unhandled.setdefault("CUI", group['regimen_cui'] append)
#             unhandled.setdefault("Variant", group['variant'] append) 
#             unhandled.setdefault("Status", ['unhandled'] append size of group)
#             unhandled.setdefault("Pattern", [name] append size of group)
#             unhandled.setdefault("Field with a pattern", ["WIP"] append size of group)
    
#     else: # preproc_cleaned
        
#         # filter group_keys ["regimen", "variant"] not in special_table
#         group_keys_filtered = [...]
#         groups = df.groupby(group_keys_filtered).unique()

#         for name, group in groups:
#             handled.setdefault("Regimen name", group['regimen'] append)
#             handled.setdefault("CUI", group['regimen_cui'] append)
#             handled.setdefault("Variant", group['variant'] append)
#             handled.setdefault("Status", ['HANDLED'] append size of group)
#             handled.setdefault("Pattern", ["-"] append size of group)
#             handled.setdefault("Field with a pattern", ["-"] append size of group)

#             for record in group:
#                 # patched define patterns: 
#                 # if group has any in fields:
#                 # timing_sequence contains \(.*\)
#                 # cycle_length_unit == indeterminate
#                 # cycle_length_ub contains "c" 
#                 patched.setdefault("Regimen name", group['regimen'] append)
#                 patched.setdefault("CUI", group['regimen_cui'] append)
#                 patched.setdefault("Variant", group['variant'] append)
#                 patched.setdefault("Status", ['PATCHED'] append size of group)
#                 patched.setdefault("Pattern", [<"try to match from above which one, e.g. cycle_length_unit == indeterminate">] append size of group)
#                 patched.setdefault("Field with a pattern", ["<try to match from above which one>, e.g. cycle_length_unit"] append size of group)


#     # finaly:

# final_report = pd.from_dicts( handled, patched, unhandled)

# final_report.to_csv(tables / "VARIANT_LEVEL.tsv", sep='\t')

skip_table = ['single_part_sigs', "unique_groups","regimen_variants_n_unique", "VARIANT_LEVEL", "null_group_keys"]
handle_table = ['preproc_cleaned']
special_table = pd.read_csv(tables / "failed.csv")

final_records = []
group_keys = ["regimen_cui", "variant"]


for table in tables.glob("*.tsv"):
    table_name = table.stem

    if table_name in skip_table:
        continue

    print(f"[INFO] Reading {table_name}")
    df = pd.read_csv(table, sep="\t", low_memory=False)
    print("[INFO] Shape:", df.shape)


    if table_name != "preproc_cleaned":
        # mark as UNHANDLED
        grouped = df.groupby(group_keys)
        for name, group in grouped:
            n = len(group)
            record = {
                "Regimen name": group["regimen"].iloc[0],
                "CUI": name[0],
                "Variant": name[1],
                "Status": "UNHANDLED",
                "Pattern": str(table_name),
                "Field with a pattern": "---",
            }
            final_records.extend([record] * n)

    # else:
        # # filtered set of groups not in special_table
        # special_keys = set(zip(special_table["regimen"], special_table["variant"]))
        # df_filtered = df[~df[["regimen", "variant"]].apply(tuple, axis=1).isin(special_keys)]

        # grouped = df_filtered.groupby(group_keys)
        # for name, group in grouped:
        #     n = len(group)
        #     base = {
        #         "Regimen name": group["regimen"].iloc[0],
        #         "CUI": name[0],
        #         "Variant": name[1],
        #     }

        #     final_records.extend([{
        #         **base,
        #         "Status": "HANDLED",
        #         "Pattern": "-",
        #         "Field with a pattern": "-"
        #     }] * n)

        #     # Now apply PATCHED pattern checks
        #     for _, row in group.iterrows():
        #         patterns = []
        #         fields = []

        #         if pd.notna(row.get("timing_sequence", "")) and pd.Series([row["timing_sequence"]]).str.contains(r"\(.*\)").any():
        #             patterns.append("Optional cycles")
        #             fields.append("timing_sequence")

        #         if str(row.get("cycle_length_unit", "")).lower() == "indeterminate":
        #             patterns.append("cycle length is indeterminate")
        #             fields.append("cycle_length_unit")

        #         if pd.notna(row.get("cycle_length_ub", "")) and "c" in str(row["cycle_length_ub"]):
        #             patterns.append("cycle length u.bound contains 'c'")
        #             fields.append("cycle_length_ub")

        #         if patterns:
        #             final_records.append({
        #                 **base,
        #                 "Status": "PATCHED",
        #                 "Pattern": "; ".join(patterns),
        #                 "Field with a pattern": "; ".join(fields)
        #             })


    else:
        special_keys = set(zip(
            special_table["regimen"].astype(str).str.strip(),
            special_table["variant"].astype(str).str.strip()
        ))

        df["regimen"] = df["regimen"].astype(str).str.strip()
        df["variant"] = df["variant"].astype(str).str.strip()

        df_filtered = df[~df[["regimen", "variant"]].apply(tuple, axis=1).isin(special_keys)]

        grouped = df_filtered.groupby(group_keys)

        for name, group in grouped:
            base = {
                "Regimen name": group["regimen"].iloc[0],
                "CUI": name[0],
                "Variant": name[1],
            }

            n = len(group)
            # Add one HANDLED row per row in group
            final_records.extend([{
                **base,
                "Status": "HANDLED",
                "Pattern": "-",
                "Field with a pattern": "-"
            }] * n)

            # Group-level pattern detection
            patterns = set()
            fields = set()

            for _, row in group.iterrows():
                if pd.notna(row.get("timing_sequence", "")) and pd.Series([row["timing_sequence"]]).str.contains(r"\(.*\)").any():
                    patterns.add("Optional cycles")
                    fields.add("timing_sequence")

                if str(row.get("cycle_length_unit", "")).lower() == "indeterminate":
                    patterns.add("cycle length is indeterminate")
                    fields.add("cycle_length_unit")

                if pd.notna(row.get("cycle_length_ub", "")) and "c" in str(row["cycle_length_ub"]):
                    patterns.add("cycle length u.bound contains 'c'")
                    fields.add("cycle_length_ub")

            if patterns:
                final_records.append({
                    **base,
                    "Status": "PATCHED",
                    "Pattern": "; ".join(sorted(patterns)),
                    "Field with a pattern": "; ".join(sorted(fields))
                })


# Create final report
final_report = pd.DataFrame(final_records)
# 🔍 Quick sanity check
dupes = final_report.groupby(["CUI", "Variant"]).nunique().reset_index()
bad = dupes[(dupes["Pattern"] > 1) | (dupes["Status"] > 1)]

if not bad.empty:
    key = bad.iloc[0][["CUI", "Variant"]]
    example = final_report[(final_report["CUI"] == key["CUI"]) & (final_report["Variant"] == key["Variant"])]
    print("[WARNING] Inconsistent entry detected:")
    print(example)
final_report.drop_duplicates().to_csv(tables / "VARIANT_LEVEL.tsv", sep="\t", index=False)
print("[INFO] Final report written.")



            


