import os
import re
import pandas as pd
import polars as pl
from tqdm import tqdm
import json


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"[^\w\s]", "", str(text)).strip().lower()


def proc_blist_naive(blist: list) -> list:
    blist = [l for subli in [s.split("(") for s in blist] for l in subli]
    return [s.strip(")").strip().lower() for s in blist]


def cleaning(frame, blacklist): # TODO - cleanify
    if blacklist:
        blacklist_p1 = proc_blist_naive(blacklist['custom'])
        blacklist_p2 = proc_blist_naive(blacklist['supportive_and_other_medications_abc'])
        blacklist = set(blacklist_p1 + blacklist_p2)
        return frame.filter(~pl.col("component").str.to_lowercase().is_in(blacklist)), blacklist
    else:
        return frame, None

def augment_regimen_groups(df1: pd.DataFrame, df2: pd.DataFrame, blacklist_items: dict) -> pd.DataFrame:
    pl_df1 = pl.from_pandas(df1)
    pl_df2 = pl.from_pandas(df2)

    meta_cols = ["regimen_cui", "variant", "allDays"]
    key_cols = ["component", "study"]
    df1_meta_cols = set(df1.columns) - set(key_cols)

    # Outer merge on ['component', 'study']
    merged = pl_df2.join(pl_df1, on=key_cols, how="left")

    recovered_map = {cui: group for cui, group in pl_df2.group_by("regimen_cui")}
    new_rows = []

    for reg_cui, group_data in tqdm(merged.group_by("regimen_cui"), desc="Augmenting by regimen_cui"):
        recovered = recovered_map.get(reg_cui)
        if recovered is None or recovered.is_empty():
            continue

        for (comp, study), k_data in group_data.group_by(key_cols):
            existing = k_data.select(list(df1_meta_cols))
            total_nulls = existing.null_count().to_series().sum()
            if total_nulls == len(existing.columns) * existing.height:
                df1_matches = pl_df1.filter(pl.col("component") == comp)
                if df1_matches.is_empty():
                    continue

                for row in recovered.iter_rows(named=True):
                    match_score = df1_matches.select(meta_cols).to_pandas().eq(
                        {k: row.get(k) for k in meta_cols}
                    ).sum(axis=1)
                    if not match_score.empty and match_score.max() >= 0.7 * len(meta_cols):
                        best_row = df1_matches.to_pandas().iloc[match_score.idxmax()].to_dict()
                        enrich_dict = {k: best_row[k] for k in df1_meta_cols if k in best_row}
                        combined = {**row, **enrich_dict}
                        new_rows.append(combined)

    if new_rows:
        enriched_df = pl.from_dicts(new_rows)
        merged = pl.concat([merged, enriched_df], how="vertical").unique()

    merged, blacklist = cleaning(merged, blacklist_items)

    print(f"🧼 Final shape after augmentation: {merged.shape}")
    return merged.to_pandas(), blacklist


def frame_all(
    indi_path="i_frame.tsv",
    sigs_path="s_frame.tsv",
    output_dir="results",
    supplementary={}
):
    os.makedirs(output_dir, exist_ok=True)

    indi_df = pd.read_csv(indi_path, sep='\t')
    sigs_df = pd.read_csv(sigs_path, sep='\t')

    for df in [indi_df, sigs_df]:
        df["component"] = df["component"].astype(str).apply(clean_text)
        df["study"] = df["study"].astype(str).apply(clean_text)

    indi_df["component"] = indi_df["component"].replace({"nan": None, "none": None})

    if not {"component", "study"}.issubset(sigs_df.columns):
        raise ValueError("❌ 'component' and/or 'study' columns missing in signature file.")

    for df in [indi_df, sigs_df]:
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).fillna("nan")

    df1 = indi_df.drop(columns=[col for col in indi_df.columns if "date_added" in col], errors='ignore')
    df2 = sigs_df.drop(columns=[col for col in sigs_df.columns if "date_added" in col], errors='ignore')

    if supplementary:
        blacklist_items = json.load(open(supplementary, "r"))

    mer_proc, blacklist = augment_regimen_groups(df1, df2, blacklist_items)

    assert mer_proc["component"].str.lower().isin(["thalidomide"]).any(), "❌ Thalidomide lost as final check!"
    assert not mer_proc['component'].str.lower().isin(['bisophonate']).any(), "❌ Bisophonate still present!"

    mer_proc.drop_duplicates(inplace=True)
    mer_proc['metaCondition'] = "all"
    mer_proc.to_csv(os.path.join(output_dir, "merged.tsv"), sep="\t", index=False)
