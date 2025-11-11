import pandas as pd
import random
from pipeline.log import Logger

def generate_reg_group(regimen_tsv, ref_reggroups, workdir="."):
    """
    Assigns regimen groups to new regimens not present in the reference file.

    Parameters:
    etl_object (object): The ETL object containing data tables.
    ref_reggroups (str): Path to the reference regimen groups TSV file.

    Returns:
    pd.DataFrame: Updated regimen groups dataframe.
    """

    # Load final table from ETL object
    df = pd.read_csv(regimen_tsv, sep='\t')

    # Load reference regimen groups dataset
    ref = pd.read_csv(ref_reggroups, sep='\t')

    # Ensure required columns exist
    required_columns = ["Var1", "regGroup"]
    if not set(required_columns).issubset(ref.columns):
        raise ValueError(f"Missing required columns in reference file: {set(required_columns) - set(ref.columns)}")

    # Normalize case and extract known regimen names
    known_var1 = set(ref["Var1"].unique())

    # Extract unique regimen names from the ETL dataset
    new_regimens = set(df["regName"].unique())

    # Find regimens that are not in the reference
    to_add = new_regimens.difference(known_var1)

    # If there are new regimens, generate entries
    if to_add:
        new_entries = pd.DataFrame({
            "Var1": list(to_add),
            "regGroup": [random.choice(ref["regGroup"].dropna().unique()) for _ in to_add]
        })

        # Concatenate new data
        updated_df = pd.concat([ref, new_entries], ignore_index=True)
    else:
        updated_df = ref
    updated_df.to_csv(f"{workdir}/regimengroups.tsv", sep='\t', index=False)
    return updated_df

def get_ref_columns(ref_validdrugs):
    col = pd.read_csv(ref_validdrugs, sep="\t", nrows=0).columns.tolist()
    scol = sorted(col)
    return scol, col

VALID_DRUGS_RELMAP = {
        "name":"concept_name",
        "concept_id":"concept_id",
        "Manual": "concept_id",
        "concept_me":"concept_name",
        "valid_concept_id": "valid_concept_id",
        "domain_id": "domain_id",
        "concept_class_id" : "concept_class_id",
        "Manual_Req": "invalid_reason"
    }



def generate_valid_drugs(regimen_tsv, validdrugs_query, workdir="."):
    """
    Creates a valid drugs dataset by remaping query_table to output valid drugs table.
    But also checks whether each queried drug exist in regimen_tsv (full format)
    and logs incosistancies. 

    Parameters:
    regimen_tsv (str): Path to final regimen table TSV file.
    query_table (str): Path to Valid drugs vocab query

    Returns:
    pd.DataFrame: Updated valid drugs dataframe.
    """

    # Load the final component table
    fin = pd.read_csv(regimen_tsv, sep='\t')
    components_lower = fin['component'].str.lower().unique().tolist()
    
    vd_query = pd.read_csv(validdrugs_query)
    vd_components_lower = vd_query['concept_name'].str.lower().unique().tolist()

    # log discrepancies
    sc = set(components_lower)
    vd = set(vd_components_lower)

    logger     = Logger(f"{workdir}/logs", filename="serialize.log", )
    logger.info(f"\n[INFO] components loaded from HemOnc: {len(sc)}\n"
          f"[INFO] components loaded from athena: {len(vd)}\n"
          f"[INFO] shared: {len(vd.intersection(sc))}\n"
          f"[INFO] in HemOnc: {len(vd.difference(sc))}\n"
          f"[INFO] in Athena: {len(sc.difference(vd))}\n"
          )
    
    # remap
    for new_col, src_col in VALID_DRUGS_RELMAP.items():
        if new_col:
            vd_query[new_col] = vd_query[src_col].values
    
    out_cols = list(filter(None, VALID_DRUGS_RELMAP.keys()))

    if not set(out_cols).issubset(vd_query.columns):
        raise ValueError(f"Missing required columns in reference file: {set(out_cols) - set(vd_query.columns)}")

    vd_query = vd_query[out_cols]
    vd_query.to_csv(f"{workdir}/validdrugs.tsv", sep='\t', index=False)
    return vd_query






