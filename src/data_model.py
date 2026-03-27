import pandas as pd
import random
from process.log import Logger

def generate_reg_group(regimen_tsv, ref_reggroups, workdir="."):
    """
    Assigns regimen groups to new regimens not present in the reference file.

    Parameters:
    etl_object (object): The ETL object containing data tables.
    ref_reggroups (str): Path to the reference regimen groups TSV file.

    Returns:
    pd.DataFrame: Updated regimen groups dataframe.
    """

    df = pd.read_csv(regimen_tsv, sep='\t')

    ref = pd.read_csv(ref_reggroups, sep='\t')

    required_columns = ["Var1", "regGroup"]
    if not set(required_columns).issubset(ref.columns):
        raise ValueError(f"Missing required columns in reference file: {set(required_columns) - set(ref.columns)}")

    known_var1 = set(ref["Var1"].unique())

    new_regimens = set(df["regName"].unique())

    to_add = new_regimens.difference(known_var1)

    if to_add:
        new_entries = pd.DataFrame({
            "Var1": list(to_add),
            "regGroup": [random.choice(ref["regGroup"].dropna().unique()) for _ in to_add]
        })

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



def generate_route_table(regimens_tsv_full, workdir="."):
    """
    Include route information in `drugs` dataframe, exploded by route per component.
    Note: Serialized under regimens.drugs object.

    Maps unique routes for each component from the full regimen table. Output is one row
    per drug-route combination, preserving regimen context and component mapping.

    Returns:
    pd.DataFrame: drugs dataframe exploded by route with one row per drug-route pair.
    Writes:
        {workdir}/regimens_drugs.tsv        — full with cui + drug + route + regimen
        {workdir}/regimens_drugs_deploy.tsv — deploy subset: regimen + drug + route
    """
    logger = Logger(f"{workdir}/logs", filename="lineage.log")
    logger.info(f"\n[ROUTE ASSIGNMENT] Processing route table from {regimens_tsv_full}")

    df = pd.read_csv(regimens_tsv_full, sep='\t')

    component2CUI = (
        df[['component', 'componentCode']]
        .drop_duplicates()
        .set_index('component')['componentCode']
        .to_dict()
    )

    route_data = []
    exclude_routes = {"Not specified", "nan", None, ""}

    for component_name, cui in component2CUI.items():
        component_rows = df[df['component'] == component_name]
        route_regimen_pairs = component_rows[['route', 'regName']].drop_duplicates()

        if len(component_rows) > len(route_regimen_pairs):
            logger.info(
                f"[ROUTE EXPLOSION] Component '{component_name}' (CODE: {cui}) "
                f"has {len(component_rows)} rows but {len(route_regimen_pairs)} "
                f"unique route-regimen pairs."
            )

        valid_pairs = route_regimen_pairs[
            ~route_regimen_pairs['route'].astype(str).isin(exclude_routes)
        ]

        if valid_pairs.empty:
            route_data.append({
                'cui': cui,
                'drug': component_name,
                'route': 'Not specified',
                'regimen': None
            })
        else:
            for _, row in valid_pairs.iterrows():
                route_data.append({
                    'cui': cui,
                    'drug': component_name,
                    'route': row['route'],
                    'regimen': row['regName']
                })

    route_df = pd.DataFrame(route_data)

    route_df.to_csv(f"{workdir}/regimens_drugs.tsv", sep='\t', index=False)
    logger.info(f"[LINEAGE] - Routes export to {workdir}/regimens_drugs.tsv")

    route_df_deploy = route_df[["regimen", "drug", "route"]]
    route_df_deploy.to_csv(f"{workdir}/regimens_drugs_deploy.tsv", sep='\t', index=False)
    logger.info(f"[LINEAGE] - Deploy subset export to {workdir}/regimens_drugs_deploy.tsv")

    return route_df


def generate_shortString_table(regimens_tsv_full, workdir="."):
    """
    Build a shortString lookup table from the full regimen table.

    Output structure:
        shortString_ID | shortString | regimen | condition | repeats

    Writes:
        {workdir}/regimens_shortStrings.tsv
    """
    logger = Logger(f"{workdir}/logs", filename="lineage.log")
    logger.info(f"\n[SHORT STRING] Building shortString table from {regimens_tsv_full}")

    df = pd.read_csv(regimens_tsv_full, sep='\t')

    shortString_df = (
        df.groupby("shortString")[['regName', 'condition']]
        .value_counts()
        .reset_index()
        .rename(columns={"count": "repeats", "regName": "regimen"})
    )

    codes, _ = pd.factorize(shortString_df['shortString'])
    shortString_df.insert(0, 'shortString_ID', codes + 1)

    shortString_df.to_csv(f"{workdir}/regimens_shortStrings.tsv", sep='\t', index=False)
    logger.info(f"[LINEAGE] - shortString export to {workdir}/regimens_shortStrings.tsv")

    return shortString_df


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

    fin = pd.read_csv(regimen_tsv, sep='\t')
    components_lower = fin['component'].str.lower().unique().tolist()
    
    vd_query = pd.read_csv(validdrugs_query)
    vd_components_lower = vd_query['concept_name'].str.lower().unique().tolist()

    sc = set(components_lower)
    vd = set(vd_components_lower)

    logger = Logger(f"{workdir}/logs", filename="lineage.log")
    logger.info(f"\n[INFO] components loaded from HemOnc: {len(sc)}\n"
          f"[INFO] components loaded from athena: {len(vd)}\n"
          f"[INFO] shared: {len(vd.intersection(sc))}\n"
          f"[INFO] in HemOnc: {len(vd.difference(sc))}\n"
          f"[INFO] in Athena: {len(sc.difference(vd))}\n"
          )
    
    for new_col, src_col in VALID_DRUGS_RELMAP.items():
        if new_col:
            vd_query[new_col] = vd_query[src_col].values
    
    out_cols = list(filter(None, VALID_DRUGS_RELMAP.keys()))

    if not set(out_cols).issubset(vd_query.columns):
        raise ValueError(f"Missing required columns in reference file: {set(out_cols) - set(vd_query.columns)}")

    vd_query = vd_query[out_cols]
    vd_query.to_csv(f"{workdir}/validdrugs.tsv", sep='\t', index=False)
    return vd_query






