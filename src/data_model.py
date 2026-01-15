import pandas as pd
import random
from process.log import Logger

def generate_reg_group(regimen_tsv, ref_reggroups, workdir="."):
    """
    Assigns regimen groups to new regimens not present in the reference file.
    Logs regimen group assignment activity.

    Parameters:
    regimen_tsv (str): Path to final regimen table TSV file.
    ref_reggroups (str): Path to the reference regimen groups TSV file.
    workdir (str): Working directory for output files and logs.

    Returns:
    pd.DataFrame: Updated regimen groups dataframe with new entries.
    """
    
    # Initialize logger
    logger = Logger(f"{workdir}/logs", filename="serialize.log")
    
    logger.info(f"\n[REG GROUP ASSIGNMENT] Processing regimen group mappings")

    # Load final table from ETL pipeline
    df = pd.read_csv(regimen_tsv, sep='\t')
    logger.info(f"[INFO] Loaded regimen TSV: {regimen_tsv} ({len(df)} rows)")

    # Load reference regimen groups dataset
    ref = pd.read_csv(ref_reggroups, sep='\t')
    logger.info(f"[INFO] Loaded reference regimen groups: {ref_reggroups} ({len(ref)} rows)")

    # Ensure required columns exist
    required_columns = ["Var1", "regGroup"]
    if not set(required_columns).issubset(ref.columns):
        missing = set(required_columns) - set(ref.columns)
        logger.error(f"[ERROR] Missing required columns: {missing}")
        raise ValueError(f"Missing required columns in reference file: {missing}")

    # Extract known regimen names
    known_var1 = set(ref["Var1"].unique())
    logger.info(f"[INFO] Known regimen names in reference: {len(known_var1)}")

    # Extract unique regimen names from ETL dataset
    new_regimens = set(df["regName"].unique())
    logger.info(f"[INFO] Unique regimen names in ETL dataset: {len(new_regimens)}")

    # Find regimens that are not in the reference
    to_add = new_regimens.difference(known_var1)
    logger.info(f"[INFO] New regimens to assign (not in reference): {len(to_add)}")

    # If there are new regimens, generate entries with random group assignment
    if to_add:
        available_groups = ref["regGroup"].dropna().unique()
        new_entries = pd.DataFrame({
            "Var1": list(to_add),
            "regGroup": [random.choice(available_groups) for _ in to_add]
        })
        logger.info(f"[INFO] Generated {len(new_entries)} new regimen group entries")
        
        # Show sample of new assignments
        for idx, row in new_entries.head(5).iterrows():
            logger.debug(f"  → {row['Var1']} → group {row['regGroup']}")

        # Concatenate new data
        updated_df = pd.concat([ref, new_entries], ignore_index=True)
        logger.info(f"[INFO] Updated regimen groups: {len(ref)} reference + {len(new_entries)} new = {len(updated_df)} total")
    else:
        updated_df = ref
        logger.info(f"[INFO] No new regimens to add; using reference as-is")
    
    # Save output
    updated_df.to_csv(f"{workdir}/regimengroups.tsv", sep='\t', index=False)
    logger.info(f"[INFO] Regimen groups saved to {workdir}/regimengroups.tsv ({len(updated_df)} rows)")
    
    return updated_df

VALID_DRUGS_RELMAP = {
    "name": "concept_name",
    "concept_id": "concept_id",
    "Manual": "concept_id",
    "concept_me": "concept_name",
    "valid_concept_id": "valid_concept_id",
    "domain_id": "domain_id",
    "concept_class_id": "concept_class_id",
    "Manual_Req": "invalid_reason",
}

def include_route_info(regimens_tsv_full, validdrugs):
    """
    Include route information in the valid drugs dataframe, exploded by route per component.
    
    Maps unique routes for each component from the full regimen table. Output is one row
    per drug-route combination, preserving regimen context and component mapping.
    
    Parameters:
    regimens_tsv_full (pd.DataFrame): Full regimen dataframe with 'component', 'component_cui', 'route', 'regName' columns.
    validdrugs (pd.DataFrame): Valid drugs dataframe with component mappings.
    
    Returns:
    pd.DataFrame: Valid drugs dataframe exploded by route with one row per drug-route pair.
    
    Output structure:
        component_cui | name | route | regimen
        DrugA         | ...  | Oral  | RegName1
        DrugA         | ...  | IV    | RegName2
        DrugB         | ...  | Oral  | RegName1
        DrugB         | ...  | SC    | RegName3
        DrugC         | ...  | Not specified | ...
    """
    # Get component name to cui mapping from regimens
    component_name_to_cui = (
        regimens_tsv_full[['component', 'component_cui']]
        .drop_duplicates()
        .set_index('component')['component_cui']
        .to_dict()
    )
    
    # Build route data: for each component, extract unique routes and regimen contexts
    route_data = []
    
    for component_name, cui in component_name_to_cui.items():
        # Get all rows for this component
        component_rows = regimens_tsv_full[regimens_tsv_full['component'] == component_name]
        
        # Extract unique route-regimen pairs for this component
        route_regimen_pairs = component_rows[['route', 'regName']].drop_duplicates()
        
        # Filter out null/placeholder routes
        exclude_routes = {"Not specified", "nan", None, ""}
        route_regimen_pairs = route_regimen_pairs[
            ~route_regimen_pairs['route'].astype(str).isin(exclude_routes)
        ]
        
        # If no valid routes, add "Not specified" default
        if route_regimen_pairs.empty:
            route_data.append({
                'component_cui': cui,
                'component_name': component_name,
                'route': 'Not specified',
                'regimen': None
            })
        else:
            # Add each unique route-regimen pair
            for _, row in route_regimen_pairs.iterrows():
                route_data.append({
                    'component_cui': cui,
                    'component_name': component_name,
                    'route': row['route'],
                    'regimen': row['regName']
                })
    
    # Create route dataframe
    route_df = pd.DataFrame(route_data)
    
    # Merge with validdrugs on component_cui
    # validdrugs should have one row per component; we'll join and explode by route
    result = validdrugs.merge(
        route_df[['component_cui', 'route', 'regimen']],
        on='component_cui',
        how='left'
    )
    
    # Fill null routes with "Not specified"
    result['route'] = result['route'].fillna('Not specified')
    
    # Drop duplicates that may arise from the merge
    result = result.drop_duplicates(subset=['component_cui', 'route', 'regimen'], keep='first')
    
    return result

def generate_valid_drugs(regimen_tsv, validdrugs_query, workdir="."):
    """
    Creates a valid drugs dataset by remapping query_table to output valid drugs table.
    Also validates that each regimen component has a corresponding valid OMOP concept.
    Logs validation results and missing component mappings.

    Parameters:
    regimen_tsv (str): Path to final regimen table TSV file.
    validdrugs_query (str): Path to Valid drugs vocab query (CSV format from Athena).

    Returns:
    pd.DataFrame: Updated valid drugs dataframe with remapped columns.
    """

    # Load the final regimen component table from pipeline
    fin = pd.read_csv(regimen_tsv, sep='\t')
    
    # Get unique components from regimens (case-insensitive)
    unique_components = fin['component'].unique().tolist()
    components_set = set(comp.lower() for comp in unique_components)
    
    # Load valid drugs from Athena vocabulary query
    vd_query = pd.read_csv(validdrugs_query)
    vd_concepts = set(comp.lower() for comp in vd_query['concept_name'].unique().tolist())

    # Calculate overlap and gaps
    shared_components = components_set.intersection(vd_concepts)
    unmapped_components = components_set.difference(vd_concepts)
    unmapped_count = len(unmapped_components)

    # Initialize logger
    logger = Logger(f"{workdir}/logs", filename="serialize.log")
    
    # Log validation summary
    logger.info(f"\n[VALIDATION] Valid Drugs Component Mapping")
    logger.info(f"[INFO] Total unique components in regimens: {len(components_set)}")
    logger.info(f"[INFO] Total unique concepts from Athena: {len(vd_concepts)}")
    logger.info(f"[INFO] Mapped components (in both): {len(shared_components)}")
    logger.info(f"[INFO] Unmapped components (in regimens only): {unmapped_count}")
    
    # Log details of unmapped components if any
    if unmapped_components:
        logger.warning(f"\n[WARNING] {unmapped_count} component(s) not found in valid drugs vocabulary:")
        for comp in sorted(unmapped_components):
            # Find original case in regimens for logging
            original = next((c for c in unique_components if c.lower() == comp), comp)
            logger.warning(f"  - {original}")
        
        # Provide context about unmapped component types
        logger.info(f"\n[NOTE] Unmapped components typically include:")
        logger.info(f"  • New/experimental drugs not yet in Athena (e.g., Lumrotatug, Datopotamab deruxtecan)")
        logger.info(f"  • Non-drug interventions (e.g., External beam radiotherapy, BCG vaccine)")
        logger.info(f"  • Biological/cell therapies (e.g., Allogeneic stem cells, Granulocyte colony-stimulating factor)")
        logger.info(f"  • Treatment categories (e.g., Androgen-deprivation therapy)")
        logger.info(f"\n[NOTE] All components are retained in the pipeline. Unmapped entries will not match OMOP concepts")
        logger.info(f"      but will remain in regimen definitions for informational purposes.")
    else:
        logger.info(f"\n[SUCCESS] All components have valid OMOP concept mappings!")
    
    # Remap columns according to VALID_DRUGS_RELMAP
    for new_col, src_col in VALID_DRUGS_RELMAP.items():
        if new_col and src_col in vd_query.columns:
            vd_query[new_col] = vd_query[src_col].values
    
    out_cols = list(filter(None, VALID_DRUGS_RELMAP.keys()))

    if not set(out_cols).issubset(vd_query.columns):
        missing = set(out_cols) - set(vd_query.columns)
        raise ValueError(f"Missing required columns in valid drugs query: {missing}")

    vd_query = vd_query[out_cols]

    logger.info(f"\n[INFO] Valid drugs output columns: {out_cols}")
    logger.info(f"[INFO] Valid drugs table shape before route explosion: {vd_query.shape}")
    
    # Include route information from regimens (exploded by route-regimen pair)
    vd_complete = include_route_info(fin, vd_query)
    
    logger.info(f"[INFO] Valid drugs table shape after route explosion: {vd_complete.shape}")
    logger.info(f"[INFO] Output columns: {vd_complete.columns.tolist()}")

    # Save output
    vd_complete.to_csv(f"{workdir}/validdrugs.tsv", sep='\t', index=False)
    logger.info(f"[INFO] Valid drugs table saved to {workdir}/validdrugs.tsv ({vd_complete.shape[0]} rows)")
    
    return vd_complete






