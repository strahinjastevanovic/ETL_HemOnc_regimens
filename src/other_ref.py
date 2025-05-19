import pandas as pd
import random

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



def generate_valid_drugs(regimen_tsv, ref_validdrugs, workdir="."):
    """
    Creates a valid drugs dataset by cross-referencing existing components 
    in the ETL process with a reference file.

    Parameters:
    etl_object (object): The ETL object containing data tables.
    ref_validdrugs (str): Path to the reference valid drugs TSV file.

    Returns:
    pd.DataFrame: Updated valid drugs dataframe.
    """

    # Load the final component table
    fin = pd.read_csv(regimen_tsv, sep='\t')
    components = fin['component'].str.lower().unique().tolist()
    
    # Read TSV as raw text and manually clean extra tabs
    with open(ref_validdrugs, "r", encoding="utf-8") as f:
        lines = [line.strip().split("\t")[:8] for line in f][1:]  # Truncate extra columns
    
    # Convert cleaned list to DataFrame
    ref = pd.DataFrame(lines)

    # Ensure the dataframe has exactly 8 columns
    expected_columns = ["name", "concept_id", "Manual", "concept_me", 
                        "valid_concept_id", "domain_id", "concept_class_id", "Manual_Req"]

    # Fix rows with more than 8 columns (truncate extra columns)
    ref = ref.iloc[:, :8]

    # Fix rows with less than 8 columns (fill missing values with None)
    ref = ref.replace("", pd.NA)
    ref = ref.fillna(pd.NA)

    # Assign proper column names
    ref.columns = expected_columns


    # Ensure column names are standardized
    required_columns = ["name", "concept_id", "Manual", "concept_me", 
                        "valid_concept_id", "domain_id", "concept_class_id", "Manual_Req"]

    if not set(required_columns).issubset(ref.columns):
        raise ValueError(f"Missing required columns in reference file: {set(required_columns) - set(ref.columns)}")

    # Normalize case for comparison &&
    # Identify known drugs already in reference
    known_df = ref[ref['name'].str.lower().isin(components)]
    print(f"Found: {known_df.shape}, Reference: {ref.shape}")

    # Find missing drugs to be added
    to_add = set(components) - set(known_df['name'].unique())

    # If there are new drugs, generate entries
    new_entries = []
    for i, component in enumerate(to_add, 1):
        name = component.strip().title()
        concept_id = pd.to_numeric(known_df['concept_id'], errors='coerce').max() + i if not known_df.empty else i


        manual = concept_id  # Same as concept_id
        concept_me = name  # Copy from name column
        valid_concept_id = pd.to_numeric(known_df['valid_concept_id'], errors='coerce').max() + i if not known_df.empty else i
        domain_id = "Drug"
        concept_class_id = "Ingredient"
        manual_req = "Yes"

        new_entries.append({
            "name": name,
            "concept_id": concept_id,
            "Manual": manual,
            "concept_me": concept_me,
            "valid_concept_id": valid_concept_id,
            "domain_id": domain_id,
            "concept_class_id": concept_class_id,
            "Manual_Req": manual_req
        })

    # Convert to DataFrame and concatenate
    if new_entries:
        new_df = pd.DataFrame(new_entries)
        updated_df = pd.concat([ref, new_df], ignore_index=True)
    else:
        updated_df = ref

    updated_df.to_csv(f"{workdir}/validdrugs.tsv", sep='\t', index=False)
    return updated_df






