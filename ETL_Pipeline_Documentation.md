# ETL Pipeline Documentation (Steps 1–5)

---

## Section Step 1 – Extraction and Preprocessing

### Overview  
This step loads the core raw HemOnc input tables — `indications` and `sigs`. These are cleaned, aligned, and processed into two foundational outputs:

- `i_frame.tsv` – Cleaned indications data  
- `s_frame.tsv` – Cleaned sigs data  

Both are essential for transformation logic later in the pipeline.

The process revolves around two important concepts:

- **SVC (Source Value Column):** the anchor column (e.g. `component`) used as the basis for matching.
- **HVC (Hidden Value Column):** a column that contains embedded or ambiguous data (e.g. similarity chains) which need to be exploded or disambiguated.

During processing, each SVC-HVC pair is iteratively compared, and if any HVC implies multiple mappings (e.g. similar drug aliases or variants), those records are **exploded** — meaning one row becomes many. The result is a fully expanded dataset with >20,000 records.

Additionally, a **modular blacklist system** is applied to remove irrelevant or noise entries early in the pipeline. This is JSON-based, easy to expand, and helps filter invalid terms by rule or pattern.

### Functional Breakdown

**Requirements**  
`nltk`, `scikit-learn`, `pandas`, `polars`, `numpy`, `regimen_auto_mapper.Nltk`
**Call script:** `pre_run.py`

- **Logging Setup**  
  Creates a persistent log file (`PRE.hve.log`) to record column matching decisions, similarity scores, and fuzzy lookups.

- **`col2col(t_col, frame)`**
  - Compares a single column (`t_col`) against all other columns in the frame using TF-IDF (char n-gram).
  - Produces a ranked similarity score between columns — used for schema inference, quality checks, or exploratory audits.

- **`val2col(column, frame, value, thr=0.02)`**
  - Compares a specific value against all values in a column.
  - Returns a filtered similarity map where entries above a threshold (`thr`) are retained — used in fuzzy entity matching.

- **`Runner` class**
  - Core class to initialize preprocessing from a root input folder.
  - Handles parsing, initial cleaning, and column alignment for `indications` and `sigs`.
  - Produces:
    - `i_frame.tsv` from indications
    - `s_frame.tsv` from sigs
  - Supports fast, compiled data handling for efficient downstream querying and processing..

---

## Section Step 2 – Transformation & Regimen Construction

### Overview  
This step merges the `i_frame.tsv` and `s_frame.tsv` tables into a unified dataset describing treatments, then applies intelligent logic to detect sequence structure and timing. The result is the creation of **regimen strings**, which represent ordered combinations of treatment components over time.

### Details on SRE (Shortest Repeating Element) module implementation

**Located in:** `RegStringHandler` and `sre_tools`

The **SRE** module is responsible for translating raw treatment data into ordered, time-aware **regimen strings**. 

#### Group Processing Logic

- Groups data by `regimen_cui` and `variant`
- Within each group:
  - Normalizes and deduplicates entries
  - Parses timing info (`allDays`, `cyclesigs`)
  - Constructs a binary event matrix representing drug administration timelines

#### Logging Setup  
- `SRE.process.log`: full debug output  
- `SRE.missing.log`: logs missing or incomplete groups  
- `SRE.output.log`: final string generation tracking

#### Component Vector Construction

- `build_component_vector()` creates a timeline-based binary vector for each drug, indicating on which days it is active.
- If no fixed cycle length (`csig=0`) is provided:
  - The function computes the full range between `min(idays)` and `max(idays)`.
  - For each active day, it generates a one-hot vector (using `np.eye`), offsets it based on the minimum day, and sums them to form the final binary vector.
- If a cycle length (`csig`) is defined:
  - Each day is treated as an index in a fixed-length vector of length `csig`, and the vector is built by summing corresponding one-hot encodings.


#### Event Matrix Collapse

- `collapse_event_matrix()` converts binary vectors into compact strings:
  - e.g. `"14.Pembrolizumab;0.Dexamethasone;1.Pembrolizumab"`

#### Multiple Representation Handling

- Handles irregular data by generating multiple fallback sequence representations:
  - `shortest`, `padded`, `truncated`, `longest`
- This is necessary because treatment components may have differing cycle lengths (`iday` vectors), missing days, or inconsistent duration tracking.
- When combining components into a unified event matrix, exact alignment can fail — this step ensures valid `regString` output by constructing the most reasonable approximation.
- The wrapper (`collapse_event_matrix_wrapper`) applies these strategies systematically and returns all viable outputs, even when the data is partially broken or uneven.

**WIP Note:**
- This fallback system is a temporary but robust strategy.
- A cleaner long-term solution involves **pre-aligning cycle lengths** or **subgrouping entries by normalized window**, so all combined components share compatible timing — eliminating the need for brute-force collapsing.
- Alternate vocabularies and structured encodings (e.g., those derived from **ATHENA** or other OMOP-compatible ontologies) are under active development to support higher fidelity matching and sequence grouping.


#### Output
- The primary output is:  `regimens.tsv` – a normalized, structured list of regimens with timing encoded
- Final result is a table with:
  - `regString`, `shortString`, `regimen_cui` (translated to regCode), `regimen` (translated to regName), etc.
- If multiple valid sequences exist, all are retained via row duplication

**WIP Note:**
- Other columns to be transformed:
    `regCodeExt`,
    `conditionCode`,
    `contextCode`,
    `day`,
    `cycleTaken`,
    `cycleLength`,
    `noCycles`,
    `branchInfo`,
    `Radio`,
    `continuous`,
    `noCycles_Original`
- However addition postponed due to optinal use



### Functional Breakdown

**Requirements**  
`pandas`, `tqdm`, custom tools: `RegStringHandler (SRE)`, `collapse_seq_naive`, `frame_ALL`

**Call script:** `etl_main.py`

- **`ETL.__init__()`**
  - Sets up directory references and a schema map for internal field naming.
  - Example mappings include:  
    `component`, `variant`, `regimen_cui`, `regString`, `condition`, etc.

- **`_load(csv_path)`**
  - Helper function to load `.tsv` files into DataFrames.

- **`process_frame(workdir)`**
  - Runs the transformation logic using:
    - `frame_all()` to combine inputs
    - `SRE_endpoint()` to generate final `regString` columns

---

## Section Step 3 – Reference Completion: Regimen Groups & Valid Drugs


### Overview  
This step ensures all new entries in the ETL output are mapped to known reference groups. If something is missing from the historical mappings, the system generates default assignments.

This serves as an **update mechanism** for supporting reference files.


### Functional Breakdown

**Requirements**
`pandas`, `random`

**Call script:** `other_ref.py`

- **`create_reg_group(etl_object, ref_reggroups)`**
  - Compares new regimens with a reference group list.
  - Assigns missing regimens to existing groups (randomized fallback).
  - Produces an updated `regimengroups.tsv`.

- **`create_valid_drugs(etl_object, ref_validdrugs)`**
  - Matches components from ETL output against a known list of valid drugs.
  - Appends new entries if not present.
  - Outputs `validdrugs.tsv`.

---

## Section Step 4 – Serialization for R

### Overview  
This step performs **final loading and serialization**. It converts tabular `.tsv` outputs into `.rda` format for use in R analytics. This is the last preparation step before R-based modeling or visualization.

### Functional Breakdown

**Requirements**
`R base`, `read.delim`, `save`

**Call script:** `build_rda.R`

- Reads:
  - `regimens.tsv` → `regimens.rda`
  - `validdrugs.tsv` → `validdrugs.rda`
  - `regimengroups.tsv` → `regimengroups.rda`
- Ensures `stringsAsFactors = FALSE`
- Saves outputs using native `save()` calls in R
- Outputs are stored in `.rda` format

---

## Section Step 5 – Validation and Regimen Harmonization

### Overview  
This step compares newly created regimens (`sigs2024`) to a legacy trusted set (`sigs2021`). The goal is to check for:

- Exact matches
- Acceptable deviations (partial matches)
- Mismatches

This ensures continuity across ETL versions and identifies issues caused by data drift or schema evolution.

### Key Behaviors

- **Tagging**
  - `sigs2021`: trusted reference
  - `sigs2024`: newly generated table

- **Normalization & Cleanup**
  - Lowercases all regimen names
  - Strips suffixes like `(tp)`, `(kr)`
  - Unifies separators (`and`, `&` → `,`)
  - Reorders components
  - Hardcoded corrections (e.g., skip `dexamethasone` if not core)

- **Comparison Logic**
  - Regimen name sets from old vs. new are compared
  - Report is generated with three levels of classification:

    - ✅ **Correct match:** regimen names and strings are identical or functionally equivalent
    - ⚠️ **Partial match:** regimen name exists in both but with structural string differences (e.g., same drugs, different context)
    - ❌ **Mismatch:** same regimen name maps to entirely different content — often due to conflicting variants or ambiguous mappings


### Functional Breakdown
**Requirements:**  
`pandas`, `os`, `sys`, `re`  
**Call script:** `validation_check.py`

- Loads:
  - `regimens.tsv` (2024)
  - `regimens_init.tsv` (2021 ref)
- Applies normalization:
  - Lowercase
  - Remove `(tp)`, `(kr)`
  - Replace `and`, `&` with `,`
  - Normalize order of components
  - Known fixups (e.g. strip `dexamethasone` in predefined contexts)

- Compares `regName` sets between old and new
- Generates validation summary at `${workdir}/validation/`


