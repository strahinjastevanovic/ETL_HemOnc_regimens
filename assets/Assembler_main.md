# Assembler Pipeline Documentation (Steps 1–5)

---

## Section Step 1 – Vocabularies Preprocessing

This step loads the core raw HemOnc input tables — `sigs`. Additional information are derived from postgres mirror of Athena database which stratify data using standardized vocabularies and provide OMOP-compatible ontology. Output of the current step preprocessing:

- `sigs_with_conditions.tsv`   

In context of condition exploit, records are **exploded** — meaning one row becomes many. 

Additionally, a **modular blacklist system** is applied to remove irrelevant or noise entries early in the pipeline. This is JSON-based, easy to expand, and helps filter invalid terms by rule or pattern.

Modular blacklist system is expanded into full section with **idiosyncracies logs**, **handlers** and **autids**. The reporting system works as a side effect during initial steps of the pipeline. Output of the current step preprocessing:

- `reports.xlsx`  

**Requirements**  
Libraries used for processing: 
`pandas`, `polars`, `numpy`, ...

See other micro-requirements in the current repository.

## Section Step 2 – Transformation & Regimen Construction
  
This step applies intelligent logic to detect sequence structure and timing. The result is the creation of **regimen strings**, which represent ordered combinations of treatment components over time.

### Details on SRE (Shortest Repeating Element) module implementation

**Located in:** `RegStringHandler` and `sre_tools`

The **SRE** module is responsible for translating raw treatment data into ordered, time-aware **regimen strings**. The formatting process leads to creation "short string" which is regimen data stored as alignment ready string objects.

#### Group Processing Logic

- Groups definitions used in SRE:
* main keys ( `CUI`-s ) for condition, regimen, variant and component
* essential signature fields - timing sequence, allDays, cycle <TODO>...

Constructs a binary event matrix representing drug administration timelines

#### Component Vector Construction

...<TODO>

#### Event Matrix Collapse

...<TODO>

#### Multiple Representation Handling (Idiosincracies)

...<TODO>


#### Output
- The primary output is:  `regimens.tsv` – a normalized, structured list of regimens with timing encoded
- Final result is a table with:
  - `regString`, `shortString`, `regimen_cui` (translated to regCode), `regimen` (translated to regName), etc.
- The final output regimens are deduplicated per unique short strings only, thus loss of group main keys is active 

Future note: Addition of main key Group safekeeping and mapping depending on use case.

- Transformed columns:
<TODO>
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


### Functional Breakdown

**Requirements**  
`pandas`, `tqdm`, custom tools: `RegStringHandler (SRE)`, `collapse_seq_naive`, `frame_ALL`

**Call script:** `etl_main.py`

...<TODO>
---

## Section Step 3 – Reference Completion: Regimen Groups & Valid Drugs

This step ensures all new entries in the pipeline output are mapped to known reference groups. 


### Functional Breakdown

**Requirements**
`pandas`, `random`

...
---

## Section Step 4 – Serialization (R)

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
<TODO>
---

## Section Step 5 – Validation and Regimen Harmonization (Legacy code)

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
  - `regimens_legacy.tsv` (2021 ref)
- Compares `regName` sets with legacy strings
- Generates validation summary at `${workdir}/validation/`


