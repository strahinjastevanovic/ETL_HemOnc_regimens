# Regimens Assembler
Pipeline for assembling regimens from HemOnc datasets (sigs table)

For strategy and details see `assets/Assembler_main.md`

## Setup

Install dependencies from `requirements.txt`:

```
pip install -r requirements.txt
```

Copy `.env.template` to `.env` and configure the Athena vocabulary source:

```
cp .env.template .env
```

### Athena vocabulary source (`DB_FRESH`)

The pipeline needs three Athena/OMOP vocabulary files
(`condition_concepts.csv`, `drug_concepts.csv`, `sigs_w_conditions.csv`).
How they are obtained is controlled by the `DB_FRESH` flag in `.env`:

| `DB_FRESH` | Behaviour |
|---|---|
| `FALSE` *(default)* | Downloads the pre-built snapshot bundle from the [`athena_mirrors`](https://github.com/strahinjastevanovic/ETL_HemOnc_regimens/tree/athena_mirrors) branch and unpacks the CSVs directly into your output directory. **No database credentials required.** |
| `TRUE` | Runs live SQL queries against your OMOP CDM instance. Requires `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_NAME` to be set in `.env`. |

**Default (no DB setup needed):**
```dotenv
DB_FRESH=FALSE
```

**Live queries:**
```dotenv
DB_FRESH=TRUE
DB_USER=myuser
DB_PASSWORD=secret
DB_HOST=db.example.com
DB_NAME=omop_cdm
```

## Run 

Create regimens with the following command:

```
./RunScript.sh -out output.assembled
```

## Regression Testing

Automated regression testing with GitHub Actions:

### Test Changes
1. Run pipeline locally: `./RunScript.sh -out output/`
2. Create checkpoint: `python src/tests/regression/main.py --pkl output/`
3. Add: `git add regression_stage`
4. Commit with trigger: `git commit -m "feat: changes [test-regression]"`
5. GitHub Action compares against baseline and publishes report
6. (Optional) `git pull --rebase` 

### Create/Update Baseline
1. Run pipeline: `./RunScript.sh -out baseline/`
2. Create and commit baseline: `python src/tests/regression/main.py --baseline v1.2.0 baseline/`
   - Auto commits and pushes `regression-data` branch

### Repo Setup Rules
- **Staging**: `regression_staging/` local checkpoints (GH actions input creation)
- **Baseline Storage**: Orphan branch `regression-data`
- **Reports**: Published to GitHub Pages at `regression-reports/{commit-sha}/`
- **Comparison**: Drift detection

See `src/tests/regression/docs` for methodology details.
