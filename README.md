# Regimens Assembler
Pipeline for assembling regimens from HemOnc datasets (sigs table)

For strategy and details see `assets/Assembler_main.md`

## Setup

- You need to setup connection with OMOP CDM DB first. 
See `.env.template`.

- Install environment from `requirements.txt`

## Run 

Create regimens with the following command:

```
./RunScript.sh -out output-assembled
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

See `src/tests/regression/research.md` for methodology details.
