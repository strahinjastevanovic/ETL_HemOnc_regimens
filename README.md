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
3. Commit with trigger: `git commit -m "feat: changes [test-regression]"`
4. GitHub Action compares against baseline and publishes report

### Create/Update Baseline
1. Run pipeline: `./RunScript.sh -out baseline`
2. Create checkpoint: `python src/tests/regression/main.py --pkl baseline/`
    - this will setup the stage for baseline
3. Create and commit baseline: `python src/tests/regression/main.py --baseline v1.2.0`
   - Auto commits and pushes
   - GitHub Action updates `regression-data` branch

### Architecture
- **Staging**: `staging/` stores checkpoints locally
- **Baseline Storage**: Orphan branch `regression-data` (single commit, force-pushed)
- **Reports**: Published to GitHub Pages at `regression-reports/{commit-sha}/`
- **Comparison**: Jensen-Shannon divergence + Jaccard similarity metrics

See `src/tests/regression/research.md` for methodology details.
