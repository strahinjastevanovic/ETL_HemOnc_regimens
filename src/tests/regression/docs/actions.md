# Usage Examples:

## Requirements

`pandas`, `numpy`,`plotly`, `jinja2`

## Test Regression:

```
git commit -m "feat: new feature [test-regression]"
git push
```
Action trigger:`[test-regression]` runs regression test, deploys HTML to GitHub Pasges


## Baseline Creation/Update (Local):
```
python src/tests/regression/main.py --baseline <path/to_assembler_output_stable>/ <version, e.g. 1.3.0>
```
Action: Creates/updates baseline.pkl.gz in regression-data branch, appends to history.csv
with a default message `chore: update baseline [baseline v1.3.0]`

Note: GITHUB_TOKEN is automatically provided by GitHub Actions. No need to create a separate secret.

## Cleanup gh pages branch

**Clone gh-pages branch**
git clone -b gh-pages https://github.com/strahinjastevanovic/ETL_HemOnc_regimens.git gh-pages-cleanup
cd gh-pages-cleanup

**Remove old reports (keep last 10)**
cd regression-reports
ls -t | tail -n +11 | xargs rm -rf

**Commit and push**
git add .
git commit -m "chore: cleanup old regression reports"
git push
