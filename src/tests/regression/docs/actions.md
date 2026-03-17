# Usage Examples:

## Requirements

`pandas`, `numpy`,`plotly`, `jinja2`

## Test Regression:

```
git commit -m "feat: new feature [test-regression]"
git push
```
Action: Triggers regression test, deploys HTML to GitHub Pages


## Baseline Creation/Update (Local):
```
python src/tests/regression/main.py --baseline <path/to_assembler_output_stable>/ <version, e.g. 1.3.0>
```
Action: Creates/updates baseline.pkl.gz in regression-data branch, appends to history.csv
with a default message `chore: update baseline [baseline v1.3.0]`

Note: GITHUB_TOKEN is automatically provided by GitHub Actions. No need to create a separate secret.
