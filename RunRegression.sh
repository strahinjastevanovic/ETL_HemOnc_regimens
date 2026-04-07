#!/bin/bash
set -e

# === .RunRegression.sh ===
# Local regression runner: compares a reference output against a new output.
#
# Usage:
#   ./.RunRegression.sh -ref <ref_dir> -new <new_dir> [-out <output_dir>] [-n <run_number>]
#
# Defaults:
#   -out  output.regression_tests
#   -n    auto-increments based on existing final_output.*.json files
#
# Example:
#   ./.RunRegression.sh -ref output.3.noblfilt -new output.workup.iter

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGRESSION_SRC="${SCRIPT_DIR}/src/tests/regression"

# --- Parse args ---
OUTPUT_DIR="output.regression_tests"
RUN_N=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -ref)
      REF_DIR="$2"
      shift 2
      ;;
    -new)
      NEW_DIR="$2"
      shift 2
      ;;
    -out)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -n)
      RUN_N="$2"
      shift 2
      ;;
    *)
      echo "%%ERR%% Unknown option: $1"
      echo "Usage: ./.RunRegression.sh -ref <ref_dir> -new <new_dir> [-out <output_dir>] [-n <run_number>]"
      exit 1
      ;;
  esac
done

if [ -z "$REF_DIR" ] || [ -z "$NEW_DIR" ]; then
  echo "%%ERR%% -ref and -new are required."
  echo "Usage: ./.RunRegression.sh -ref <ref_dir> -new <new_dir> [-out <output_dir>] [-n <run_number>]"
  exit 1
fi

if [ ! -d "$REF_DIR" ]; then
  echo "%%ERR%% Reference directory not found: $REF_DIR"
  exit 1
fi

if [ ! -d "$NEW_DIR" ]; then
  echo "%%ERR%% New directory not found: $NEW_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# --- Auto-increment run number ---
if [ -z "$RUN_N" ]; then
  RUN_N=1
  while [ -f "${OUTPUT_DIR}/final_output.${RUN_N}.json" ]; do
    RUN_N=$((RUN_N + 1))
  done
fi

COMPARISON_JSON="${OUTPUT_DIR}/final_output.${RUN_N}.json"
HTML_OUT="${OUTPUT_DIR}/final_output.${RUN_N}.html"

echo -e "\n%%% Running Regression: ${REF_DIR} (ref) vs ${NEW_DIR} (new) %%%\n"
echo "  Run #       : ${RUN_N}"
echo "  Output JSON : ${COMPARISON_JSON}"
echo "  Output HTML : ${HTML_OUT}"
echo ""

python3 - <<EOF
import sys, json
from pathlib import Path

sys.path.insert(0, "${REGRESSION_SRC}")
from utils import set_path, to_pkl, from_pkl
from compare import compare_frames
from create_report import create_interactive_report

# --- Pickle reference (workup.iter / stable build) ---
print("* Loading reference: ${REF_DIR}")
_, ref_tsvs = set_path("${REF_DIR}")
import tempfile, pickle, zipfile
from datetime import datetime

def build_pkl(tsvs, label):
    import pandas as pd
    dataframes = {}
    for f in tsvs:
        p = Path(f)
        if not p.exists():
            print(f"  [SKIP] {p.name} not found")
            continue
        df = pd.read_csv(p, sep="\t", low_memory=False)
        dataframes[p.name] = df
        print(f"  Loaded {p.name}, shape={df.shape}")
    dataframes["__metadata__"] = {
        "created_at": datetime.now().isoformat(),
        "table_count": len(dataframes) - 1,
        "label": label
    }
    return dataframes

ref_data = build_pkl(ref_tsvs, "${REF_DIR}")

print("\n* Loading new:       ${NEW_DIR}")
_, new_tsvs = set_path("${NEW_DIR}")
new_data = build_pkl(new_tsvs, "${NEW_DIR}")

# --- Compare ---
print("\n* Comparing tables...")
out = {}
all_tables = (set(new_data.keys()) | set(ref_data.keys())) - {"__metadata__"}

for table in sorted(all_tables):
    if table not in new_data:
        out[table] = f"skipped, missing in new (${NEW_DIR})"
    elif table not in ref_data:
        out[table] = f"skipped, missing in ref (${REF_DIR})"
    else:
        out[table] = compare_frames(ref_data[table], new_data[table])

# --- Save JSON ---
comparison_json = Path("${COMPARISON_JSON}")
with open(comparison_json, "w") as f:
    json.dump(out, f, indent=4)
print(f"\n* Comparison JSON saved: {comparison_json}")

# --- Generate HTML report ---
print("* Generating HTML report...")
create_interactive_report(
    report_path=comparison_json,
    output_dir=Path("${OUTPUT_DIR}")
)

# Rename html/final_output.html -> final_output.<N>.html
src_html = Path("${OUTPUT_DIR}") / "final_output.html"
dst_html = Path("${HTML_OUT}")
if src_html.exists() and src_html != dst_html:
    src_html.rename(dst_html)
    print(f"* HTML report saved:     {dst_html}")
elif dst_html.exists():
    print(f"* HTML report saved:     {dst_html}")

print("\n%%% Regression complete. %%%")
EOF
