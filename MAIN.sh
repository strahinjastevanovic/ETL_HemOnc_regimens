#!/bin/bash
set -e

# === MAIN.sh ===
# Usage: ./MAIN.sh -out <workdir>

# --- Resolve script location to allow relative paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    -out)
      WORKDIR="$2"
      shift 2
      ;;
    *)
      echo "%%ERR%% Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ -z "$WORKDIR" ]; then
  echo "%%ERR%% Missing -out <workdir> argument"
  exit 1
fi

mkdir -p "$WORKDIR"

# --- Paths ---
FILES_ROOT="INPUT_FILES_HEMONC"
REF_DIR="OTHER_REF"
REF_RGROUPS="${REF_DIR}/rgroups_template.tsv"
REF_VALIDDRUGS="${REF_DIR}/validdrugs_template.tsv"
SUPP_FILE="${REF_DIR}/supplementary_therapy.json"
REGIMEN_TSV="${WORKDIR}/regimens.tsv"
REGIMEN_TSV_FULL="${WORKDIR}/regimens_full.tsv"
NLTK_DATA_DIR="nltk_data"
LOGS="${WORKDIR}/logs"

echo -e "\n%%% Starting Configuration and Pre-process... %%%\n"
python3 - <<EOF
import sys
sys.path.insert(0, "${SRC_DIR}")
from pre_run import pre_run
if __name__ == "__main__":
    pre_run("${FILES_ROOT}", "${WORKDIR}", "${NLTK_DATA_DIR}", "${LOGS}")
EOF

echo -e "\n%%% Running ETL... %%%\n"
python3 - <<EOF
import sys
sys.path.insert(0, "${SRC_DIR}")
from etl_main import ETL  
if __name__ == "__main__":
    etl = ETL()
    etl.run("${REGIMEN_TSV}", "${SUPP_FILE}")
EOF

echo -e "\n%%% Generating updated regimen groups and valid drugs... %%%\n"
python3 - <<EOF
import sys
sys.path.insert(0, "${SRC_DIR}")
from other_ref import generate_reg_group, generate_valid_drugs 
if __name__ == "__main__":
    generate_reg_group("${REGIMEN_TSV_FULL}", "${REF_RGROUPS}", workdir="${WORKDIR}")
    generate_valid_drugs("${REGIMEN_TSV_FULL}", "${REF_VALIDDRUGS}", workdir="${WORKDIR}")
EOF

echo -e "\n%%% Converting TSVs to RDA... %%%\n"
Rscript - <<EOF
regimens <- read.delim("${WORKDIR}/regimens.tsv", stringsAsFactors = FALSE)
save(regimens, file = "${WORKDIR}/regimens.rda")

validdrugs <- read.delim("${WORKDIR}/validdrugs.tsv", stringsAsFactors = FALSE)
save(validdrugs, file = "${WORKDIR}/validdrugs.rda")

regimengroups <- read.delim("${WORKDIR}/regimengroups.tsv", stringsAsFactors = FALSE)
save(regimengroups, file = "${WORKDIR}/regimengroups.rda")
EOF

echo -e "\n%%% Writing Validation report... %%%\n"
python3 "${SRC_DIR}/validation_check.py" "${WORKDIR}" "${REGIMEN_TSV_FULL}"

echo -e "\n%%% Done. Outputs saved in: $WORKDIR %%%\n"
