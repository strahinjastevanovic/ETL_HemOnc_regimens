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

# --- Run Definition ---
VERSION="2025-03-15"
DESCRIPTION="$VERSION\nSource:\nJeremy Warner MD, MS, 2021, \nHemOnc knowledgebase, \nhttps://doi.org/10.7910/DVN/FPO4HB, Harvard Dataverse, V45, UNF:6:hDCRZx6AGBRIU68MJeG21Q== [fileUNF]"
SIGS_FILE="sigs_march_2025.csv"
FILES_ROOT="INPUT_FILES_HEMONC"
REF_DIR="OTHER_REF"
REF_RGROUPS="${REF_DIR}/rgroups_template.tsv"
REF_VALIDDRUGS="${REF_DIR}/validdrugs_template.tsv"
REGIMEN_TSV="${WORKDIR}/regimens.tsv"
REGIMEN_TSV_FULL="${WORKDIR}/regimens_full.tsv"
LOGS="${WORKDIR}/logs"
SHEET_CONFIG="${REF_DIR}/sheets_config.json"

echo -e "\n%%% Starting ETL... %%%\n"
echo -e "%%%\n\nHemOnc Version:\n ${DESCRIPTION} \n\n%%%"
echo -e "\n%%% Running Queries... %%%\n"

# Load .env (silently; file may not exist in CI)
if [ -f "${SCRIPT_DIR}/.env" ]; then
  set -o allexport
  source "${SCRIPT_DIR}/.env"
  set +o allexport
fi

DB_FRESH="${DB_FRESH:-FALSE}"

if [ "$DB_FRESH" = "TRUE" ]; then
  echo "%%% [queries] DB_FRESH=TRUE — running live Athena queries %%%"
  python3 - <<EOF
import sys, os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join("${SCRIPT_DIR}", ".env"))
sys.path.insert(0, "${SRC_DIR}")
from query_vocab import main
if __name__ == "__main__":
    credentials = {
        "username": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
        "host":     os.environ["DB_HOST"],
        "db":       os.environ["DB_NAME"],
    }
    main(
        credentials,
        "${FILES_ROOT}/${SIGS_FILE}",
        "${WORKDIR}/condition_concepts.csv",
        "${WORKDIR}/drug_concepts.csv",
        "${WORKDIR}/sigs_w_conditions.csv",
        "${WORKDIR}/concepts.tsv",
        "${LOGS}",
    )
EOF
else
  echo "%%% [queries] DB_FRESH=FALSE — using Athena snapshot from athena_mirrors %%%"
  python3 - <<EOF
import sys
sys.path.insert(0, "${SRC_DIR}")
from tools.athena_snapshot import load_snapshot
if __name__ == "__main__":
    load_snapshot("${WORKDIR}")
EOF
fi

echo -e "\n%%% Pre-processing... %%%\n"
python3 - <<EOF
import sys
sys.path.insert(0, "${SRC_DIR}")
from preproc import preprocessing
if __name__ == "__main__":
    preprocessing("${WORKDIR}/sigs_w_conditions.csv", "${WORKDIR}", "${LOGS}", "${SHEET_CONFIG}")
EOF

echo -e "\n%%% Processing SIGs... %%%\n"
python3 - <<EOF
import sys
sys.path.insert(0, "${SRC_DIR}")
from transform import Transform  
if __name__ == "__main__":
    transform = Transform()
    transform.run("${WORKDIR}/s_frame.parquet", "${REGIMEN_TSV}", "${LOGS}")
EOF

echo -e "\n%%% Generating updated regimen groups and valid drugs... %%%\n"
python3 - <<EOF
import sys
sys.path.insert(0, "${SRC_DIR}")
from data_model import generate_reg_group, generate_valid_drugs, generate_route_table, generate_shortString_table
if __name__ == "__main__":
    generate_reg_group("${REGIMEN_TSV_FULL}", "${REF_RGROUPS}", workdir="${WORKDIR}")
    generate_valid_drugs("${REGIMEN_TSV_FULL}", "${WORKDIR}/drug_concepts.csv", workdir="${WORKDIR}")
    generate_route_table("${REGIMEN_TSV_FULL}", workdir="${WORKDIR}")
    generate_shortString_table("${REGIMEN_TSV_FULL}", workdir="${WORKDIR}")
EOF

echo -e "\n%%% Converting TSVs to RDA... %%%\n"
Rscript "${SRC_DIR}/export_artifacts.R" "${WORKDIR}"

echo -e "\n%%% Running Validation... %%%\n"
python3 "${SRC_DIR}/validate.py" "${WORKDIR}" "${REGIMEN_TSV_FULL}"

echo -e "\n%%% Done. Outputs saved in: $WORKDIR %%%\n"
