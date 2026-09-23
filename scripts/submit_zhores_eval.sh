#!/bin/bash
# Submit eval array for the latest (or given) RUN_ID.
#   bash scripts/submit_zhores_eval.sh
#   RUN_ID=20260923_171500 bash scripts/submit_zhores_eval.sh
#   FORCE=1 bash scripts/submit_zhores_eval.sh

set -euo pipefail
mkdir -p logs

PROJECT_ROOT="/gpfs/gpfs0/timofey.glukhikh/SuperReLoRa"
if [[ -z "${RUN_ID:-}" ]]; then
  if [[ -f "${PROJECT_ROOT}/runs/LATEST" ]]; then
    RUN_ID="$(tr -d '[:space:]' < "${PROJECT_ROOT}/runs/LATEST")"
  else
    echo "ERROR: set RUN_ID or create ${PROJECT_ROOT}/runs/LATEST"
    exit 1
  fi
fi
export RUN_ID

EXPORT_LIST="ALL,RUN_ID"
if [[ -n "${FORCE:-}" ]]; then
  export FORCE
  EXPORT_LIST+=",FORCE"
fi

JOB_ID="$(sbatch --export="${EXPORT_LIST}" --parsable scripts/eval_zhores.sbatch)"
echo "Submitted eval array job ${JOB_ID}"
echo "RUN_ID=${RUN_ID}"
echo "Check: logs/eval_${JOB_ID}_*.out"
