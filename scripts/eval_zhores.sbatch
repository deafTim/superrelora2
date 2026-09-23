#!/bin/bash
# Eval LoRA / ReLoRA / SuperReLoRa final_model on Zhores (SLURM).
# Submit from repo root (home clone):
#   bash scripts/submit_zhores_eval.sh
#   # or point at a run:
#   RUN_ID=20260923_j8472336 sbatch --export=ALL,RUN_ID scripts/eval_zhores.sbatch
#
# Reads:  .../runs/<RUN_ID>/{method}/final_model
# Writes: .../runs/<RUN_ID>/{method}/metrics.json
# RUN_ID defaults to contents of .../runs/LATEST

#SBATCH -J eval_srl
#SBATCH -p gpu_devel
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH -o logs/eval_%A_%a.out
#SBATCH -e logs/eval_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"
mkdir -p logs
export PYTHONPATH="${REPO_ROOT}"

PROJECT_ROOT="/gpfs/gpfs0/timofey.glukhikh/SuperReLoRa"
CONDA_ENV="/home/timofey.glukhikh/.conda/science_phan"

export HF_HOME="${PROJECT_ROOT}/hf_cache"
export SUPERRELORA_HF_HOME="${HF_HOME}"
mkdir -p "${HF_HOME}"

source /trinity/shared/opt/mambaforge3/etc/profile.d/conda.sh
module unload python/3.12 || true
set +u
conda activate "${CONDA_ENV}"
set -u

if [[ -z "${RUN_ID:-}" ]]; then
  if [[ -f "${PROJECT_ROOT}/runs/LATEST" ]]; then
    RUN_ID="$(tr -d '[:space:]' < "${PROJECT_ROOT}/runs/LATEST")"
  else
    echo "ERROR: RUN_ID not set and ${PROJECT_ROOT}/runs/LATEST missing"
    exit 1
  fi
fi

METHODS=(lora relora superrelora)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
RUN_DIR="${PROJECT_ROOT}/runs/${RUN_ID}/${METHOD}"
# Legacy flat layout fallback (pre-timestamp runs)
if [[ ! -d "${RUN_DIR}" ]] && [[ -d "${PROJECT_ROOT}/runs/${METHOD}" ]]; then
  RUN_DIR="${PROJECT_ROOT}/runs/${METHOD}"
fi
FINAL_DIR="${RUN_DIR}/final_model"
METRICS_OUT="${RUN_DIR}/metrics.json"

# Idempotent only if FORCE is unset; FORCE=1 to overwrite.
if [[ -z "${FORCE:-}" ]] && [[ -f "${METRICS_OUT}" ]]; then
  echo "Skip ${METHOD}: ${METRICS_OUT} exists (FORCE=1 to overwrite)"
  exit 0
fi

if [[ ! -d "${FINAL_DIR}" ]] && [[ ! -f "${RUN_DIR}/final_model.pt" ]]; then
  echo "ERROR: no final model for ${METHOD} under ${RUN_DIR}"
  exit 1
fi

if [[ -d "${FINAL_DIR}" ]]; then
  WEIGHT="${FINAL_DIR}"
elif [[ -f "${RUN_DIR}/final_model.pt" ]]; then
  WEIGHT="${RUN_DIR}/final_model.pt"
else
  echo "ERROR: cannot resolve weights for ${METHOD}"
  exit 1
fi

echo "RUN_ID=${RUN_ID}"
echo "method=${METHOD}"
echo "WEIGHT=${WEIGHT}"
echo "METRICS_OUT=${METRICS_OUT}"
echo "HF_HOME=${HF_HOME}"
nvidia-smi || true

python scripts/eval_superrelora.py \
  --model_path "${WEIGHT}" \
  --method "${METHOD}" \
  --metrics_out "${METRICS_OUT}" \
  --dataset_name Salesforce/wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --batch_size 8 \
  --max_length 128 \
  --num_samples 1000

echo "Finished eval ${METHOD} (RUN_ID=${RUN_ID})"
