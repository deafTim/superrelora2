#!/bin/bash
# Eval LoRA / ReLoRA / SuperReLoRa final_model on Zhores (SLURM).
# Submit from repo root (home clone):
#   mkdir -p logs
#   sbatch scripts/eval_zhores.sbatch
#
# Code:  $SLURM_SUBMIT_DIR  (home)
# Runs:  /gpfs/gpfs0/timofey.glukhikh/SuperReLoRa/runs/{method}/final_model
# Writes: .../runs/{method}/metrics.json

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

METHODS=(lora relora superrelora)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
RUN_DIR="${PROJECT_ROOT}/runs/${METHOD}"
FINAL_DIR="${RUN_DIR}/final_model"
METRICS_OUT="${RUN_DIR}/metrics.json"

if [[ -f "${METRICS_OUT}" ]]; then
  echo "Skip ${METHOD}: ${METRICS_OUT} exists"
  exit 0
fi

if [[ ! -d "${FINAL_DIR}" ]] && [[ ! -f "${RUN_DIR}/final_model.pt" ]]; then
  echo "ERROR: no final model for ${METHOD} under ${RUN_DIR}"
  exit 1
fi

# Prefer Trainer dir; fall back to .pt from manual train
if [[ -d "${FINAL_DIR}" ]]; then
  WEIGHT="${FINAL_DIR}"
elif [[ -f "${RUN_DIR}/final_model.pt" ]]; then
  WEIGHT="${RUN_DIR}/final_model.pt"
else
  echo "ERROR: cannot resolve weights for ${METHOD}"
  exit 1
fi

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

echo "Finished eval ${METHOD}"
