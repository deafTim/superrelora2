#!/bin/bash
# SuperReLoRa / ReLoRA / LoRA training on Zhores (SLURM).
# Submit from repo root (home clone):
#   bash scripts/submit_zhores.sh
#   # or:
#   RUN_ID=$(date +%Y%m%d_%H%M%S) sbatch --export=ALL,RUN_ID scripts/train_zhores.sbatch
#
# Outputs:
#   /gpfs/.../SuperReLoRa/runs/<RUN_ID>/{lora,relora,superrelora}/
#   /gpfs/.../SuperReLoRa/runs/LATEST  -> text file with RUN_ID
#
# Code:  $SLURM_SUBMIT_DIR  (home)
# Data:  /gpfs/gpfs0/timofey.glukhikh/SuperReLoRa/{hf_cache,runs}

#SBATCH -J superrelora
#SBATCH -p gpu_devel
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH -o logs/train_%A_%a.out
#SBATCH -e logs/train_%A_%a.err
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
mkdir -p "${HF_HOME}" "${PROJECT_ROOT}/runs" "${PROJECT_ROOT}/figures"

source /trinity/shared/opt/mambaforge3/etc/profile.d/conda.sh
module unload python/3.12 || true
set +u
conda activate "${CONDA_ENV}"
set -u

# Shared stamp for the whole array: prefer explicit RUN_ID, else job array id.
if [[ -z "${RUN_ID:-}" ]]; then
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
    RUN_ID="$(date +%Y%m%d)_j${SLURM_ARRAY_JOB_ID}"
  else
    RUN_ID="$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID}"
  fi
fi
# Persist for eval / notebook (same for every array task).
echo "${RUN_ID}" > "${PROJECT_ROOT}/runs/LATEST"

METHODS=(lora relora superrelora)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
OUT_DIR="${PROJECT_ROOT}/runs/${RUN_ID}/${METHOD}"
CONFIG="${REPO_ROOT}/training_configs/${METHOD}_160m.yaml"

# Idempotent resubmit within this RUN_ID only
if [[ -f "${OUT_DIR}/DONE" ]]; then
  echo "Skip ${METHOD}: ${OUT_DIR}/DONE exists"
  exit 0
fi
if [[ -f "${OUT_DIR}/final_model.pt" ]] || [[ -d "${OUT_DIR}/final_model" ]]; then
  echo "Skip ${METHOD}: final model already present"
  touch "${OUT_DIR}/DONE"
  exit 0
fi

mkdir -p "${OUT_DIR}"
echo "RUN_ID=${RUN_ID}"
echo "method=${METHOD}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "HF_HOME=${HF_HOME}"
echo "OUT_DIR=${OUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

python scripts/train_superrelora.py \
  --config "${CONFIG}" \
  --method "${METHOD}" \
  --output_dir "${OUT_DIR}" \
  --use_trainer \
  --merge_every 500 \
  --max_steps 8000 \
  --batch_size 8 \
  --num_epochs 3 \
  --logging_steps 100 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --resume_from_checkpoint auto

touch "${OUT_DIR}/DONE"
echo "Finished ${METHOD} (RUN_ID=${RUN_ID})"
