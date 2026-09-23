#!/bin/bash
# Submit LoRA / ReLoRA / SuperReLoRa array job on Zhores.
# Run from repo root on the cluster:
#   bash scripts/submit_zhores.sh
#
# Creates a new timestamped run folder:
#   /gpfs/.../SuperReLoRa/runs/<RUN_ID>/{lora,relora,superrelora}/

set -euo pipefail
mkdir -p logs

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export RUN_ID
JOB_ID="$(sbatch --export=ALL,RUN_ID --parsable scripts/train_zhores.sbatch)"
echo "Submitted train array job ${JOB_ID}"
echo "RUN_ID=${RUN_ID}"
echo "Outputs: /gpfs/gpfs0/timofey.glukhikh/SuperReLoRa/runs/${RUN_ID}/"
echo "Check: squeue -u \$USER  and  logs/train_${JOB_ID}_*.out"
