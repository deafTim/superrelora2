#!/bin/bash
# Submit LoRA / ReLoRA / SuperReLoRa array job on Zhores.
# Run from repo root on the cluster:
#   bash scripts/submit_zhores.sh

set -euo pipefail
mkdir -p logs
sbatch scripts/train_zhores.sbatch
echo "Submitted. Check: squeue -u \$USER  and  logs/train_*.out"
