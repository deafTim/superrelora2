#!/bin/bash

# === SuperReLoRA A100 run script ===

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Результаты
mkdir -p results
: > results/train.log  # очистим старый лог

# Используем stdbuf + python -u, чтобы отключить буферизацию
stdbuf -oL -eL python3 -u scripts/train_superrelora.py \
    --config training_configs/superrelora_160m.yaml \
    --output_dir results \
    --use_trainer \
    --merge_every 500 \
    --merge_alpha 0.1 \
    --max_steps 8000 \
    --batch_size 64 \
    --num_epochs 3 \
    --logging_steps 20 \
    --eval_steps 1000 \
    2>&1 | tee -a results/train.log
