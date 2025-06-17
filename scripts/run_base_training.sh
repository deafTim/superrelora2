#!/bin/bash

# === Base model training on A100 (no LoRA / no merge) ===

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Папка для логов и результатов base-модели
mkdir -p results_base
: > results_base/train.log  # очистить старый лог

# Запуск с отключением LoRA merge
stdbuf -oL -eL python3 -u scripts/train_superrelora.py \
    --config training_configs/superrelora_160m.yaml \
    --output_dir results_base \
    --use_trainer \
    --max_steps 8000 \
    --batch_size 64 \
    --num_epochs 3 \
    --logging_steps 20 \
    --eval_steps 1000 \
    2>&1 | tee -a results_base/train.log
