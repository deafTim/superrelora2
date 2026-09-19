#!/bin/bash

# === SuperReLoRA vs ReLoRA A100 run (DataSphere) ===

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=.

mkdir -p results/superrelora results/relora

stdbuf -oL -eL python3 -u scripts/train_superrelora.py \
    --config training_configs/superrelora_160m.yaml \
    --method superrelora \
    --output_dir results/superrelora \
    --use_trainer \
    --merge_every 500 \
    --max_steps 8000 \
    --batch_size 64 \
    --num_epochs 3 \
    --logging_steps 20 \
    --eval_steps 1000 \
    2>&1 | tee -a results/superrelora/train.log

stdbuf -oL -eL python3 -u scripts/train_superrelora.py \
    --config training_configs/relora_160m.yaml \
    --method relora \
    --output_dir results/relora \
    --use_trainer \
    --merge_every 500 \
    --max_steps 8000 \
    --batch_size 64 \
    --num_epochs 3 \
    --logging_steps 20 \
    --eval_steps 1000 \
    2>&1 | tee -a results/relora/train.log
