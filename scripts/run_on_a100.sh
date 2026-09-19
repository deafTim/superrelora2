#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=.

mkdir -p results/superrelora results/relora

# SuperReLoRa (orthogonal reinit)
python scripts/train_superrelora.py \
    --config training_configs/superrelora_160m.yaml \
    --method superrelora \
    --output_dir results/superrelora \
    --use_trainer \
    --merge_every 500 \
    --max_steps 8000 \
    --batch_size 8 \
    --num_epochs 3 \
    --logging_steps 100 \
    --eval_steps 1000

# ReLoRA baseline (merge + reinit, no orthogonality)
python scripts/train_superrelora.py \
    --config training_configs/relora_160m.yaml \
    --method relora \
    --output_dir results/relora \
    --use_trainer \
    --merge_every 500 \
    --max_steps 8000 \
    --batch_size 8 \
    --num_epochs 3 \
    --logging_steps 100 \
    --eval_steps 1000
