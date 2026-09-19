#!/bin/bash

# === Quick SuperReLoRA debug run ===

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=.

mkdir -p debug_results

python scripts/train_superrelora.py \
  --config training_configs/superrelora_160m.yaml \
  --method superrelora \
  --output_dir debug_results \
  --use_trainer \
  --merge_every 2 \
  --max_steps 5 \
  --batch_size 2 \
  --num_epochs 1 \
  --logging_steps 1 \
  --eval_steps 5 \
  --limit_train_examples 10
