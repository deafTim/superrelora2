#!/bin/bash

# === Быстрый 1-минутный запуск SuperReLoRA на Mac ===

export TOKENIZERS_PARALLELISM=false

# Создаём папку для результатов
mkdir -p debug_results

# Запуск с переопределёнными параметрами
python scripts/train_superrelora.py \
  --config training_configs/superrelora_160m.yaml \
  --output_dir debug_results \
  --use_trainer \
  --merge_every 2 \
  --merge_alpha 0.1 \
  --max_steps 5 \
  --batch_size 2 \
  --num_epochs 1 \
  --logging_steps 1 \
  --eval_steps 5 \
  --limit_train_examples 10
