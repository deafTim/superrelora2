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
  --merge_every 20 \
  --merge_alpha 0.1 \
  --max_steps 30 \
  --logging_steps 5 \
  --eval_steps 30
