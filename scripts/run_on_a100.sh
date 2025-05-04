#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Create results directory
mkdir -p debug_results

# Run long training with SuperReLoRA parameters (full dataset)
python scripts/train_superrelora.py \
    --config training_configs/superrelora_160m.yaml \
    --output_dir results \
    --use_trainer \
    --merge_every 500 \
    --merge_alpha 0.1 \
    --max_steps 8000 \
    --batch_size 8 \
    --num_epochs 3 \
    --logging_steps 100 \
    --eval_steps 1000

# Optional: Run with manual training loop
# python scripts/train_superrelora.py \
#     --config training_configs/superrelora_160m.yaml \
#     --output_dir results 