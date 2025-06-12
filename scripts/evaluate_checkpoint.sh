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

# Run evaluation
python scripts/evaluate_model.py \
    --model_path results/final_model.pt \
    --dataset_name wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --batch_size 8 \
    --max_length 128 \
    --num_samples 1000 