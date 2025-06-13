#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Run base model evaluation with quick settings
python3 scripts/eval_base_model.py \
    --dataset_name wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --batch_size 8 \
    --max_length 128 \
    --num_samples 1000
    
    