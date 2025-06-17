#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=.

# Run evaluation
python3 scripts/eval_superrelora.py \
    --model_path results_full/final_model/pytorch_model.bin \
    --dataset_name wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --batch_size 8 \
    --max_length 128 \
    --num_samples 1000
    
