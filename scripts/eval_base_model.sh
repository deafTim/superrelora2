#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Run base model evaluation
python scripts/eval_base_model.py 