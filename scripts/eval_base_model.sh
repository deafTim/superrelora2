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

# Run base model evaluation
python scripts/eval_base_model.py 