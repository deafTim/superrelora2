#!/bin/bash

# # Run base model evaluation
# echo "Starting base model evaluation..."
# nohup bash scripts/eval_base_model.sh > nohup_base_eval.out 2>&1 &
# echo "Base model evaluation started in background. Logs: nohup_base_eval.out"

# Run SuperReLoRA evaluation
echo "Starting SuperReLoRA evaluation..."
nohup bash scripts/eval_superrelora.sh > nohup_superrelora_eval.out 2>&1 &
echo "SuperReLoRA evaluation started in background. Logs: nohup_superrelora_eval.out"
