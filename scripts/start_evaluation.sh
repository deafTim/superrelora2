#!/bin/bash

# # Run base model evaluation
# echo "Starting base model evaluation..."
# nohup bash scripts/eval_full.sh > nohup_full_eval.out 2>&1 &
# echo "Full-trained model evaluation started in background. Logs: nohup_full_eval.out"

# # Run base model evaluation
# echo "Starting base model evaluation..."
# nohup bash scripts/eval_base_model.sh > nohup_base_eval.out 2>&1 &
# echo "Base model evaluation started in background. Logs: nohup_base_eval.out"

# # Run SuperReLoRA evaluation
# echo "Starting SuperReLoRA evaluation..."
# nohup bash scripts/eval_superrelora.sh > nohup_superrelora_eval.out 2>&1 &
# echo "SuperReLoRA evaluation started in background. Logs: nohup_superrelora_eval.out"


#!/bin/bash

# === Universal Evaluation Runner ===

# Create logs directory if it doesn't exist
mkdir -p logs

# === Target script to run ===
TARGET_SCRIPT="scripts/eval_superrelora.sh"

# Extract base name without extension
SCRIPT_NAME=$(basename "$TARGET_SCRIPT" .sh)

# Get timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Log file path
LOG_FILE="logs/nohup_${SCRIPT_NAME}_${TIMESTAMP}.out"

# Run the target script with nohup and save output
echo "Starting $SCRIPT_NAME in background..."
nohup bash "$TARGET_SCRIPT" > "$LOG_FILE" 2>&1 &
echo "$SCRIPT_NAME started. Logs: $LOG_FILE"
