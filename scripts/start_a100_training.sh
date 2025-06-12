#!/bin/bash

nohup bash scripts/run_on_a100_datasphere.sh > nohup_a100.out 2>&1 &
echo "Started run_on_a100.sh in background. Logs: nohup_a100.out" 