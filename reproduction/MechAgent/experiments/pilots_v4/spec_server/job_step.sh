#!/usr/bin/env bash
# Single-step (reference previous state) predictions; resumable.
#   Example cluster submission (adjust for your scheduler): 1 GPU, 8 CPUs, 32 GiB RAM
set -uo pipefail
cd ~/mechspec/run
export HF_HOME=~/mechspec/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PY=~/mechspec/.venv/bin/python
mkdir -p ~/mechspec/out
echo "start $(date -u) cuda=${CUDA_VISIBLE_DEVICES:-none}"
$PY spec_step.py flower ~/mechspec/data/steps.json ~/mechspec/out/preds_flower.jsonl 0/1
$PY spec_step.py chrimp ~/mechspec/data/steps.json ~/mechspec/out/preds_chrimp.jsonl 0/1
echo "STEP_JOB_DONE $(date -u)"
