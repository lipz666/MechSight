#!/usr/bin/env bash
# Free-rollout rerun, 4 GPUs: one FlowER and one ChRIMP shard per GPU; per-case JSON output, resumable.
#   Example cluster submission (adjust for your scheduler): 4 GPUs, 32 CPUs, 96 GiB RAM
set -uo pipefail
cd ~/mechspec/run
export HF_HOME=~/mechspec/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PY=~/mechspec/.venv/bin/python
OUT=~/mechspec/out/rollout; mkdir -p $OUT/flower $OUT/chrimp $OUT/logs
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
N=${#G[@]}
echo "start $(date -u) gpus=${CUDA_VISIBLE_DEVICES:-none} n=$N"
pids=()
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=${G[$i]} $PY spec_rollout.py flower ~/mechspec/data/spec_inputs.json $OUT/flower $i/$N >> $OUT/logs/flower_$i.log 2>&1 & pids+=($!)
  CUDA_VISIBLE_DEVICES=${G[$i]} $PY spec_rollout.py chrimp ~/mechspec/data/spec_inputs.json $OUT/chrimp $i/$N >> $OUT/logs/chrimp_$i.log 2>&1 & pids+=($!)
done
wait "${pids[@]}"
echo "ROLLOUT_JOB_DONE $(date -u)"
