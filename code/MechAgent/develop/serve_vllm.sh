#!/usr/bin/env bash
set -euo pipefail

MECH_MODEL_PATH="${MECH_MODEL_PATH:-/path/to/model/gemma-4-31B-it}"
MECH_SERVED_MODEL="${MECH_SERVED_MODEL:-gemma4-31b}"
MECH_CHAT_TEMPLATE="${MECH_CHAT_TEMPLATE:-${MECH_MODEL_PATH}/chat_template.jinja}"
MECH_SERVER_HOST="${MECH_SERVER_HOST:-127.0.0.1}"
MECH_SERVER_PORT="${MECH_SERVER_PORT:-8000}"
MECH_TENSOR_PARALLEL_SIZE="${MECH_TENSOR_PARALLEL_SIZE:-4}"
MECH_GPU_MEMORY_UTILIZATION="${MECH_GPU_MEMORY_UTILIZATION:-0.90}"
MECH_MAX_MODEL_LEN="${MECH_MAX_MODEL_LEN:-30000}"
MECH_MAX_NUM_BATCHED_TOKENS="${MECH_MAX_NUM_BATCHED_TOKENS:-8192}"

exec vllm serve "${MECH_MODEL_PATH}" \
  --served-model-name "${MECH_SERVED_MODEL}" \
  --chat-template "${MECH_CHAT_TEMPLATE}" \
  --host "${MECH_SERVER_HOST}" \
  --port "${MECH_SERVER_PORT}" \
  --tensor-parallel-size "${MECH_TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${MECH_GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MECH_MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MECH_MAX_NUM_BATCHED_TOKENS}" \
  --reasoning-parser gemma4 \
  "$@"
