#!/usr/bin/env bash
set -euo pipefail

: "${MECH_API_KEY:?Set MECH_API_KEY without writing it to this repository}"

MECH_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MECH_REPO_ROOT="$(cd -- "${MECH_SCRIPT_DIR}/.." && pwd)"
MECH_SKILLRL_ROOT="${MECH_SKILLRL_ROOT:-${MECH_REPO_ROOT}}"
MECH_PYTHON_BIN="${MECH_PYTHON_BIN:-python}"
MECH_API_BASE_URL="${MECH_API_BASE_URL:-http://127.0.0.1:8317/v1}"
MECH_CLOSED_MODEL="${MECH_CLOSED_MODEL:-gemini-3.7-flash-high}"
MECH_EXPERIMENT_STAGE="${MECH_EXPERIMENT_STAGE:-smoke}"
MECH_CORE_DIR="${MECH_CORE_DIR:-${MECH_REPO_ROOT}/../core}"

if [[ "${MECH_CORE_DIR}" = /* ]]; then
  MECH_RESOLVED_CORE_DIR="${MECH_CORE_DIR}"
else
  MECH_RESOLVED_CORE_DIR="${MECH_SKILLRL_ROOT}/${MECH_CORE_DIR}"
fi
if [[ ! -d "${MECH_RESOLVED_CORE_DIR}" ]]; then
  echo "MechFinder core data directory is missing: ${MECH_RESOLVED_CORE_DIR}" >&2
  exit 2
fi

case "${MECH_EXPERIMENT_STAGE}" in
  smoke)
    MECH_NUM_CLASSES="${MECH_NUM_CLASSES:-5}"
    MECH_NUM_RUNS="${MECH_NUM_RUNS:-5}"
    MECH_REPEATS_PER_TASK="${MECH_REPEATS_PER_TASK:-1}"
    ;;
  full)
    MECH_NUM_CLASSES="${MECH_NUM_CLASSES:-60}"
    MECH_NUM_RUNS="${MECH_NUM_RUNS:-60}"
    MECH_REPEATS_PER_TASK="${MECH_REPEATS_PER_TASK:-3}"
    ;;
  *)
    echo "MECH_EXPERIMENT_STAGE must be smoke or full" >&2
    exit 2
    ;;
esac

MECH_ANALYSIS_OUTPUT="${MECH_ANALYSIS_OUTPUT:-outputs/gemini37_experiment_d/${MECH_EXPERIMENT_STAGE}.jsonl}"
mkdir -p "$(dirname "${MECH_ANALYSIS_OUTPUT}")"

export NO_PROXY="localhost,127.0.0.1,${NO_PROXY:-}"
export no_proxy="localhost,127.0.0.1,${no_proxy:-}"
export PLANNER_API_KEY="${MECH_API_KEY}"
export EXECUTOR_API_KEY="${MECH_API_KEY}"
export GT_EVALUATOR_API_KEY="${MECH_API_KEY}"

exec "${MECH_PYTHON_BIN}" -m develop.run_agent \
  --skillrl-root "${MECH_SKILLRL_ROOT}" \
  --data-source core \
  --split all \
  --core-dir "${MECH_CORE_DIR}" \
  --core-subset mechfinder \
  --mechfinder-num-classes "${MECH_NUM_CLASSES}" \
  --mechfinder-per-class 1 \
  --max-tasks "${MECH_NUM_RUNS}" \
  --num-runs "${MECH_NUM_RUNS}" \
  --repeats-per-task "${MECH_REPEATS_PER_TASK}" \
  --task-input-mode structures_only \
  --planner-base-url "${MECH_API_BASE_URL}" \
  --executor-base-url "${MECH_API_BASE_URL}" \
  --gt-evaluator-base-url "${MECH_API_BASE_URL}" \
  --planner-model "${MECH_CLOSED_MODEL}" \
  --executor-model "${MECH_CLOSED_MODEL}" \
  --gt-evaluator-model "${MECH_CLOSED_MODEL}" \
  --planner-temperature 0.2 \
  --executor-temperature 0.2 \
  --gt-evaluator-temperature 0.0 \
  --planner-max-tokens 4096 \
  --executor-max-tokens 4096 \
  --gt-evaluator-max-tokens 4096 \
  --max-planner-review-rounds 3 \
  --omit-thinking-extra-body \
  --analysis-output "${MECH_ANALYSIS_OUTPUT}" \
  "$@"
