#!/bin/zsh
# 117-only queue (user decision 2026-11-04): after the literature/blind/randsmi run, the scale-up
# confirmation, then the compute curve and the USPTO in-template run. Low concurrency throughout.
cd "$(dirname "$0")"
: "${FALLBACK_API_KEY:?}"
export OPENAI_API_KEY=${OPENAI_API_KEY:-unused}
PY=../../../external/venv-llm/bin/python
MAP=../../../external/env-map/bin/python
O=../../outputs/pilots_v4_20261103
while pgrep -f run_priority.sh > /dev/null; do sleep 60; done
echo "queue start $(date)" >> $O/console_queue.log
CAP117=${S117:-160} CAPVPS=0 ./run_scale.sh
( export FAILOVER_PREFER=117 FAILOVER_CAP_117=${X117:-160} FAILOVER_CAP_VPS=0
  $PY p22_extras.py --task curve --workers 160 > /dev/null; echo "curve done $(date)"
  $PY p16_supplement_g38.py --task uspto_pool --workers 160
  $PY p16_supplement_g38.py --task uspto_requests && $MAP stepmap.py
  $PY p16_supplement_g38.py --task uspto_r1 --workers 160
  $PY p16_supplement_g38.py --task uspto_requests && $MAP stepmap.py
  $PY p16_supplement_g38.py --task uspto_r2 --workers 160
  $PY p16_supplement_g38.py --task uspto_analyze > /dev/null
  $PY p23_offline.py > /dev/null
  echo "queue done $(date)" ) 2>&1 | grep --line-buffered -v -i warning >> $O/console_queue.log
