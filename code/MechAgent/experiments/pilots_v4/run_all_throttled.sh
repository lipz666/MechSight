#!/bin/zsh
# Relaunch of the reviewer reruns with a total concurrency the gateways sustain (~300 on 117, ~200 on vps).
# Everything is cached per call, so finished work is skipped.
cd "$(dirname "$0")"
: "${OPENAI_API_KEY:?}" "${FALLBACK_API_KEY:?}"
PY=../../../external/venv-llm/bin/python
O=../../outputs/pilots_v4_20261103
f() { grep --line-buffered -v -i warning; }

CAP117=${S117:-25} CAPVPS=${SVPS:-0} ./run_scale.sh &
CAP117=${L117:-25} CAPVPS=${LVPS:-0} ./run_lit150.sh &
( export FAILOVER_PREFER=117 FAILOVER_CAP_117=${X117:-10} FAILOVER_CAP_VPS=${XVPS:-0}
  for t in blind randsmi curve; do $PY p22_extras.py --task $t --workers 80 > /dev/null; echo "$t done $(date)"; done
  # USPTO rerun after the extras (fills the calls that failed under overload)
  export FAILOVER_CAP_117=${X117:-10} FAILOVER_CAP_VPS=${XVPS:-0}
  MAP=../../../external/env-map/bin/python
  $PY p16_supplement_g38.py --task uspto_pool --workers 90
  $PY p16_supplement_g38.py --task uspto_requests && $MAP stepmap.py
  $PY p16_supplement_g38.py --task uspto_r1 --workers 90
  $PY p16_supplement_g38.py --task uspto_requests && $MAP stepmap.py
  $PY p16_supplement_g38.py --task uspto_r2 --workers 90
  $PY p16_supplement_g38.py --task uspto_analyze > /dev/null
  echo "USPTO rerun done $(date)" ) 2>&1 | f > $O/console_p22_uspto.log &
wait
echo "all done $(date)"
