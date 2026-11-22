#!/bin/zsh
# Pre-registered confirmation (REGISTRATION_v5_confirm.json) on T3_confirm + oMe-Gold.
# Run after p13_confirm_pool.py. Keys only from the environment.
set -e
cd "$(dirname "$0")"
: "${OPENAI_API_KEY:?export OPENAI_API_KEY first}" "${FALLBACK_API_KEY:?export FALLBACK_API_KEY first}"
PY=../../../external/venv-llm/bin/python
MAP=../../../external/env-map/bin/python
S=T3_confirm,gold
{
  echo "start $(date)"
  $PY p11_mechagent.py --phase requests --sets $S
  $MAP stepmap.py
  FAILOVER_CAP_VPS=160 FAILOVER_CAP_117=160 $PY p11_mechagent.py --phase r1 --sets $S --workers 320 &
  FAILOVER_CAP_VPS=80 FAILOVER_CAP_117=80 $PY p11_mechagent.py --phase none --sets $S --workers 160 &
  FAILOVER_CAP_VPS=100 FAILOVER_CAP_117=100 $PY p12_controls.py --phase refine --sets $S --workers 200 &
  FAILOVER_CAP_VPS=60 FAILOVER_CAP_117=60 $PY p12_controls.py --phase oracle --sets T3_confirm --workers 120 &
  wait
  echo "round 1 + controls done $(date)"
  $PY p11_mechagent.py --phase requests --sets $S
  $MAP stepmap.py
  FAILOVER_CAP_VPS=400 FAILOVER_CAP_117=400 $PY p11_mechagent.py --phase r2 --sets $S --workers 800
  $PY p14_confirm.py > /dev/null
  echo "done $(date): SUMMARY_P14_confirm.json"
} 2>&1 | grep --line-buffered -v -i "warning" | tee -a ../../outputs/pilots_v4_20261103/console_confirm.log
