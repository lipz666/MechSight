#!/bin/zsh
# Dev run of Mech-Agent v5 (P11) and the Oracle / self-refine controls (P12) on T3_dev + J3.
# Needs OPENAI_API_KEY (vps) and FALLBACK_API_KEY (117) in the environment; keys are never written.
set -e
cd "$(dirname "$0")"
: "${OPENAI_API_KEY:?export OPENAI_API_KEY first}" "${FALLBACK_API_KEY:?export FALLBACK_API_KEY first}"
PY=../../../external/venv-llm/bin/python
MAP=../../../external/env-map/bin/python
LOG=../../outputs/pilots_v4_20261103/console_p11_p12.log
{
  echo "start $(date)"
  FAILOVER_CAP_VPS=200 FAILOVER_CAP_117=200 $PY p11_mechagent.py --phase r1 --workers 400 &
  FAILOVER_CAP_VPS=100 FAILOVER_CAP_117=100 $PY p12_controls.py --phase oracle --workers 200 &
  FAILOVER_CAP_VPS=100 FAILOVER_CAP_117=100 $PY p12_controls.py --phase refine --workers 200 &
  wait
  echo "round 1 + controls done $(date)"
  $PY p11_mechagent.py --phase requests
  $MAP stepmap.py
  FAILOVER_CAP_VPS=400 FAILOVER_CAP_117=400 $PY p11_mechagent.py --phase r2 --workers 800
  $PY p11_mechagent.py --phase analyze > /dev/null
  $PY p12_controls.py --phase analyze > /dev/null
  echo "done $(date): SUMMARY_P11.json, SUMMARY_P12.json"
} 2>&1 | grep -v -i "warning" | tee -a $LOG
