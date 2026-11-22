#!/bin/zsh
# Gateway-degraded mode (2026-11-04 evening): vps gemini-3.8-flash-high returns 503 auth_unavailable and 117
# returns 502 above ~30 concurrent long calls. Run only the literature set and the two small extras.
cd "$(dirname "$0")"
: "${OPENAI_API_KEY:?}" "${FALLBACK_API_KEY:?}"
PY=../../../external/venv-llm/bin/python
O=../../outputs/pilots_v4_20261103
CAP117=${L117:-120} CAPVPS=0 ./run_lit150.sh &
( export FAILOVER_PREFER=117 FAILOVER_CAP_117=${X117:-40} FAILOVER_CAP_VPS=0
  for t in blind randsmi; do $PY p22_extras.py --task $t --workers 80 > /dev/null; echo "$t done $(date)"; done
) 2>&1 | grep --line-buffered -v -i warning > $O/console_p22.log &
wait
echo "priority done $(date)"
