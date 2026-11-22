#!/bin/zsh
# Expanded 2026 literature set: flash-lite extraction (vps), then pool / v5 / clue arms (gemini-3.8-flash, 117 first).
cd "$(dirname "$0")"
: "${OPENAI_API_KEY:?}" "${FALLBACK_API_KEY:?}"
export FAILOVER_PREFER=117 FAILOVER_CAP_117=${CAP117:-60} FAILOVER_CAP_VPS=${CAPVPS:-100}
O=../../outputs/pilots_v4_20261103
export STEPMAP_REQ=$O/stepmap_lit150_requests.jsonl STEPMAP_CACHE=$O/stepmap_lit150_cache.json
PY=../../../external/venv-llm/bin/python
MAP=../../../external/env-map/bin/python
W=$((FAILOVER_CAP_117 + FAILOVER_CAP_VPS))
{
  echo "start $(date)"
  $PY p20_lit150.py --phase extract --workers 40
  $PY p20_lit150.py --phase pool --workers $W
  $PY -c "import p20_lit150 as L, p11_mechagent as M; M.requests({'lit150': L.cases()})" && $MAP stepmap.py
  $PY p20_lit150.py --phase r1 --workers $W
  $PY -c "import p20_lit150 as L, p11_mechagent as M; M.requests({'lit150': L.cases()})" && $MAP stepmap.py
  $PY p20_lit150.py --phase r2 --workers $W
  $PY p20_lit150.py --phase clues --workers $W
  $PY p20_lit150.py --phase analyze > /dev/null
  echo "lit150 done $(date)"
} 2>&1 | grep --line-buffered -v -i warning > $O/console_lit150.log
