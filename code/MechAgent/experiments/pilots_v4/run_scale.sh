#!/bin/zsh
# Pre-registered scale-up confirmation (REGISTRATION_v5_scale.json): oMe-Template + 600 oMe-Silver.
cd "$(dirname "$0")"
: "${OPENAI_API_KEY:?}" "${FALLBACK_API_KEY:?}"
export FAILOVER_PREFER=117 FAILOVER_CAP_117=${CAP117:-50} FAILOVER_CAP_VPS=${CAPVPS:-250}
O=../../outputs/pilots_v4_20261103
export STEPMAP_REQ=$O/stepmap_scale_requests.jsonl STEPMAP_CACHE=$O/stepmap_scale_cache.json
[ -f $STEPMAP_CACHE ] || cp $O/stepmap_cache.json $STEPMAP_CACHE
PY=../../../external/venv-llm/bin/python
MAP=../../../external/env-map/bin/python
W=$((FAILOVER_CAP_117 + FAILOVER_CAP_VPS))
{
  echo "start $(date)"
  $PY p21_scale.py --phase pool --workers $W
  $PY p21_scale.py --phase requests && $MAP stepmap.py
  $PY p21_scale.py --phase r1 --workers $W
  $PY p21_scale.py --phase requests && $MAP stepmap.py
  $PY p21_scale.py --phase r2 --workers $W
  $PY p21_scale.py --phase relabel --workers $W
  $PY p21_scale.py --phase analyze > /dev/null
  echo "scale done $(date)"
} 2>&1 | grep --line-buffered -v -i warning > $O/console_scale.log
