#!/bin/zsh
# Overnight queue (2026-11-05). 117 gateway only, total concurrency ~300 split over two tracks.
# Every step is cached per call and independent ("; " not "&&"): a failing step does not stop the night.
cd "$(dirname "$0")"
: "${OPENAI_API_KEY:?}" "${FALLBACK_API_KEY:?}"
PY=../../../external/venv-llm/bin/python
MAP=../../../external/env-map/bin/python
O=../../outputs/pilots_v4_20261103
LOG=$O/console_night.log
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> $LOG; }
f() { grep --line-buffered -v -i warning | grep --line-buffered -v '"fn"'; }

while pgrep -f p27_g7_confirm.py > /dev/null; do sleep 30; done
log "G7 held-out confirmation finished"

# Track A (117 cap 80): literature 172 papers -> compute curve -> USPTO
(
  log "A: lit150 start"
  CAP117=80 CAPVPS=0 ./run_lit150.sh; log "A: lit150 done"
  export FAILOVER_PREFER=117 FAILOVER_CAP_117=80 FAILOVER_CAP_VPS=0
  $PY p22_extras.py --task curve --workers 80 2>&1 | f > $O/console_curve.log; log "A: curve done"
  $PY p16_supplement_g38.py --task uspto_pool --workers 80 2>&1 | f
  $PY p16_supplement_g38.py --task uspto_requests 2>&1 | f; $MAP stepmap.py 2>&1 | tail -1
  $PY p16_supplement_g38.py --task uspto_r1 --workers 80 2>&1 | f
  $PY p16_supplement_g38.py --task uspto_requests 2>&1 | f; $MAP stepmap.py 2>&1 | tail -1
  $PY p16_supplement_g38.py --task uspto_r2 --workers 80 2>&1 | f
  $PY p16_supplement_g38.py --task uspto_analyze > /dev/null 2>&1; log "A: USPTO done"
) >> $O/console_trackA.log 2>&1 &

# Track B (117 cap 120): scale-up confirmation (766, pre-registered v5) -> Guided v7 on its blind spots
(
  log "B: scale start"
  CAP117=120 CAPVPS=0 ./run_scale.sh; log "B: scale done"
  export FAILOVER_PREFER=117 FAILOVER_CAP_117=120 FAILOVER_CAP_VPS=0
  $PY p27_g7_confirm.py --phase run --sets template,silver_conf --workers 120 2>&1 | f > $O/console_g7_scale.log
  log "B: Guided v7 on scale blind spots done"
) >> $O/console_trackB.log 2>&1 &

wait
$PY p23_offline.py > /dev/null 2>&1; log "offline summary (p23) refreshed"
log "NIGHT DONE"
