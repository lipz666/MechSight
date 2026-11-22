#!/bin/zsh
# Batch 1 of the reviewer reruns (2026-11-04): USPTO in-template run, argon on all of oMe-Gold,
# ablation ladder arm. 117 gateway first (FAILOVER_PREFER=117); keys only from the environment.
cd "$(dirname "$0")"
: "${OPENAI_API_KEY:?}" "${FALLBACK_API_KEY:?}"
export FAILOVER_PREFER=117
PY=../../../external/venv-llm/bin/python
MAP=../../../external/env-map/bin/python
O=../../outputs/pilots_v4_20261103
f() { grep --line-buffered -v -i warning; }

( export FAILOVER_CAP_117=150 FAILOVER_CAP_VPS=60
  $PY p16_supplement_g38.py --task uspto_pool --workers 210
  $PY p16_supplement_g38.py --task uspto_requests && $MAP stepmap.py
  $PY p16_supplement_g38.py --task uspto_r1 --workers 210
  $PY p16_supplement_g38.py --task uspto_requests && $MAP stepmap.py
  $PY p16_supplement_g38.py --task uspto_r2 --workers 210
  $PY p16_supplement_g38.py --task uspto_analyze > /dev/null
  echo "USPTO done $(date)" ) 2>&1 | f > $O/console_b1_uspto.log &

( export FAILOVER_CAP_117=100 FAILOVER_CAP_VPS=40
  $PY p19_ladder.py --phase run --workers 140 > /dev/null
  echo "ladder done $(date)" ) 2>&1 | f > $O/console_b1_ladder.log &

( for ph in pool requests r1 requests r2 relabel; do $PY p17_argon_confirm.py --phase $ph --sets gold --workers 100; done
  $PY p17_argon_confirm.py --phase analyze --sets T3_confirm,gold > /dev/null
  echo "argon done $(date)" ) 2>&1 | f > $O/console_b1_argon.log &

wait
echo "batch1 all done $(date)"
