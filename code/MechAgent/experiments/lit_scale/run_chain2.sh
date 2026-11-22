#!/bin/zsh
# after layer-2 gold (argon) -> argon R1 (cap 100); after flash/lite R1 -> layer-2 ability arms (flash)
cd "$(dirname "$0")"
PY=../../../external/venv-llm/bin/python
while kill -0 62119 2>/dev/null; do sleep 30; done
ARGON_CAP=100 $PY r1_scale.py --models argon --workers 100 > r1_argon.log 2>&1 &
A=$!
while kill -0 65184 2>/dev/null; do sleep 30; done
$PY ability_scale.py --phase run --workers 200 > ability_run.log 2>&1
wait $A
echo chain2 done
