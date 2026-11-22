#!/bin/zsh
# argon jobs one after another (cap 100): wave-1 extraction (running) -> wave-2 extraction -> layer-2 gold
cd "$(dirname "$0")"
PY=../../../external/venv-llm/bin/python
while kill -0 61430 2>/dev/null; do sleep 30; done
ARGON_CAP=100 $PY extract_scale.py --workers 100 > extract2.log 2>&1
ARGON_CAP=100 $PY ability_scale.py --phase gold --workers 100 > gold.log 2>&1
echo chain done
