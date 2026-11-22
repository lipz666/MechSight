#!/bin/bash
# Keeps the specialist runs alive: restarts any shard that died before printing "done";
# once both ChRIMP shards are done, starts a second FlowER process walking from the other end.
cd ~/mech_agent/spec || exit 1
PY=~/miniconda3/envs/flower/bin/python
exec 9>.watchdog.lock; flock -n 9 || exit 0
run() {  # $1 log name, rest = run_spec args
  log=spec_v5_$1.log; shift
  grep -q '^done' "$log" 2>/dev/null && return
  pgrep -f "run_spec.py $*\$" >/dev/null && return
  echo "$(date '+%F %T') (re)start $*" >> watchdog.log
  nohup setsid $PY run_spec.py "$@" >> "$log" 2>&1 < /dev/null &
}
run chrimp_0 chrimp spec_inputs.json spec_v5/chrimp 0/2
run chrimp_1 chrimp spec_inputs.json spec_v5/chrimp 1/2
run flower_0 flower spec_inputs.json spec_v5/flower 0/1
if grep -q '^done' spec_v5_chrimp_0.log 2>/dev/null && grep -q '^done' spec_v5_chrimp_1.log 2>/dev/null; then
  run flower_rev flower spec_inputs.json spec_v5/flower 0/1 rev
fi
