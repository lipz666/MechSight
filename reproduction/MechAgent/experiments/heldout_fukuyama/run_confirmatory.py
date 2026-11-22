"""Run the frozen shortlist method and the direct baseline on the held-out set.

The method code is imported from experiments/p23_planning_probe/run.py, and
every component is checked against the hashes in frozen_method/MANIFEST.json
before any call is made; a mismatch aborts the run. Results are cached per
arm and case, so an interrupted run resumes; infrastructure failures are
retried on the same case, never replaced.

  phase 1: direct, direct@2, direct@3 (compute-matched extra), rank
  phase 2: shortlist, shortlist@2 (both use the phase-1 rank result)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments' / 'p23_planning_probe'))
import run as R

BASE = ROOT / 'outputs' / 'heldout_fukuyama_20261024'
PHASES = {1: ['direct', 'direct@2', 'direct@3', 'rank'], 2: ['shortlist', 'shortlist@2']}


def verify_frozen():
    manifest = json.loads((BASE / 'frozen_method' / 'MANIFEST.json').read_text())
    for name, expected in manifest['component_sha256'].items():
        value = getattr(R, name)
        text = value if isinstance(value, str) else inspect.getsource(value)
        if hashlib.sha256(text.encode()).hexdigest() != expected:
            raise SystemExit(f'Frozen component changed: {name}; refusing to run.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', type=int, choices=[1, 2], required=True)
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--limit', type=int, default=0, help='smoke test on the first N registered cases')
    ap.add_argument('--passes', type=int, default=3)
    ap.add_argument('--output', type=Path, default=BASE / 'confirmatory')
    cli = ap.parse_args()
    verify_frozen()
    reg = json.loads((BASE / 'REGISTRATION.json').read_text())
    wanted = reg['sets']['all_218']
    cases = {c['case_id']: c for c in map(json.loads, (BASE / 'heldout.jsonl').read_text().splitlines())}
    run_cases = [cases[c] for c in wanted][: cli.limit or None]
    args = SimpleNamespace(output=cli.output, base_url=reg['model']['base_url'], model=reg['model']['model'],
                           timeout=300.0, max_tokens=8192, mock=False)
    args.output.mkdir(parents=True, exist_ok=True)
    modules = {'prompts': importlib.import_module('exp1_agent.prompts'),
               'skills': importlib.import_module('exp1_agent.planner_skills')}
    jobs = [(arm, c) for c in run_cases for arm in PHASES[cli.phase]]
    for attempt in range(cli.passes):
        pending = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=cli.workers) as pool:
            futures = {pool.submit(R.rank_case if arm == 'rank' else R.run_case, arm, case, args, modules): (arm, case)
                       for arm, case in jobs}
            for fut in concurrent.futures.as_completed(futures):
                arm, case = futures[fut]
                try:
                    status = fut.result()[-1]
                except Exception as exc:
                    status = f'error:{type(exc).__name__}'
                if status not in ('ok', 'cached'):
                    pending.append((arm, case))
                print(json.dumps({'pass': attempt + 1, 'done': [arm, case['case_id'], status]}), flush=True)
        print(json.dumps({'pass': attempt + 1, 'pending': len(pending)}), flush=True)
        if not pending:
            break
        jobs = pending


if __name__ == '__main__':
    main()
