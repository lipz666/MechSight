"""Resume a frozen run, refresh its report, and finalize its status on exit."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def counts(out):
    result = {}
    for arm in ['direct', 'agent_no_skills', 'agent_skills79']:
        paths = list((out / arm).glob('exp1_*.json'))
        result[arm] = 0
        for path in paths:
            try:
                result[arm] += bool(json.loads(path.read_text()).get('finished'))
            except FileNotFoundError:
                continue
    return result


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def refresh_report(here, out):
    result = subprocess.run([sys.executable, str(here / 'summarize.py'), '--output', str(out)],
                            stdout=subprocess.DEVNULL, check=False)
    if result.returncode:
        print(json.dumps({'event': 'report_refresh_failed', 'exit_code': result.returncode,
                          'inference_continues': True}), flush=True)
    return result.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    cfg = json.loads((out / 'run_config.json').read_text())
    here = Path(__file__).resolve().parent
    command = [sys.executable, '-u', str(here / 'run.py')]
    for key, value in cfg.items():
        if key == 'notes':
            continue
        command.append('--' + key.replace('_', '-'))
        command.extend(map(str, value)) if isinstance(value, list) else command.append(str(value))
    history = out / 'supervisor_history.jsonl'
    stamp = time.strftime('%Y%m%dT%H%M%S', time.gmtime())
    log = out / f'console_resume_{stamp}.log'
    with history.open('a') as f:
        f.write(json.dumps({'event': 'resume', 'unix': time.time(), 'counts': counts(out), 'log': str(log)}) + '\n')
    with log.open('a') as handle:
        child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        write_json(out / 'supervisor.json', {'pid': os.getpid(), 'runner_pid': child.pid,
                                           'log': str(log), 'started_unix': time.time()})
        previous = None
        while child.poll() is None:
            current = counts(out)
            if current != previous:
                print(json.dumps({'event': 'progress', 'counts': current, 'unix': time.time()}), flush=True)
                previous = current if refresh_report(here, out) else None
            time.sleep(20)
    current = counts(out)
    complete = all(n == 160 for n in current.values())
    status = {'status': 'complete' if complete else 'needs_attention', 'expected_rollouts': 480,
              'valid_completed_rollouts': sum(current.values()), 'counts': current,
              'runner_stopped': True, 'exit_code': child.returncode, 'updated_unix': time.time()}
    write_json(out / 'STATUS.json', status)
    refresh_report(here, out)
    with history.open('a') as f:
        f.write(json.dumps({'event': 'exit', **status}) + '\n')
    print(json.dumps(status), flush=True)
    raise SystemExit(0 if complete else child.returncode or 1)


if __name__ == '__main__':
    main()
