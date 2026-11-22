"""D1: does the built-in self-audit turn recover hidden events? Zero new calls.

Every executable condition (E, E_r2, E_nofacts, E_cue) answers twice: a path, then an audit turn
that may revise it. The traces keep both answers, so hidden-event recall before and after the
audit can be compared on the same samples.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe as P  # noqa: E402
from probe import A, fm, metrics  # noqa: E402

CONDS = ['E', 'E_r2', 'E_nofacts', 'E_cue']


def answers(cond, cid):
    f = P.OUT / cond / 'traces' / f'{cid}.jsonl'
    if not f.exists():
        return None, None
    path = audit = None
    for line in f.read_text().splitlines():
        ev = json.loads(line)
        if 'answer' not in ev:
            continue
        if ev['phase'].endswith('.path'):
            path = ev['answer']
        elif ev['phase'].endswith('.audit'):
            audit = ev['answer']
    return path, audit


def parse(text):
    try:
        return [repr(a) for a in A.P23.parse_path(text)]
    except Exception:
        return None


def main():
    import j3_dev
    cases = j3_dev.load_cases()
    rows, l1 = [], []
    for cond in CONDS:
        for cid, case in cases.items():
            path, audit = answers(cond, cid)
            before, after = parse(path or ''), parse(audit or '')
            if before is None or after is None:
                continue
            sb, sa = fm.score_case(case, before), fm.score_case(case, after)
            l1.append({'cond': cond, 'case': cid, 'before': float(sb['sigma_L1']), 'after': float(sa['sigma_L1']),
                       'changed': before != after})
            ub, ua = set(map(tuple, sb['predicted_transient'])), set(map(tuple, sa['predicted_transient']))
            for t, info in P.hidden_events(case).items():
                if info['keys'] and info['heavy']:
                    rows.append({'cond': cond, 'case': cid, 'before': t in ub, 'after': t in ua})

    def ci(vals, ids):
        lo, hi = metrics.clustered_bootstrap_ci(vals, ids, seed=20261102)
        return [round(statistics.mean(vals), 3), round(lo, 3), round(hi, 3), len(vals)]

    ids = [f"{r['cond']}:{r['case']}" for r in rows]
    lid = [f"{r['cond']}:{r['case']}" for r in l1]
    report = {
        'samples': len(l1), 'paths_changed_by_audit': sum(r['changed'] for r in l1),
        'heavy_hidden_recall_before': ci([float(r['before']) for r in rows], ids),
        'heavy_hidden_recall_after': ci([float(r['after']) for r in rows], ids),
        'heavy_hidden_after_minus_before': ci([float(r['after']) - float(r['before']) for r in rows], ids),
        'hidden_events_gained_by_audit': sum(r['after'] and not r['before'] for r in rows),
        'hidden_events_lost_by_audit': sum(r['before'] and not r['after'] for r in rows),
        'sigma_L1_before': ci([r['before'] for r in l1], lid), 'sigma_L1_after': ci([r['after'] for r in l1], lid),
        'sigma_L1_fixed': sum(r['after'] > r['before'] for r in l1), 'sigma_L1_broken': sum(r['after'] < r['before'] for r in l1),
    }
    (P.OUT / 'SUMMARY_self_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
