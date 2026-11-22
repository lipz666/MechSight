"""P3' pilot: is the hidden-event answer among independent samples, and can it be picked?

Samples: the frozen `direct` arm plus `direct@2..@4` (same prompts, temperature
0.2, two calls each). Reported per metric (sigma_L1, intermediates_in_order):

  pass@k   unbiased estimator 1 - C(n-c, k)/C(n, k) with n = 4 samples
  vote@4   the most frequent sigma-event multiset among the 4 samples
           (ties -> earliest sample); uses no GT
  reach+vote  vote among the samples whose final state contains every
           molecule of the supplied target state (an input, not GT); falls
           back to vote@4 when none reaches it

against the single-sample oracle_lat diagnostic (GT transient bonds given).
"""
from __future__ import annotations

import collections
import json
import statistics
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from develop import fukuyama_metrics as fm
from develop import metrics
from develop import standalone_chemistry as chem

OUT = ROOT / 'outputs' / 'p23_planning_probe_20261024'
SAMPLES = ['direct', 'direct@2', 'direct@3', 'direct@4']
METRICS = ['sigma_L1', 'intermediates_in_order']


def pass_at(n, c, k):
    return 1.0 - comb(n - c, k) / comb(n, k) if n - c >= k else 1.0


def sigma_signature(case, actions):
    states, _ = fm.replay(case['initial_state_pgl'], actions)
    initial = chem.parse_pgl_literal(case['initial_state_pgl'])
    steps = [fm.step_from_states(b, a) for b, a in zip([initial] + states[:-1], states)]
    total = sum((fm.sigma_events(s) for s in steps), collections.Counter())
    target = fm.state_view(chem.parse_pgl_literal(case['target_state_pgl']))
    reaches = bool(states) and fm.same_state(fm.state_view(states[-1]), target)
    return tuple(sorted(total.items())), reaches


def vote(candidates):
    counts = collections.Counter(sig for _, sig in candidates)
    best = max(counts.values())
    return next(i for i, sig in candidates if counts[sig] == best)


def ci(values, ids):
    lo, hi = metrics.clustered_bootstrap_ci(values, ids, seed=20261024)
    return [round(lo, 3), round(hi, 3)]


def main():
    cases = {c['case_id']: c for c in map(json.loads, (ROOT / 'outputs/fukuyama/clean.jsonl').read_text().splitlines())}
    ids = json.loads((OUT / 'cases.json').read_text())
    rows = []
    for cid in ids:
        case = cases[cid]
        samples = []
        for arm in SAMPLES:
            path = OUT / arm / f'{cid}.json'
            r = json.loads(path.read_text()) if path.exists() else None
            if not r or not r.get('finished'):
                continue
            score = fm.score_case(case, r['actions_pgl'])
            sig, reaches = sigma_signature(case, r['actions_pgl'])
            samples.append({'arm': arm, 'sig': sig, 'reaches_target': reaches, **{m: score[m] for m in METRICS}})
        oracle = fm.score_case(case, json.loads((OUT / 'oracle_lat' / f'{cid}.json').read_text())['actions_pgl'])
        n = len(samples)
        row = {'case_id': cid, 'partition': case['partition'], 'n_samples': n,
               'transient': len(fm.sigma_scores([], fm.gold_steps_from_clean(case))['gold_transient']),
               'distinct_sigma_sets': len({s['sig'] for s in samples}),
               'reaching_target': sum(s['reaches_target'] for s in samples)}
        voted = vote([(i, s['sig']) for i, s in enumerate(samples)])
        reaching = [(i, s['sig']) for i, s in enumerate(samples) if s['reaches_target']]
        reach_voted = vote(reaching) if reaching else voted
        for m in METRICS:
            c = sum(bool(s[m]) for s in samples)
            row[f'{m}_correct'] = c
            for k in (1, 2, 4):
                row[f'{m}_pass@{k}'] = pass_at(n, c, k) if n >= k else None
            row[f'{m}_vote@4'] = bool(samples[voted][m])
            row[f'{m}_reach+vote'] = bool(samples[reach_voted][m])
            row[f'{m}_oracle'] = bool(oracle[m])
            row[f'{m}_sample1'] = bool(samples[0][m])
        rows.append(row)

    summary = {'cases': len(rows), 'samples_per_case': dict(collections.Counter(r['n_samples'] for r in rows)),
               'mean_distinct_sigma_sets': statistics.mean(r['distinct_sigma_sets'] for r in rows),
               'mean_samples_reaching_target': statistics.mean(r['reaching_target'] for r in rows)}
    for m in METRICS:
        block = {}
        for key in ('sample1', 'pass@1', 'pass@2', 'pass@4', 'vote@4', 'reach+vote', 'oracle'):
            sel = [r for r in rows if r[f'{m}_{key}'] is not None]
            vals = [float(r[f'{m}_{key}']) for r in sel]
            block[key] = {'cases': len(sel), 'rate': round(statistics.mean(vals), 3) if vals else None,
                          'ci95': ci(vals, [r['case_id'] for r in sel]) if vals else None}
        for a, b in (('pass@1', 'pass@4'), ('pass@4', 'oracle'), ('pass@1', 'vote@4'), ('pass@1', 'reach+vote')):
            sel = [r for r in rows if r[f'{m}_{a}'] is not None and r[f'{m}_{b}'] is not None]
            d = [float(r[f'{m}_{b}']) - float(r[f'{m}_{a}']) for r in sel]
            block[f'{a}->{b}'] = {'cases': len(sel), 'delta': round(statistics.mean(d), 3) if d else None,
                                  'ci95': ci(d, [r['case_id'] for r in sel]) if d else None}
        strata = collections.defaultdict(list)
        for r in rows:
            strata['0' if r['transient'] == 0 else '1-2' if r['transient'] <= 2 else '>=3'].append(r)
        block['by_transient'] = {s: {'cases': len(v), **{k: round(sum(float(r[f'{m}_{k}'] or 0) for r in v), 2)
                                                        for k in ('pass@1', 'pass@4', 'vote@4', 'oracle')}}
                                 for s, v in sorted(strata.items())}
        summary[m] = block
    (OUT / 'passk_rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    (OUT / 'passk_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
