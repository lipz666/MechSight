"""Compare arm families on the 24 dev cases with the primary Fukuyama metrics.

A family is an arm plus its repeat samples ('cond', 'cond@2', ...). Per case
the score is the mean over that family's samples (a pass@1 estimate); paired
differences against `direct` (4 samples) are bootstrapped over cases.
"""
from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from develop import fukuyama_metrics as fm
from develop import metrics

OUT = ROOT / 'outputs' / 'p23_planning_probe_20261024'
FAMILIES = ['direct', 'interm', 'latent', 'cond', 'cond_analog', 'oracle_lat', 'oracle_half', 'oracle_noisy', 'enum', 'shortlist']
METRICS = ['sigma_L1', 'intermediates_in_order']


def main():
    cases = {c['case_id']: c for c in map(json.loads, (ROOT / 'outputs/fukuyama/clean.jsonl').read_text().splitlines())}
    ids = json.loads((OUT / 'cases.json').read_text())
    per = collections.defaultdict(dict)
    samples = collections.Counter()
    tokens = collections.defaultdict(list)
    for fam in FAMILIES:
        dirs = [d for d in OUT.iterdir() if d.is_dir() and d.name.split('@')[0] == fam]
        for cid in ids:
            scores = []
            for d in dirs:
                f = d / f'{cid}.json'
                if f.exists() and json.loads(f.read_text()).get('finished'):
                    r = json.loads(f.read_text())
                    scores.append(fm.score_case(cases[cid], r['actions_pgl']))
                    tokens[fam].append(r['generation_tokens'])
            if scores:
                per[fam][cid] = {m: statistics.mean(float(s[m]) for s in scores) for m in METRICS}
                samples[(fam, len(scores))] += 1
    strata = {cid: len(fm.sigma_scores([], fm.gold_steps_from_clean(cases[cid]))['gold_transient']) for cid in ids}
    label = lambda n: '0' if n == 0 else '1-2' if n <= 2 else '>=3'
    report = {'samples': {f'{f}x{n}': c for (f, n), c in sorted(samples.items())}}
    for fam in FAMILIES:
        if not per[fam]:
            continue
        block = {'cases': len(per[fam]), 'mean_generation_tokens': round(statistics.mean(tokens[fam]))}
        for m in METRICS:
            vals = [per[fam][c][m] for c in ids if c in per[fam]]
            block[m] = round(statistics.mean(vals), 3)
            block[m + '_by_transient'] = {s: round(sum(per[fam][c][m] for c in ids if c in per[fam] and label(strata[c]) == s), 2)
                                          for s in ('0', '1-2', '>=3')}
            if fam != 'direct':
                keys = [c for c in ids if c in per[fam] and c in per['direct']]
                d = [per[fam][c][m] - per['direct'][c][m] for c in keys]
                lo, hi = metrics.clustered_bootstrap_ci(d, keys, seed=20261024)
                block[m + '_vs_direct'] = {'delta': round(statistics.mean(d), 3), 'ci95': [round(lo, 3), round(hi, 3)],
                                           'better_cases': sum(x > 0 for x in d), 'worse_cases': sum(x < 0 for x in d)}
        report[fam] = block
    report['strata_sizes'] = dict(collections.Counter(label(n) for n in strata.values()))
    (OUT / 'compare_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (OUT / 'compare_rows.json').write_text(json.dumps({f: per[f] for f in FAMILIES}, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
