"""P6 + shared baselines: direct generation with the official oMeBench prompts.

  base_default   official default prompt, T=0.2 (= arm A0 of P1)
  base_cot       official CoT prompt, T=0.2 (oMe-Gold and T3_dev only)
  samp1..samp4   official default prompt, T=0.7 (pool for best-of-4 / consensus, arm A1)
Sets: oMe-Gold (196), Silver-dev (60), T3_dev (47), J3 (147).
"""
from __future__ import annotations

import argparse
import json

import common4 as K

ARMS = {'base_default': ('default', 0.2), 'base_cot': ('cot', 0.2),
        **{f'samp{i}': ('default', 0.7) for i in range(1, 5)}}


def sets():
    return {'gold': K.load_gold(), 'silver_dev': K.silver_dev(), 'T3_dev': K.load_fukuyama('T3_dev'),
            'J3': K.load_fukuyama('J3')}


def job(ds, arm, case):
    kind, temp = ARMS[arm]
    ans = K.call(f'{ds}/{arm}', case['id'], K.official_prompt(case, kind), temperature=temp)
    return 'ok' if ans else 'error'


def plan(all_sets):
    jobs = []
    for ds, cases in all_sets.items():
        arms = ['base_default'] + (['base_cot'] if ds in ('gold', 'T3_dev') else []) + \
               ([f'samp{i}' for i in range(1, 5)] if ds != 'gold' else [])
        jobs += [(ds, arm, c) for arm in arms for c in cases.values()]
    return jobs


def analyze(all_sets):
    rep = {}
    for ds, cases in all_sets.items():
        keys = ['S_total', 'S_partial', 'V', 'L', 'in_order_full', 'all_present', 'parsed'] if ds in ('gold', 'silver_dev') \
            else ['all_present', 'in_order_frac', 'parsed']
        rep[ds] = {}
        for arm in ARMS:
            rows = [(cid, K.score(c, K.parse_mech(a))) for cid, c in cases.items() if (a := K.cached(f'{ds}/{arm}', cid))]
            if rows:
                rep[ds][arm] = K.summarize(rows, keys)
    (K.OUT / 'SUMMARY_P6.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=300)
    ap.add_argument('--analyze-only', action='store_true')
    cli = ap.parse_args()
    s = sets()
    if not cli.analyze_only:
        K.run_pool(job, plan(s), cli.workers)
    analyze(s)
