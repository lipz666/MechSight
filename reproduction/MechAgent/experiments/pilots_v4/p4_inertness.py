"""P4: fair inertness recheck in SMILES (J3, gemini-3.8-flash).

The SMILES inertness figure compared the union of 5 free-text narratives (S, G at default
temperature, N1-N3 at T=0.7) with ONE resample (M2, T=0.2). Here five more resamples of the
same SMILES-mechanism prompt (smiles_mech.TASK) are drawn at T=0.7 (R1..R5). For the heavy
hidden events omitted in M1 (outputs/smiles_main_20261103/J3/gemini/M1):
  per draw     re-appearance in one resample vs. presence in one narrative
  union of 5   any of R1..R5 vs. any of the 5 narratives (equal draws)
Inertness holds only if the narrative side exceeds the resample side by >= 15 pp at equal draws.
"""
from __future__ import annotations

import argparse
import json
import re

import common4 as K

SMAIN = K.ROOT / 'outputs' / 'smiles_main_20261103' / 'J3' / 'gemini'
IK = K.ROOT / 'outputs' / 'inert_knowledge_20261102'
ACT = K.ROOT / 'outputs' / 'activation_pilot_20261103'
NARR = {'S': IK / 'S', 'G': IK / 'G', 'N1': ACT / 'N1', 'N2': ACT / 'N2', 'N3': ACT / 'N3'}
RES = [f'R{i}' for i in range(1, 6)]


def prompt(case):
    return K.SM.TASK.format(reactants='.'.join(case['reactants']), products='.'.join(case['products']),
                            conditions=case['conditions'])


def job(arm, case):
    return 'ok' if K.call(f'J3/p4_{arm}', case['id'], prompt(case), temperature=0.7) else 'error'


def states(ans):
    m = re.search(r'<mechanism>(.*?)</mechanism>', ans or '', re.S)
    lines = [ln.strip().strip('`').strip() for ln in (m.group(1) if m else '').splitlines()]
    return [re.sub(r'^(step\s*\d+\s*[:.)-]\s*|\d+\s*[:.)]\s*)', '', ln, flags=re.I) for ln in lines if ln]


def analyze(cases):
    rows = []
    for cid, case in cases.items():
        m1 = K.PL.load(SMAIN / 'M1', cid)
        res = [K.cached(f'J3/p4_{r}', cid) for r in RES]
        if not m1 or any(r is None for r in res):
            continue
        s1 = K.SM.score(case['pgl'], m1['states'])
        rs = [K.SM.score(case['pgl'], states(a))['heavy_events'] for a in res]
        ev = K.P.hidden_events(case['pgl'])
        narr = {}
        for n, f in NARR.items():
            rec = K.PL.load(f, cid)
            narr[n] = {k for s in (rec['intermediates'] if rec else []) for k, _ in K.P.smiles_keys(s)}
        for t, hit in s1['heavy_events'].items():
            if hit:
                continue
            keys = ev[tuple(t.split('-'))]['keys']
            rows.append({'case': cid, 'res': [r[t] for r in rs], 'narr': [bool(keys & narr[n]) for n in NARR]})
    ids = [r['case'] for r in rows]
    ci = lambda v: K.phase2.ci([float(x) for x in v], ids)
    rep = {'omitted_heavy_events_M1': len(rows), 'cases': len(set(ids)),
           'per_draw_resample': ci([sum(r['res']) / 5 for r in rows]),
           'per_draw_narrative': ci([sum(r['narr']) / 5 for r in rows]),
           'union5_resample': ci([any(r['res']) for r in rows]),
           'union5_narrative': ci([any(r['narr']) for r in rows]),
           'union5_narrative_minus_resample': ci([any(r['narr']) - any(r['res']) for r in rows]),
           'criterion': 'inertness if union5 narrative - resample >= 0.15'}
    (K.OUT / 'SUMMARY_P4.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=200)
    ap.add_argument('--analyze-only', action='store_true')
    cli = ap.parse_args()
    cases = K.load_fukuyama('J3')
    if not cli.analyze_only:
        K.run_pool(job, [(r, c) for r in RES for c in cases.values()], cli.workers)
    analyze(cases)
