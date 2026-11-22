"""P2: can explicit enumeration of competing pathways cover hidden intermediates that
plain sampling never writes (10% of J3 / 28% of T3 hidden GT species were absent from all 4
earlier samples)?

One call per case (T=0.7) asks for three mechanistically distinct candidate mechanisms,
each in the official oMeBench JSON format. Pool coverage of GT intermediates (exact PGL
ground truth, J3 and T3_dev) is compared for:
  S4    union of the 4 plain samples (p6 samp1..4)
  E3    union of the 3 enumerated pathways
  M     the analog-mapped candidate list of P1 (map_std)
  and their unions; precision = share of pool species that are GT intermediates.
"""
from __future__ import annotations

import argparse
import json
import re

import common4 as K

ENUM = ('\n\n## Different task for this reaction\n\nInstead of a single mechanism, write THREE mechanistically '
        'distinct candidate mechanisms for the reaction below. They must differ in their intermediates. Include the '
        'most direct pathway, and consider pathways through short-lived species that do not appear in the overall '
        'equation: bonds that form and later break again, rearrangements, onium or ylide intermediates, tetrahedral '
        'adducts, neighbouring-group participation, catalyst- or reagent-bound species, pericyclic versus stepwise '
        'courses. Reason first. Then give each candidate as a JSON list in the format above, between [PATH1] and '
        '[/PATH1], [PATH2] and [/PATH2], [PATH3] and [/PATH3].\n\n## Input')


def prompt(case):
    p = K.official_prompt(case).replace('Return only valid JSON. Do not include explanatory text, markdown, or code fences.',
                                        'Format each mechanism as valid JSON.')
    return p.replace('\n\n## Input', ENUM, 1)


def paths(ans):
    out = []
    for i in (1, 2, 3):
        m = re.search(rf'\[PATH{i}\](.*?)\[/PATH{i}\]', ans or '', re.S)
        out.append(K.parse_mech(m.group(1)) if m else None)
    return out


def job(ds, case):
    return 'ok' if K.call(f'{ds}/enum3', case['id'], prompt(case), temperature=0.7) else 'error'


def gt_info(case):
    endpoint, seq = K.SM.gt_species(case['pgl'])
    gt = set().union(*seq) if seq else set()
    ev = K.P.hidden_events(case['pgl'])
    hk = set().union(*(i['keys'] for i in ev.values() if i['keys'])) if ev else set()
    return endpoint, gt, gt & hk


def keys_of(smis, endpoint):
    return {k for s in smis for k in K.species_keys(s)} - endpoint - {''}


def analyze(all_sets):
    rep = {}
    for ds, cases in all_sets.items():
        tot = {}
        for cid, case in cases.items():
            smp = [K.parse_mech(K.cached(f'{ds}/samp{i}', cid)) for i in range(1, 5)]
            enum = K.cached(f'{ds}/enum3', cid)
            mp = K.OUT / ds / 'retrieved_std' / f'{cid}.json'
            if enum is None or any(s is None for s in smp) or not mp.exists():
                continue
            endpoint, gt, hid = gt_info(case)
            pools = {'S4': keys_of([s['intermediate_smiles'] for m in smp for s in m], endpoint),
                     'E3': keys_of([s['intermediate_smiles'] for m in paths(enum) if m for s in m], endpoint),
                     'M': keys_of(json.loads(mp.read_text())['candidates'], endpoint)}
            pools['S4+E3'] = pools['S4'] | pools['E3']
            pools['S4+M'] = pools['S4'] | pools['M']
            pools['S4+E3+M'] = pools['S4+E3'] | pools['M']
            for name, pool in pools.items():
                t = tot.setdefault(name, {'gt_hit': 0, 'hid_hit': 0, 'pool': 0, 'pool_gt': 0})
                t['gt_hit'] += len(gt & pool)
                t['hid_hit'] += len(hid & pool)
                t['pool'] += len(pool)
                t['pool_gt'] += len(pool & gt)
            tot.setdefault('_n', {'cases': 0, 'gt': 0, 'hid': 0})
            tot['_n']['cases'] += 1
            tot['_n']['gt'] += len(gt)
            tot['_n']['hid'] += len(hid)
        n = tot.pop('_n', None)
        if not n:
            continue
        rep[ds] = {'cases': n['cases'], 'gt_species': n['gt'], 'hidden_gt_species': n['hid'],
                   **{k: {'gt_recall': round(v['gt_hit'] / n['gt'], 3), 'hidden_recall': round(v['hid_hit'] / n['hid'], 3),
                          'precision': round(v['pool_gt'] / max(1, v['pool']), 3),
                          'pool_per_case': round(v['pool'] / n['cases'], 1)} for k, v in tot.items()}}
    (K.OUT / 'SUMMARY_P2.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=30)
    ap.add_argument('--analyze-only', action='store_true')
    cli = ap.parse_args()
    s = {'T3_dev': K.load_fukuyama('T3_dev'), 'J3': K.load_fukuyama('J3')}
    if not cli.analyze_only:
        K.run_pool(job, [(ds, c) for ds, cs in s.items() for c in cs.values()], cli.workers)
    analyze(s)
