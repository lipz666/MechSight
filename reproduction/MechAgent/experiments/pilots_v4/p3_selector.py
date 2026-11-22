"""P3: which selector recovers the gain hidden in a pool of candidate mechanisms?

Pool per case: A0, samp1..4, A2, A3 (P1, std mode) and, on Fukuyama sets, the three P2
pathways. Selectors:
  oracle      best candidate by the target metric (upper bound)
  consensus   highest mean species-key Jaccard to the other candidates (MBR)
  valid_cons  consensus among candidates that parse and whose last step contains the product
  judge       listwise LLM judge (one gemini call, T=0.2) shown up to 8 candidates in random order
Gate: a selector captures >= 50% of the oracle gain over A0.
"""
from __future__ import annotations

import argparse
import json
import random
import re

import common4 as K
import p2_enumerate as P2

JUDGE = ('Reactants and reagents (SMILES): {reactants}\nProducts (SMILES): {products}\nReaction conditions: '
         '{conditions}\n\nBelow are candidate stepwise mechanisms (intermediate SMILES after each step). Judge them as an '
         'expert physical organic chemist: which one is the most plausible complete mechanism under these conditions? '
         'Consider every intermediate, including short-lived ones, and whether each step is a sound elementary step.\n\n'
         '{cands}\n\nReason, then give the number of the best candidate inside <best> and </best>.')


def pool(ds, cid):
    names = ['base_default'] + [f'samp{i}' for i in range(1, 5)] + ['A2_std', 'A3_std']
    out = {}
    for n in names:
        m = K.parse_mech(K.cached(f'{ds}/{n}', cid))
        if m:
            out[n] = m
    if ds != 'silver_dev':
        for i, m in enumerate(P2.paths(K.cached(f'{ds}/enum3', cid)), 1):
            if m:
                out[f'enum{i}'] = m
    return out


def render(cands):
    blocks = []
    for i, m in enumerate(cands, 1):
        blocks.append(f'Candidate {i}:\n' + '\n'.join(f"  step {j}: {s.get('subtype', '')} -> {s['intermediate_smiles']}"
                                                         for j, s in enumerate(m, 1)))
    return '\n\n'.join(blocks)


def order(cid, names):
    names = sorted(names)
    random.Random(cid).shuffle(names)
    return names[:8]


def job(ds, case):
    p = pool(ds, case['id'])
    if len(p) < 2:
        return 'skip'
    names = order(case['id'], p)
    prompt = JUDGE.format(reactants='.'.join(case['reactants']), products='.'.join(case['products']),
                          conditions=case['conditions'], cands=render([p[n] for n in names]))
    return 'ok' if K.call(f'{ds}/judge', case['id'], prompt) else 'error'


def _keys(m):
    return {k for s in m for k in K.species_keys(s.get('intermediate_smiles') or '')}


def consensus(p, names):
    ks = {n: _keys(p[n]) for n in names}
    jac = lambda a, b: len(a & b) / len(a | b) if a | b else 1.0
    return max(names, key=lambda n: (sum(jac(ks[n], ks[o]) for o in names if o != n), n == 'base_default'))


def reaches_product(case, m):
    prod = {k for s in case['products'] for k in K.species_keys(s)}
    return bool(m) and bool(prod & K.species_keys(m[-1].get('intermediate_smiles') or ''))


def analyze(all_sets):
    rep = {}
    for ds, cases in all_sets.items():
        metric = 'all_present' if ds != 'silver_dev' else 'S_partial'
        rows = {k: [] for k in ('A0', 'oracle', 'consensus', 'valid_cons', 'judge')}
        for cid, case in cases.items():
            p = pool(ds, cid)
            if 'base_default' not in p or len(p) < 2:
                continue
            sc = {n: K.score(case, m) for n, m in p.items()}
            rows['A0'].append((cid, sc['base_default']))
            rows['oracle'].append((cid, max(sc.values(), key=lambda s: s[metric])))
            rows['consensus'].append((cid, sc[consensus(p, list(p))]))
            valid = [n for n in p if reaches_product(case, p[n])] or list(p)
            rows['valid_cons'].append((cid, sc[consensus(p, valid)]))
            ans = K.cached(f'{ds}/judge', cid)
            m = re.search(r'<best>\s*(\d+)', ans or '')
            names = order(cid, p)
            pick = names[int(m.group(1)) - 1] if m and 0 < int(m.group(1)) <= len(names) else 'base_default'
            rows['judge'].append((cid, sc[pick]))
        if not rows['A0']:
            continue
        keys = [metric, 'all_present'] if metric != 'all_present' else ['all_present']
        block = {k: K.summarize(v, keys) for k, v in rows.items()}
        gain = block['oracle'][metric][0] - block['A0'][metric][0]
        for k in ('consensus', 'valid_cons', 'judge'):
            block[f'{k}_minus_A0'] = K.paired(rows[k], rows['A0'], metric)
            block[f'{k}_share_of_oracle_gain'] = round((block[k][metric][0] - block['A0'][metric][0]) / gain, 2) if gain else None
        rep[ds] = block
    (K.OUT / 'SUMMARY_P3.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=30)
    ap.add_argument('--analyze-only', action='store_true')
    cli = ap.parse_args()
    s = {'T3_dev': K.load_fukuyama('T3_dev'), 'J3': K.load_fukuyama('J3'), 'silver_dev': K.silver_dev()}
    if not cli.analyze_only:
        K.run_pool(job, [(ds, c) for ds, cs in s.items() for c in cs.values()], cli.workers)
    analyze(s)
