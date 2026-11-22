"""P1: does analogical retrieval supply hidden intermediates the model does not propose?

Arms (official oMeBench format and prompt in every arm, T=0.2):
  A0  direct generation                                     (p6 base_default)
  A1  4 samples at T=0.7: oracle best-of-4 and consensus     (p6 samp1..4)
  A2  naive RAG: 3 retrieved analog mechanisms pasted in
  A3  MechSight-lite: identify -> retrieve -> map analog intermediates onto the target
      (candidate list) -> rewrite with the candidates visible
Sets: T3_dev (47), J3 (147), Silver-dev (60) in two modes: std (same-name analogs allowed)
and strict (same-name analogs removed).
Gate (pre-set): T3_dev all-GT-intermediates-present A3 - A0 >= 10 pp, A3 > A2 and A3 > A1
consensus; same direction on Silver-dev strict.
"""
from __future__ import annotations

import argparse
import json

import common4 as K
import retrieval as RT

MAP = ('TARGET REACTION\nReactants and reagents (SMILES): {reactants}\nProducts (SMILES): {products}\n'
       'Reaction conditions: {conditions}\n\nMechanisms of related reactions:\n\n{analogs}\n\n'
       'Use these related mechanisms as analogies for the TARGET reaction: decide which analog (if any) follows the '
       'same mechanistic course, and map its intermediates onto the target\'s own atoms and substituents. Consider '
       'also a different course if no analog fits. Reason first. Then list the SMILES of every intermediate you '
       'propose for the target, in order of formation, one per line, inside <intermediates> and </intermediates>. '
       'Give explicit formal charges. Do not list the starting materials or the final products.')
RAG = ('## Mechanisms of related reactions (for reference; they may or may not apply)\n\n{analogs}\n\n## Input')
VIS = ('## Candidate intermediates (proposed by analogy to known mechanisms of related reactions; some may be wrong '
       '- use a candidate only if it is chemically justified for this reaction, and add any intermediates that are '
       'missing)\n\n{cands}\n\n## Input')


def sets():
    return {'T3_dev': K.load_fukuyama('T3_dev'), 'J3': K.load_fukuyama('J3'), 'silver_dev': K.silver_dev()}


def modes(ds):
    return ['std', 'strict'] if ds == 'silver_dev' else ['std']


def with_block(prompt, block):
    return prompt.replace('## Input', block, 1)


def job(ds, case):
    names = RT.identify(case, ds)
    for mode in modes(ds):
        analogs = RT.retrieve(case, names, strict=(mode == 'strict'))
        text = RT.render(analogs)
        base = K.official_prompt(case)
        K.call(f'{ds}/A2_{mode}', case['id'], with_block(base, RAG.format(analogs=text)))
        mapped = K.call(f'{ds}/map_{mode}', case['id'], MAP.format(
            reactants='.'.join(case['reactants']), products='.'.join(case['products']),
            conditions=case['conditions'], analogs=text))
        cands = K.tagged(mapped, 'intermediates')
        if cands:
            K.call(f'{ds}/A3_{mode}', case['id'], with_block(base, VIS.format(cands='\n'.join(cands))))
        meta = K.OUT / ds / f'retrieved_{mode}' / f"{case['id']}.json"
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps({'names': names, 'analogs': [a['id'] for a in analogs],
                                    'analog_names': [a['name'] for a in analogs], 'candidates': cands}, indent=1))
    return 'ok'


def _jac(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def consensus_pick(cands):
    """Index of the candidate mechanism with highest mean species-key Jaccard to the others."""
    keys = [{k for s in (m or []) for k in K.species_keys(s.get('intermediate_smiles') or '')} for m in cands]
    return max(range(len(cands)), key=lambda i: sum(_jac(keys[i], keys[j]) for j in range(len(cands)) if j != i))


def analyze(all_sets):
    rep = {}
    for ds, cases in all_sets.items():
        fuku = ds != 'silver_dev'
        main = 'all_present'
        keys = ['all_present', 'in_order_frac', 'parsed'] if fuku else \
            ['S_partial', 'S_total', 'all_present', 'in_order_full', 'parsed']
        for mode in modes(ds):
            arms = {}
            for cid, case in cases.items():
                a0 = K.cached(f'{ds}/base_default', cid)
                smp = [K.cached(f'{ds}/samp{i}', cid) for i in range(1, 5)]
                a2, a3 = K.cached(f'{ds}/A2_{mode}', cid), K.cached(f'{ds}/A3_{mode}', cid)
                if a0 is None or a2 is None or any(s is None for s in smp):
                    continue
                if a3 is None:  # no candidates mapped: A3 falls back to the direct answer
                    a3 = a0
                pm = [K.parse_mech(x) for x in smp]
                sc = [K.score(case, m) for m in pm]
                arms.setdefault('A0', []).append((cid, K.score(case, K.parse_mech(a0))))
                arms.setdefault('A1_oracle4', []).append((cid, max(sc, key=lambda s: s[main])))
                arms.setdefault('A1_consensus4', []).append((cid, sc[consensus_pick(pm)]))
                arms.setdefault('A2', []).append((cid, K.score(case, K.parse_mech(a2))))
                arms.setdefault('A3', []).append((cid, K.score(case, K.parse_mech(a3))))
            block = {arm: K.summarize(rows, keys) for arm, rows in arms.items()}
            if arms:
                for metric in (['all_present'] if fuku else ['S_partial', 'all_present']):
                    for b in ('A0', 'A2', 'A1_consensus4', 'A1_oracle4'):
                        block[f'A3_minus_{b}_{metric}'] = K.paired(arms['A3'], arms[b], metric)
                    block[f'A2_minus_A0_{metric}'] = K.paired(arms['A2'], arms['A0'], metric)
            rep[f'{ds}_{mode}'] = block
    (K.OUT / 'SUMMARY_P1.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=60)
    ap.add_argument('--analyze-only', action='store_true')
    cli = ap.parse_args()
    s = sets()
    if not cli.analyze_only:
        K.run_pool(job, [(ds, c) for ds, cases in s.items() for c in cases.values()], cli.workers)
    analyze(s)
