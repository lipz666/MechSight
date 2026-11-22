"""P5: contamination probes for the benchmarks used in the paper.

(a) oMe-Gold product prediction, original vs minimal homologue (one CH3 >= 3 bonds from the
    atom-mapped reaction centre; ome_homologs.py). From reactants + conditions only; correct =
    the main product's sigma-graph key is among the predicted products. A memorised benchmark
    should be clearly better on the originals.
(b) J3 mechanism level: official default prompt on the homologue (ks.analog_case, exact PGL
    ground truth) vs the original (p6 J3/base_default); all GT intermediates present and heavy
    hidden events recalled.
Criterion (as for the T3 probe): no memorisation sign if the original - homologue CI lies within
+-10 pp.
"""
from __future__ import annotations

import argparse
import json
import sys

import common4 as K

sys.path.insert(0, str(K.HERE.parent / 'inert_knowledge'))
import ks  # noqa: E402

TASK = ('Reactants and reagents (SMILES): {reactants}\nReaction conditions: {conditions}\n\n'
        'Predict the major organic product(s) of this reaction. Reason briefly, then give the product SMILES, one '
        'per line, inside <products> and </products>.')


def homologs():
    return {h['reaction_id']: h for h in json.loads((K.OUT / 'ome_homologs.json').read_text())}


def j3_homologs():
    out = {}
    for cid, case in K.load_fukuyama('J3').items():
        new = ks.analog_case(case['pgl'])
        if new:
            out[cid] = K._fuku_case(new, 'J3')
    return out


def job_a(h, which):
    rs = h['reactants'] if which == 'hom' else h['orig_reactants']
    p = TASK.format(reactants='.'.join(rs), conditions=h['conditions'])
    return 'ok' if K.call(f'gold/p5_{which}', h['reaction_id'], p) else 'error'


def job_b(case):
    return 'ok' if K.call('J3/p5_hom_mech', case['id'], K.official_prompt(case)) else 'error'


def main_product_key(products, reactants):
    start = {k for s in reactants for k in (K.j4_score.keys_of_smiles(s) or set())}
    cands = []
    for s in products:
        ks_ = K.j4_score.keys_of_smiles(s) or set()
        from rdkit import Chem
        mol = Chem.MolFromSmiles(s)
        if mol is not None and ks_ and not ks_ <= start:
            cands.append((mol.GetNumHeavyAtoms(), ks_))
    return max(cands, key=lambda x: x[0])[1] if cands else None


def correct(ans, products, reactants):
    key = main_product_key(products, reactants)
    if key is None or ans is None:
        return None
    pred = {k for s in K.tagged(ans, 'products') for k in (K.j4_score.keys_of_smiles(s) or set())}
    return float(key <= pred)


def analyze():
    rows = []
    for rid, h in homologs().items():
        a, b = K.cached('gold/p5_orig', rid), K.cached('gold/p5_hom', rid)
        ca, cb = correct(a, h['orig_products'], h['orig_reactants']), correct(b, h['products'], h['reactants'])
        if ca is not None and cb is not None:
            rows.append((rid, ca, cb))
    ids = [r for r, _, _ in rows]
    rep = {'a_oMe_product_prediction': {
        'pairs': len(rows), 'original': K.phase2.ci([x for _, x, _ in rows], ids),
        'homologue': K.phase2.ci([y for _, _, y in rows], ids),
        'original_minus_homologue': K.phase2.ci([x - y for _, x, y in rows], ids),
        'only_original / only_homologue': [sum(x > y for _, x, y in rows), sum(y > x for _, x, y in rows)]}}
    orig = K.load_fukuyama('J3')
    ra, rb = [], []
    for cid, hc in j3_homologs().items():
        a, b = K.cached('J3/base_default', cid), K.cached('J3/p5_hom_mech', cid)
        if a and b:
            ra.append((cid, K.score(orig[cid], K.parse_mech(a))))
            rb.append((cid, K.score(hc, K.parse_mech(b))))
    if ra:
        rep['b_J3_mechanism'] = {'original': K.summarize(ra, ['all_present']), 'homologue': K.summarize(rb, ['all_present']),
                                 'original_minus_homologue_all_present': K.paired(ra, rb, 'all_present')}
    (K.OUT / 'SUMMARY_P5.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--analyze-only', action='store_true')
    cli = ap.parse_args()
    if not cli.analyze_only:
        jobs_a = [(h, w) for h in homologs().values() for w in ('orig', 'hom')]
        K.run_pool(job_a, jobs_a, cli.workers)
        K.run_pool(job_b, [(c,) for c in j3_homologs().values()], cli.workers)
    analyze()
