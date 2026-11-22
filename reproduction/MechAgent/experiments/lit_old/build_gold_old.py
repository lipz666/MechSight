"""Gold standard for the old exposure strata: the unchanged p29 argon-build pipeline (PDF incl. schemes,
role definitions, consistency checks, one repair round), pointed at lit_old/papers.

Outputs: OUT/lit_old_argon/{extract2,repair2,gold2}/<PMCID>.json (screen calls live in OUT/lit_old_argon/screen).
python build_gold_old.py --workers 100            (all 302 papers of strata_old.json)
python build_gold_old.py --summary
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'pilots_v4'))
import common4 as K  # noqa: E402
import p29_gold_rebuild as P29  # noqa: E402

P29.PAPERS = HERE / 'papers'
P29.OUTD = K.OUT / 'lit_old_argon'
STRATA = json.loads((HERE / 'strata_old.json').read_text())['papers']


ORG = set('C N O S P F Cl Br I B Si Se'.split())


def organic_size(smis):
    """Heavy atoms of the largest all-organic (C-containing, no metal) product fragment."""
    from rdkit import Chem
    best = 0
    for s in smis or []:
        m = Chem.MolFromSmiles(s or '')
        for fr in (Chem.GetMolFrags(m, asMols=True) if m else []):
            syms = {a.GetSymbol() for a in fr.GetAtoms()}
            if 'C' in syms and syms <= ORG | {'H'}:
                best = max(best, fr.GetNumHeavyAtoms())
    return best


def summary():
    """Scope rule (fixed 2026-11-06, before any model output on these papers): an organic transformation, i.e. the
    largest all-organic product fragment has >= 6 heavy atoms (drops electro/photo-catalytic small-molecule
    conversions, water oxidation, gas-phase ion and cluster chemistry). Writes gold_scope_old.json."""
    by = Counter()
    nint = Counter()
    scope = []
    for p in STRATA:
        f = P29.OUTD / 'gold2' / f'{p["pmcid"]}.json'
        if not f.exists():
            by[(p['exposure'], 'missing')] += 1
            continue
        g = json.loads(f.read_text())
        ok = g['reaction_ok'] and g['gold_intermediates']
        by[(p['exposure'], 'usable' if ok else 'unusable')] += 1
        if ok and organic_size(g['record']['model_reaction'].get('products')) >= 6:
            by[(p['exposure'], 'in_scope')] += 1
            nint[p['exposure']] += len(g['gold_intermediates'])
            scope.append(dict(p, n_gold=len(g['gold_intermediates'])))
    (HERE / 'gold_scope_old.json').write_text(json.dumps(scope, indent=1, ensure_ascii=False))
    out = {b: {k: by[(b, k)] for k in ('usable', 'unusable', 'missing', 'in_scope')} | {'gold_species_in_scope': nint[b]}
           for b in ('low', 'mid', 'high')}
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=100)
    ap.add_argument('--summary', action='store_true')
    cli = ap.parse_args()
    if not cli.summary:
        K.run_pool(P29.job, [(p['pmcid'],) for p in STRATA], cli.workers)
    summary()
