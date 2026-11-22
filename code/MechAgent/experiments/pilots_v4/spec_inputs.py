"""Inputs for the specialised models (ChRIMP/MechSMILES T5, FlowER) on the shared benchmarks.

Each case: id, bench, reactants (list), main product (largest new species), gold intermediates
(oMe-Gold: official list; T3: GT species sequence; uspto31k: states built on the 4060 host).
Writes outputs/pilots_v4_20261103/spec_inputs.json (uspto31k is merged from spec_bench.json).
"""
from __future__ import annotations

import json
import sys

from rdkit import Chem

import common4 as K


def main_product(products, reactants):
    start = {Chem.MolToSmiles(m) for s in reactants if (m := Chem.MolFromSmiles(s))}
    new = [s for s in products if (m := Chem.MolFromSmiles(s)) and Chem.MolToSmiles(m) not in start]
    return max(new or products, key=lambda s: Chem.MolFromSmiles(s).GetNumHeavyAtoms() if Chem.MolFromSmiles(s) else 0)


def build(uspto_path=None):
    out = []
    for c in K.load_gold().values():
        out.append({'id': c['id'], 'bench': 'gold', 'reactants': c['reactants'],
                    'product': main_product(c['products'], c['reactants'])})
    for which in ('T3_dev', 'T3_confirm'):
        for c in K.load_fukuyama(which).values():
            out.append({'id': c['id'], 'bench': which, 'reactants': c['reactants'],
                        'product': main_product(c['products'], c['reactants'])})
    if uspto_path:
        for c in json.loads(open(uspto_path).read()):
            out.append({'id': c['id'], 'bench': 'uspto31k', 'reactants': c['reactants'], 'product': c['products'][0],
                        'states': c['states']})
    (K.OUT / 'spec_inputs.json').write_text(json.dumps(out, indent=1))
    print(len(out))


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else None)
