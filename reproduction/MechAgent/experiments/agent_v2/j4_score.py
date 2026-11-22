"""J4 species-level scoring against oMe-Gold intermediate SMILES (no atom maps).

Molecule identity = standard InChI with the /FixedH layer: insensitive to how a
resonance structure is drawn, sensitive to where a proton sits (a tautomer or
a protonation is a mechanistic event). Falls back to canonical SMILES when no
InChI can be made.

Per case:
  in_order_fraction  share of gold steps whose molecules (all of them) appear,
                     in order, among the predicted states (greedy, monotone;
                     a finer predicted path may visit extra states)
  in_order_full      every scorable gold step hit in order
  product_reached    every product molecule present in the final state
  oMeS (type-agnostic) official oMeBench alignment with step types neutralised
                     (our paths carry no step-type labels)
Gold steps whose SMILES RDKit cannot parse are not scorable and are dropped
from the denominator for every arm alike.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C
from common import ROOT, chem, fm

sys.path.insert(0, str(ROOT.parent / 'external' / 'oMeBench' / 'scripts'))
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
try:
    from utils_eval import oMeS
except Exception:  # pandas missing etc.
    oMeS = None


def key_of_mol(mol):
    try:
        inchi = Chem.MolToInchi(mol, options='/FixedH')
        if inchi:
            return inchi
    except Exception:
        pass
    return 'smi:' + Chem.MolToSmiles(mol)


def keys_of_smiles(smiles):
    out = set()
    for part in smiles.split('.'):
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            return None
        out.add(key_of_mol(mol))
    return out


def state_species(state):
    smis = [C._molecule_smiles(state, comp) for comp in C._components(state)]
    keys = set()
    for s in smis:
        mol = Chem.MolFromSmiles(s)
        keys.add(key_of_mol(mol) if mol is not None else 'smi:' + s)
    return keys, smis


def main_species(before, after):
    _, species = C.step_text(before, after)
    parts = [s for s in species.split(', ') if s]
    if not parts:
        return ''
    def heavy(s):
        m = Chem.MolFromSmiles(s)
        return m.GetNumHeavyAtoms() if m is not None else 0
    return max(parts, key=heavy)


def score_case(case, actions):
    initial = chem.parse_pgl_literal(case['initial_state_pgl'])
    states, executed = fm.replay(case['initial_state_pgl'], actions)
    pred_keys = [state_species(s)[0] for s in states]
    gold = [(g, keys_of_smiles(g['smiles'])) for g in case['gold_steps']]
    scorable = [(g, k) for g, k in gold if k]
    hit, cursor = 0, 0
    for g, keys in scorable:
        found = next((i for i in range(cursor, len(pred_keys)) if keys <= pred_keys[i]), None)
        if found is None:
            continue  # keep scanning later gold steps (fraction, not prefix)
        hit, cursor = hit + 1, found + 1
    product_keys = set()
    for p in case['products_smiles']:
        k = keys_of_smiles(p)
        if k:
            product_keys |= k
    reached = bool(states) and bool(product_keys) and product_keys <= pred_keys[-1]
    result = {'executed_steps': executed, 'proposed_steps': len(actions), 'scorable_gold_steps': len(scorable),
              'in_order_hits': hit, 'in_order_fraction': hit / len(scorable) if scorable else 0.0,
              'in_order_full': bool(scorable) and hit == len(scorable), 'product_reached': reached}
    if oMeS is not None:
        pred_steps = [('x', main_species(b, a)) for b, a in zip([initial] + states[:-1], states)]
        gold_steps = [('x', g['smiles'], float(g.get('weight') or 1.0)) for g in case['gold_steps']]
        r = oMeS(gold_steps, pred_steps)
        total = sum(w for _, _, w in gold_steps) or 1.0
        result.update(oMeS_total=r.S_total / total, oMeS_partial=r.S_partial / total)
    return result
