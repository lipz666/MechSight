"""J4 data: convert oMe-Gold (external/oMeBench, MIT) into cases for the agent.

Two steps, two interpreters (RXNMapper needs its own torch env):
  --map    (external/env-map python)  reactants[+reagent pool]>>product -> atom maps
  --build  (external/venv-llm python) mapped reaction -> labelled initial/target PGL

Rules fixed before any J4 model call:
  * reagent pool: conditions mentioning acid (acid, H+, H3O+, HCl, H2SO4, TsOH,
    TFA, HBr) add H3O+ and H2O; conditions mentioning base (base, OH-, NaOH,
    KOH) add OH- and H2O; otherwise nothing is added.
  * target = principal product(s) only (oMe-Gold omits byproducts); the prompt
    says so. Labels come from the RXNMapper map; endpoints are then normalized
    exactly as in exp1 (explicit hydrogens, endpoint-only H correspondence,
    random relabelling), so no annotation leaks into labels.
  * a reaction is excluded (and counted) if mapping fails, a product heavy atom
    is unmapped, or exp1 normalization rejects it.
Scoring uses only the gold intermediate SMILES; the map never enters scoring.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OME = ROOT.parent / 'external' / 'oMeBench' / 'data' / 'oMe_Gold.json'
OUT = ROOT / 'outputs' / 'agent_v2_J4'
ACID = re.compile(r'acid|\bH\+|H3O\+|\bHCl\b|H2SO4|TsOH|TFA|\bHBr\b', re.I)
BASE = re.compile(r'\bbase\b|OH-|NaOH|KOH', re.I)


def pool(conditions):
    extra = []
    if ACID.search(conditions or ''):
        extra += ['[OH3+]', 'O']
    if BASE.search(conditions or ''):
        extra += ['[OH-]', 'O']
    return list(dict.fromkeys(extra))


def do_map():
    from rxnmapper import RXNMapper
    mapper = RXNMapper()
    gold = json.loads(OME.read_text())
    out = []
    for r in gold:
        reactants = r['reactants_smiles'] + pool(r['conditions'])
        rxn = '.'.join(reactants) + '>>' + '.'.join(r['products_smiles'])
        rec = {'reaction_id': r['reaction_id'], 'rxn': rxn}
        try:
            res = mapper.get_attention_guided_atom_maps([rxn])[0]
            rec.update(mapped=res['mapped_rxn'], confidence=res['confidence'])
        except Exception as exc:
            rec['error'] = f'{type(exc).__name__}: {str(exc)[:200]}'
        out.append(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'mapped.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({'mapped': sum('mapped' in r for r in out), 'failed': sum('error' in r for r in out)}))


def mol_to_state(mol, label_of, role):
    from develop import standalone_chemistry as chem
    from rdkit import Chem
    Chem.Kekulize(mol, clearAromaticFlags=False)
    atomdict, bonds = {}, []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 1:
            continue
        atomdict[label_of[a.GetIdx()]] = chem.Atom(a.GetSymbol(), Hcount=a.GetTotalNumHs(), charge=a.GetFormalCharge())
    kinds = {Chem.BondType.SINGLE: ('single', 1.0), Chem.BondType.DOUBLE: ('double', 2.0),
             Chem.BondType.TRIPLE: ('triple', 3.0), Chem.BondType.AROMATIC: ('aromatic', 1.5)}
    for b in mol.GetBonds():
        x, y = b.GetBeginAtom(), b.GetEndAtom()
        if x.GetAtomicNum() == 1 or y.GetAtomicNum() == 1:
            continue
        kind, order = kinds.get(b.GetBondType(), ('single', 1.0))
        bonds.append(chem.Bond(label_of[x.GetIdx()], label_of[y.GetIdx()], kind, order=order))
    sub = chem.Substructure('', atomdict, bonds)
    return chem.ReactionState([chem.Species(role, 'stable', chem.Molecule('', [sub]))])


def do_build():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'experiments' / 'exp1_core_comparison'))
    from rdkit import Chem, RDLogger
    from prepare import normalize_endpoints
    RDLogger.DisableLog('rdApp.*')
    gold = {r['reaction_id']: r for r in json.loads(OME.read_text())}
    mapped = json.loads((OUT / 'mapped.json').read_text())
    cases, excluded = [], {}
    for rec in mapped:
        rid = rec['reaction_id']
        if 'mapped' not in rec:
            excluded[rid] = 'mapping_failed'
            continue
        left, right = rec['mapped'].split('>>')
        rmol, pmol = Chem.MolFromSmiles(left), Chem.MolFromSmiles(right)
        if rmol is None or pmol is None:
            excluded[rid] = 'mapped_smiles_unparseable'
            continue
        rlabel, by_map = {}, {}
        for a in rmol.GetAtoms():
            rlabel[a.GetIdx()] = f'{a.GetSymbol()}{a.GetIdx() + 1}'
            if a.GetAtomMapNum():
                by_map[a.GetAtomMapNum()] = rlabel[a.GetIdx()]
        plabel = {}
        unmapped = False
        for a in pmol.GetAtoms():
            if a.GetAtomMapNum() in by_map:
                plabel[a.GetIdx()] = by_map[a.GetAtomMapNum()]
            else:
                unmapped = True
        if unmapped:
            excluded[rid] = 'product_atom_unmapped'
            continue
        try:
            initial = mol_to_state(rmol, rlabel, 'reactant')
            target = mol_to_state(pmol, plabel, 'product')
            ini, tgt, _ = normalize_endpoints(initial, target, 20261030 + len(cases))
        except Exception as exc:
            excluded[rid] = 'normalize:' + str(exc)[:60]
            continue
        g = gold[rid]
        cases.append({'case_id': rid, 'level': g['level'], 'name': g['name'], 'conditions': g['conditions'],
                      'initial_state_pgl': repr(ini), 'target_state_pgl': repr(tgt),
                      'products_smiles': g['products_smiles'], 'map_confidence': rec['confidence'],
                      'gold_steps': [{'type': s['type'], 'smiles': s['intermediate_smiles'], 'weight': s.get('step_weight', 1.0)}
                                     for s in g['mechanism']]})
    (OUT / 'cases.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in cases))
    reasons = {}
    for v in excluded.values():
        reasons[v] = reasons.get(v, 0) + 1
    summary = {'source_reactions': len(gold), 'usable_cases': len(cases), 'excluded': reasons,
               'levels': {lv: sum(c['level'] == lv for c in cases) for lv in ('easy', 'medium', 'hard')}}
    (OUT / 'build_summary.json').write_text(json.dumps({**summary, 'excluded_ids': excluded}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', action='store_true')
    ap.add_argument('--build', action='store_true')
    cli = ap.parse_args()
    if cli.map:
        do_map()
    if cli.build:
        do_build()
