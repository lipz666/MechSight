"""Retrieve analogous solved mechanisms for the cond_analog arm (pilot A+B).

Memory: the 250 FukuyamaBench cases outside clean.jsonl (same textbook, none of
the 60 evaluation cases). Each test case is matched using only what the agent
is given: its endpoint net-change signature (P1 net_signature) and its
condition text. GT of the test case is never read.

Leakage control: a memory case is excluded for a test case when the Morgan
Tanimoto similarity of either endpoint (reactants or products, heavy atoms)
is >= --max-similarity. The similarity of every retrieved analog is recorded
so results can be stratified by it.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'p1_template_baseline'))
from develop import standalone_chemistry as chem
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

import run as P1

RDLogger.DisableLog('rdApp.*')
OUT = ROOT / 'outputs' / 'p23_planning_probe_20261024'
KINDS = {1.0: Chem.BondType.SINGLE, 2.0: Chem.BondType.DOUBLE, 3.0: Chem.BondType.TRIPLE, 1.5: Chem.BondType.AROMATIC}
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def to_mol(state):
    atoms = dict(chem.pgl.iter_atoms(state))
    names = sorted(atoms)
    index = {n: i for i, n in enumerate(names)}
    rw = Chem.RWMol()
    for n in names:
        a = Chem.Atom(atoms[n].element)
        a.SetFormalCharge(int(atoms[n].charge or 0))
        a.SetNumExplicitHs(int(atoms[n].Hcount or 0))
        a.SetNoImplicit(True)
        rw.AddAtom(a)
    for (x, y), bond in chem.pgl.bond_map(state).items():
        rw.AddBond(index[x], index[y], KINDS.get(float(bond.order or 1.0), Chem.BondType.SINGLE))
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
    except Exception:
        mol.UpdatePropertyCache(strict=False)
        mol = Chem.RemoveHs(mol, sanitize=False)
    return mol


def smiles(state):
    try:
        return Chem.MolToSmiles(to_mol(state))
    except Exception:
        return '(unrenderable)'


def fingerprint(state):
    mol = to_mol(state)
    try:
        return FPGEN.GetFingerprint(mol)
    except Exception:
        return None


def condition_tokens(conditions):
    tokens = set()
    for item in conditions:
        if item['role'] == 'temperature':
            continue
        for tok in re.split(r'[\s,;/]+', item['text'].lower()):
            tok = tok.strip('()')
            if tok and not re.fullmatch(r'[\d.\-–to°cmhg%eq]+', tok) and tok not in {'aq', 'cat', 'then', 'rt'}:
                tokens.add(tok)
    return tokens


def cosine(a, b):
    a, b = dict(a), dict(b)
    dot = sum(a[k] * b.get(k, 0) for k in a)
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


def describe(case, max_steps=10):
    """Render a solved analog with its own labels: conditions, endpoints, every
    step's bond changes with the intermediate SMILES, and its transient bonds."""
    states = [chem.parse_pgl_literal(s) for s in case['states_pgl']]
    seen = collections.defaultdict(set)
    lines = [f"Conditions: {'; '.join(t['text'] for t in case['conditions'] if t['text'])}",
             f"Overall: {smiles(states[0])} >> {smiles(states[-1])}"]
    for k, step in enumerate(case['gt_step_changes'][:max_steps]):
        parts = []
        for a, b, kind in step['bonds_formed']:
            parts.append(f'form {a}-{b} ({kind})')
            seen[frozenset((a, b))].add('form')
        for a, b, kind in step['bonds_broken']:
            parts.append(f'break {a}-{b}')
            seen[frozenset((a, b))].add('break')
        for a, b, x, y in step['bond_order_changes']:
            parts.append(f'{a}-{b} {x}->{y}')
        for n, x, y in step['charge_changes']:
            parts.append(f'charge {n} {x:+d}->{y:+d}')
        lines.append(f"  step {k + 1}: {'; '.join(parts)}  => {smiles(states[k + 1])}")
    transient = sorted('-'.join(sorted(p)) for p, kinds in seen.items() if kinds == {'form', 'break'})
    lines.append('  transient bonds (formed then broken): ' + (', '.join(transient) if transient else 'none'))
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--max-similarity', type=float, default=0.7)
    args = ap.parse_args()
    clean = {c['case_id']: c for c in map(json.loads, (ROOT / 'outputs/fukuyama/clean.jsonl').read_text().splitlines())}
    pool = [c for c in map(json.loads, (ROOT / 'outputs/fukuyama/all.jsonl').read_text().splitlines()) if c['case_id'] not in clean]
    memory = []
    for c in pool:
        try:
            ini, tgt = chem.parse_pgl_literal(c['initial_state_pgl']), chem.parse_pgl_literal(c['target_state_pgl'])
            memory.append({'case': c, 'sig': P1.net_signature(ini, tgt), 'cond': condition_tokens(c['conditions']),
                           'fp': (fingerprint(ini), fingerprint(tgt))})
        except Exception:
            continue
    ids = json.loads((OUT / 'cases.json').read_text())
    result, report = {}, []
    for cid in ids:
        case = clean[cid]
        ini, tgt = chem.parse_pgl_literal(case['initial_state_pgl']), chem.parse_pgl_literal(case['target_state_pgl'])
        sig, cond, fps = P1.net_signature(ini, tgt), condition_tokens(case['conditions']), (fingerprint(ini), fingerprint(tgt))
        scored, excluded = [], 0
        for m in memory:
            sims = [DataStructs.TanimotoSimilarity(a, b) for a, b in zip(fps, m['fp']) if a is not None and b is not None]
            similarity = max(sims) if sims else 0.0
            if similarity >= args.max_similarity:
                excluded += 1
                continue
            jac = len(cond & m['cond']) / len(cond | m['cond']) if cond | m['cond'] else 0.0
            scored.append((0.5 * jac + 0.5 * cosine(sig, m['sig']), jac, cosine(sig, m['sig']), similarity, m['case']))
        scored.sort(key=lambda x: (-x[0], x[4]['case_id']))
        top = scored[:args.k]
        result[cid] = {'analogs': [{'case_id': c['case_id'], 'score': round(s, 3), 'condition_jaccard': round(j, 3),
                                    'signature_cosine': round(g, 3), 'max_tanimoto': round(t, 3), 'text': describe(c)}
                                   for s, j, g, t, c in top],
                       'excluded_near_duplicates': excluded}
        report.append((cid, excluded, [(a['case_id'], a['score'], a['condition_jaccard'], a['signature_cosine'], a['max_tanimoto'])
                                       for a in result[cid]['analogs']]))
    (OUT / 'analogs.json').write_text(json.dumps({'config': vars(args), 'memory_size': len(memory), 'cases': result},
                                                 ensure_ascii=False, indent=2))
    print('memory', len(memory))
    for row in report:
        print(row)


if __name__ == '__main__':
    main()
