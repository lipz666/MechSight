"""Freeze an endpoint-only, sample-held-out MechFinder comparison dataset."""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import random
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from develop import standalone_chemistry as chem
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
SEED = 20261018


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_key(text):
    """Remove atom mapping and component order, retaining substrate structure."""
    parts = []
    for side in text.split('>'):
        mol = Chem.MolFromSmiles(side) if side else None
        if mol is None:
            parts.append(side)
            continue
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        parts.append(Chem.MolToSmiles(mol))
    return '>'.join(parts)


def endpoint_info(state):
    """Infer all hydrogen counts from this endpoint alone, never the GT path."""
    atoms = dict(chem.pgl.iter_atoms(state))
    bonds = chem.pgl.bond_map(state)
    labels = sorted(atoms)
    indices = {name: i for i, name in enumerate(labels)}
    rw = Chem.RWMol()
    for name in labels:
        a = atoms[name]
        rd = Chem.Atom(a.element)
        rd.SetFormalCharge(int(a.charge or 0))
        if a.Hcount:
            rd.SetNumExplicitHs(int(a.Hcount))
            rd.SetNoImplicit(True)
        rw.AddAtom(rd)
    kinds = {1.: Chem.BondType.SINGLE, 2.: Chem.BondType.DOUBLE,
             3.: Chem.BondType.TRIPLE, 1.5: Chem.BondType.AROMATIC}
    for (a, b), bond in bonds.items():
        order = float(bond.order or 0)
        if order not in kinds:
            raise ValueError('unsupported_fractional_bond')
        rw.AddBond(indices[a], indices[b], kinds[order])
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    h = collections.Counter()
    heavy = {}
    heavy_bonds = []
    free_h = []
    for name in labels:
        atom = atoms[name]
        rd = mol.GetAtomWithIdx(indices[name])
        if atom.element == 'H':
            neighbours = [labels[n.GetIdx()] for n in rd.GetNeighbors()]
            if len(neighbours) == 1 and atoms[neighbours[0]].element != 'H':
                h[neighbours[0]] += 1
            elif not neighbours:
                free_h.append(int(atom.charge or 0))
            else:
                raise ValueError('unsupported_hydrogen_component')
        else:
            heavy[name] = copy.deepcopy(atom)
            h[name] += int(rd.GetNumImplicitHs() + rd.GetNumExplicitHs())
    for (a, b), bond in bonds.items():
        if a in heavy and b in heavy:
            heavy_bonds.append(copy.deepcopy(bond))
    return heavy, heavy_bonds, h, sorted(free_h)


def normalize_endpoints(initial, target, seed):
    """Uniform explicit H, neutral labels, and endpoint-only H correspondence."""
    ih, ib, ic, ifree = endpoint_info(initial)
    th, tb, tc, tfree = endpoint_info(target)
    if not set(th).issubset(ih):
        raise ValueError('target_has_missing_heavy_atoms')
    total_i, total_t = sum(ic.values()) + len(ifree), sum(tc.values()) + len(tfree)
    if total_t > total_i:
        raise ValueError('target_requires_unprovided_hydrogen')
    if len(ih) > 45 or len(ih) + total_i > 110:
        raise ValueError('outside_predeclared_size_limit')
    rng = random.Random(seed)
    ordering = sorted(ih)
    rng.shuffle(ordering)
    remap = {old: f'{ih[old].element}{i}' for i, old in enumerate(ordering, 1)}
    ip = {old: [] for old in sorted(ih)}
    label = 1
    for old in sorted(ih):
        for _ in range(ic[old]):
            ip[old].append(f'H{label}')
            label += 1
    ifp = [f'H{i}' for i in range(label, label + len(ifree))]
    tp, available = {}, []
    for old in sorted(ih):
        keep = min(len(ip[old]), tc[old]) if old in th else 0
        tp[old] = ip[old][:keep]
        available.extend(ip[old][keep:])
    available.extend(ifp)
    for old in sorted(th):
        missing = tc[old] - len(tp[old])
        tp[old].extend(available[:missing])
        available = available[missing:]
    tfp = available[:len(tfree)]

    def render(heavy, bonds, owners, free, free_charges, role):
        atomdict = {remap[k]: chem.Atom(v.element, charge=v.charge) for k, v in heavy.items()}
        bl = [chem.Bond(remap[b.atom1], remap[b.atom2], b.kind, b.order) for b in bonds]
        for old in sorted(heavy):
            for name in owners[old]:
                atomdict[name] = chem.Atom('H')
                bl.append(chem.Bond(remap[old], name, 'single'))
        for name, charge in zip(free, free_charges):
            atomdict[name] = chem.Atom('H', charge=charge)
        sub = chem.Substructure('', dict(sorted(atomdict.items())), bl)
        return chem.ReactionState([chem.Species(role, 'stable', chem.Molecule('', [sub]))])

    return (render(ih, ib, ip, ifp, ifree, 'reactant'),
            render(th, tb, tp, tfp, tfree, 'product'), remap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    if (out / 'manifest.json').exists():
        raise SystemExit('Dataset already frozen; refusing to overwrite.')
    out.mkdir(parents=True, exist_ok=True)
    excluded = set()
    rx = re.compile(rb'mechfinder::[A-Za-z0-9_()/-]+::\d+')
    scanned = []
    for root in [ROOT / 'outputs', ROOT / 'doc', ROOT / 'develop-2', ROOT.parent / 'MechAgent-rmr' / 'outputs']:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or out in p.parents or p.suffix not in {'.json', '.jsonl', '.md', '.py'}:
                continue
            if '__pycache__' in p.parts:
                continue
            data = p.read_bytes()
            matches = rx.findall(data)
            if matches:
                excluded.update(v.decode() for v in matches)
                scanned.append(str(p.relative_to(ROOT.parent)))
    corpus = ROOT.parent / 'core/mechfinder_merged_pgl_paths.core.jsonl'
    used_keys = set()
    counts = collections.Counter()
    # First pass identifies every source reaction corresponding to prior IDs.
    with corpus.open() as f:
        for line in f:
            row = json.loads(line)
            rid = f"mechfinder::{row['reaction_class']}::{row['source_row_index']}"
            if rid in excluded:
                used_keys.add(source_key(row['source_reaction_smiles']))
    buckets = collections.defaultdict(list)
    with corpus.open() as f:
        for line in f:
            row = json.loads(line)
            rid = f"mechfinder::{row['reaction_class']}::{row['source_row_index']}"
            counts['corpus_rows'] += 1
            if rid in excluded:
                counts['previously_used_id'] += 1
                continue
            if not 2 <= int(row['merged_step_count']) <= 10:
                counts['outside_2_to_10_step_scope'] += 1
                continue
            score = hashlib.sha256(f'{SEED}:{rid}'.encode()).hexdigest()
            buckets[row['reaction_class']].append((score, row))
    candidates = {}
    for cls, rows in sorted(buckets.items()):
        candidates[cls] = []
        for _, row in sorted(rows, key=lambda item: item[0]):
            key = source_key(row['source_reaction_smiles'])
            if key in used_keys:
                counts['duplicate_reaction_structure'] += 1
                continue
            try:
                path = chem.parse_pgl_literal(row['mechanism_graph'])
                states = [s for s in path.sequence if isinstance(s, chem.ReactionState)]
                initial, target, mapping = normalize_endpoints(states[0], states[-1], SEED + int(row['source_row_index']))
                # No newly invented heavy atoms can be executed in this task.
                all_heavy = {n for s in states for n, a in chem.pgl.iter_atoms(s) if a.element != 'H'}
                if not all_heavy.issubset(mapping):
                    raise ValueError('gt_introduces_unprovided_heavy_atom')
            except Exception as e:
                counts['ineligible:' + str(e).split('\n')[0][:90]] += 1
                continue
            row['_prepared'] = (initial, target, mapping, states)
            candidates[cls].append(row)
            used_keys.add(key)
            if len(candidates[cls]) == 2:
                break
    classes = [c for c in candidates if len(candidates[c]) == 2]
    classes.sort(key=lambda c: hashlib.sha256(f'{SEED}:class:{c}'.encode()).hexdigest())
    if len(classes) < 40:
        raise SystemExit(f'Only {len(classes)} eligible classes; do not silently reduce sample size.')
    inputs, gold = [], []
    for cls in classes[:40]:
        for row in candidates[cls]:
            initial, target, mapping, states = row.pop('_prepared')
            blind = f'exp1::{len(inputs) + 1:03d}'
            source_id = f"mechfinder::{cls}::{row['source_row_index']}"
            inputs.append({'case_id': blind, 'initial_state_pgl': repr(initial), 'target_state_pgl': repr(target)})
            gold.append({'case_id': blind, 'source_reaction_id': source_id, 'reaction_class': cls,
                         'source_row_index': row['source_row_index'], 'source_reaction_smiles': row['source_reaction_smiles'],
                         'label_map': mapping, 'gt_actions_pgl':
                         [repr(a) for a in chem.parse_pgl_literal(row['mechanism_graph']).sequence if isinstance(a, chem.ReactionTransformation)],
                         'gt_states_pgl': [repr(s) for s in states], 'merged_step_count': row['merged_step_count']})
    for name, rows in [('inputs.jsonl', inputs), ('gold.jsonl', gold)]:
        (out / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    snapshot = out / 'snapshot'
    for dirname, targetname in [('develop-2', 'exp1_agent'), ('develop', 'develop')]:
        source = ROOT / dirname
        dest = snapshot / targetname
        dest.mkdir(parents=True, exist_ok=True)
        for p in source.glob('*.py'):
            shutil.copy2(p, dest / p.name)
        if dirname == 'develop-2':
            shutil.copytree(source / 'planner_skills', dest / 'planner_skills', ignore=shutil.ignore_patterns('__pycache__'))
    hashes = {str(p.relative_to(out)): digest(p) for p in snapshot.rglob('*') if p.is_file()}
    manifest = {'seed': SEED, 'cases': 80, 'classes': 40, 'per_class': 2, 'repeats': 2,
                'arms': ['direct', 'agent_no_skills', 'agent_skills79'], 'corpus_sha256': digest(corpus),
                'inputs_sha256': digest(out / 'inputs.jsonl'), 'gold_sha256': digest(out / 'gold.jsonl'),
                'excluded_prior_ids': sorted(excluded), 'exclusion_scan_files': scanned,
                'selection_counts': dict(counts), 'snapshot_sha256': hashes,
                'claim_scope': 'Sample-held-out within known mechanism families; not unseen-family or pretraining-contamination-free.',
                'hydrogen_policy': 'Every endpoint hydrogen explicit; counts inferred independently from each endpoint. Target H correspondence uses endpoints only. All labels blinded. No GT intermediate supplied at inference.'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({k: manifest[k] for k in ['cases', 'classes', 'repeats', 'selection_counts', 'claim_scope']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
