"""End-to-end check of the FlowER driver + scorer on FlowER's own test data.

Takes sequences from FlowER's test set (in-distribution, where FlowER reports
~89% top-1 pathway accuracy), converts them into the same case format as the
FukuyamaBench cases (initial/target/states PGL + per-step bond changes), and
runs them through exactly the driver and scoring used for the head-to-head.
If FlowER solved none of these either, the 0% on FukuyamaBench would point at
the harness, not at FlowER.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import flower_driver as D  # sets FlowER env and paths

import networkx as nx
import torch
from rdkit import Chem

from develop import standalone_chemistry as chem

DATA = D.FLOWER / 'data' / 'flower_new_dataset' / 'test.txt'


def state_from_mapped(smiles):
    mol = Chem.MolFromSmiles(smiles, D.BP.ps)
    mapping = {a.GetAtomMapNum(): f'{a.GetSymbol()}{a.GetAtomMapNum()}' for a in mol.GetAtoms()}
    return D.mapped_smiles_to_state(smiles, mapping)


def case_from_sequence(idx, steps):
    states = [steps[0].split('>>')[0]]
    for s in steps:
        tgt = s.split('>>')[1]
        if tgt != states[-1]:
            states.append(tgt)
    pgl = [state_from_mapped(s) for s in states]
    changes = []
    for before, after in zip(pgl, pgl[1:]):
        b = {frozenset(k): v for k, v in chem.pgl.bond_map(before).items()}
        a = {frozenset(k): v for k, v in chem.pgl.bond_map(after).items()}
        changes.append({'bonds_formed': [list(sorted(k)) + ['single'] for k in a.keys() - b.keys()],
                        'bonds_broken': [list(sorted(k)) + ['single'] for k in b.keys() - a.keys()],
                        'bond_order_changes': [], 'charge_changes': []})
    return {'case_id': f'flower_test_{idx}', 'partition': 'F', 'initial_state_pgl': repr(pgl[0]),
            'target_state_pgl': repr(pgl[-1]), 'states_pgl': [repr(s) for s in pgl], 'gt_step_changes': changes}


def main():
    import signal
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 900
    out = D.ROOT / 'outputs' / 'flower_compare_v2_20261031' / 'sanity'
    out.mkdir(parents=True, exist_ok=True)
    seqs = {}
    with open(DATA) as f:
        for i, line in enumerate(f):
            if i > 40000:
                break
            rxn, idx = line.rstrip('\n').rsplit('|', 1)
            seqs.setdefault(idx, []).append(rxn)

    def n_atoms(idx):
        return Chem.MolFromSmiles(seqs[idx][0].split('>>')[0], D.BP.ps).GetNumAtoms()
    rng = random.Random(7)
    ids = [k for k in seqs if 3 <= len(seqs[k]) <= 6]
    rng.shuffle(ids)
    ids = [k for k in ids if n_atoms(k) <= 40][:n]
    torch.set_num_threads(6)
    args = D.Args
    args.local_rank, args.device = -1, torch.device('cpu')
    state = torch.load(os.path.join(args.model_path, args.model_name), weights_only=False, map_location='cpu')
    pre = state['args']
    pre.load_from, pre.device, pre.local_rank = None, args.device, -1
    model, flow, _ = D.init_model(pre)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['state_dict'].items()})
    model.eval()
    for idx in ids:
        dest = out / f'{idx}.json'
        if dest.exists():
            continue
        t0 = time.time()
        case = case_from_sequence(idx, seqs[idx])
        smiles, mapping = D.pgl_to_mapped_smiles(chem.parse_pgl_literal(case['initial_state_pgl']))
        graph = nx.DiGraph()
        graph.add_node(smiles, depth=1)
        def alarm(signum, frame):
            raise TimeoutError
        signal.signal(signal.SIGALRM, alarm)
        signal.alarm(cap)
        timed_out = False
        try:
            D.BP.beam_search(args, model, flow, {0: [smiles]}, [(graph, smiles, (smiles, []))])
        except TimeoutError:
            timed_out = True
        finally:
            signal.alarm(0)
        res = D.score_graph(case, graph, smiles, mapping, sample_size=args.sample_size)
        rec = {'seq': idx, 'gold_steps': len(case['gt_step_changes']), 'atoms': n_atoms(idx), 'timed_out': timed_out,
               **{k: res[k] for k in ('graph_nodes', 'n_paths', 'n_finished_paths', 'any_reaches_main_product',
                                      'any_sigma_L1', 'any_in_order')},
               'top1_sigma_L1': bool(res['top1'] and res['top1']['sigma_L1']),
               'target_sigma_L1': bool(res['target'] and res['target']['sigma_L1']),
               'target_in_order': bool(res['target'] and res['target']['intermediates_in_order']),
               'seconds': round(time.time() - t0)}
        dest.write_text(json.dumps({**rec, 'graph': D.graph_to_json(graph, smiles, mapping)}))
        print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
