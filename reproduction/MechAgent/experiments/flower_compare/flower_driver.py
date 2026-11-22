"""Stage 1a: run FlowER (Nature 2025, newest public checkpoint) on FukuyamaBench.

FlowER's own beam search is used unchanged; only the input differs. Its
script assigns atom maps itself, so here the reactant SMILES is built from
the PGL initial state with every hydrogen explicit and atom map n = the n-th
atom, and the map -> PGL label table is kept. Every node of FlowER's search
graph is converted back to a PGL ReactionState and scored with the same
develop/fukuyama_metrics.py functions as the LLM arms.

Path choices reported (all from the same beam search):
  top1        highest cumulative-probability root-to-terminal path (forward,
              target unused)
  target      highest-probability path whose final state contains every
              molecule of the supplied target state; falls back to top1
              (target-given setting, comparable to the LLM arms)
  any         whether ANY path in the beam graph scores correct (upper bound)

Run with the isolated Python 3.10 env (torch 2.4):
  <scratchpad>/flowerenv/bin/python experiments/flower_compare/flower_driver.py --set dev
"""
from __future__ import annotations

import argparse
import itertools
import signal
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FLOWER = ROOT.parent / 'external' / 'FlowER'
os.environ.update({'DATA_NAME': 'flower_new_dataset', 'EXP_NAME': 'best_large_hyperparam', 'EMB_DIM': '256',
                   'RBF_HIGH': '12', 'RBF_GAP': '0.1', 'SIGMA': '0.15', 'MODEL_NAME': 'model.2940000_97.pt',
                   'SCALE': '1', 'TEST_BATCH_SIZE': '2048', 'NUM_WORKERS': '0', 'NUM_GPUS_PER_NODE': '1',
                   'MODEL_PATH': str(FLOWER / 'checkpoints/flower_new_dataset/best_large_hyperparam') + '/',
                   'RESULT_PATH': '/dev/null/'})
sys.path.insert(0, str(FLOWER))
sys.path.insert(1, str(ROOT))

import networkx as nx
import numpy as np
import torch
from rdkit import Chem, RDLogger

import beam_predict as BP
from settings import Args
from train import init_model

from develop import fukuyama_metrics as fm
from develop import standalone_chemistry as chem

RDLogger.DisableLog('rdApp.*')
KINDS = {1.0: Chem.BondType.SINGLE, 2.0: Chem.BondType.DOUBLE, 3.0: Chem.BondType.TRIPLE, 1.5: Chem.BondType.AROMATIC}
ORDER = {Chem.BondType.SINGLE: 1.0, Chem.BondType.DOUBLE: 2.0, Chem.BondType.TRIPLE: 3.0, Chem.BondType.AROMATIC: 1.5}


def pgl_to_mapped_smiles(state):
    """All hydrogens explicit; returns (smiles, map number -> PGL label)."""
    atoms = dict(chem.pgl.iter_atoms(state))
    names = sorted(atoms)
    rw = Chem.RWMol()
    index = {}
    for n in names:
        a = Chem.Atom(atoms[n].element)
        a.SetFormalCharge(int(atoms[n].charge or 0))
        a.SetNoImplicit(True)
        index[n] = rw.AddAtom(a)
    for (x, y), bond in chem.pgl.bond_map(state).items():
        rw.AddBond(index[x], index[y], KINDS.get(float(bond.order or 1.0), Chem.BondType.SINGLE))
    labels = {index[n]: n for n in names}
    for n in names:  # recorded Hcount -> explicit hydrogen atoms
        for k in range(int(atoms[n].Hcount or 0)):
            h = rw.AddAtom(Chem.Atom('H'))
            rw.AddBond(index[n], h, Chem.BondType.SINGLE)
            labels[h] = f'Hx{n}_{k}'
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    mapping = {}
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
        mapping[atom.GetIdx() + 1] = labels[atom.GetIdx()]
    return Chem.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=True), mapping


def mapped_smiles_to_state(smiles, mapping):
    mol = Chem.MolFromSmiles(smiles, BP.ps if hasattr(BP, 'ps') else None)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    Chem.Kekulize(mol, clearAromaticFlags=False) if False else None
    atomdict, bonds = {}, []
    for atom in mol.GetAtoms():
        label = mapping[atom.GetAtomMapNum()]
        atomdict[label] = chem.Atom(atom.GetSymbol(), charge=atom.GetFormalCharge())
    for b in mol.GetBonds():
        x, y = mapping[b.GetBeginAtom().GetAtomMapNum()], mapping[b.GetEndAtom().GetAtomMapNum()]
        order = ORDER.get(b.GetBondType(), 1.0)
        bonds.append(chem.Bond(x, y, chem._order_kind(order), order=order))
    return chem.ReactionState([chem.Species('reactant', 'stable', chem.Molecule('', [chem.Substructure('', atomdict, bonds)]))])


def score_states(case, states, main_view=None):
    initial = chem.parse_pgl_literal(case['initial_state_pgl'])
    steps = [fm.step_from_states(b, a) for b, a in zip([initial] + states[:-1], states)]
    gold = fm.gold_steps_from_clean(case)
    gold_states = [chem.parse_pgl_literal(s) for s in case['states_pgl']]
    ordered, hits = fm.intermediates_in_order(states, gold_states)
    target = fm.state_view(chem.parse_pgl_literal(case['target_state_pgl']))
    final = fm.state_view(states[-1]) if states else None
    reaches = bool(states) and fm.same_state(final, target)
    main = bool(states) and main_view is not None and fm.same_state(final, main_view)
    sig = fm.sigma_scores(steps, gold)
    return {'sigma_L1': sig['sigma_L1'], 'sigma_L2': sig['sigma_L2'], 'intermediates_in_order': ordered,
            'intermediates_hit': hits, 'reaches_target': reaches, 'reaches_main_product': main,
            'steps': len(states), 'predicted_transient': sig['predicted_transient']}


def main_product_view(case):
    """The target molecule with the most heavy atoms that is not already in the initial state.

    FlowER's own success check looks for the principal product only; the full
    target also contains every byproduct and salt, which is stricter.
    """
    target = chem.parse_pgl_literal(case['target_state_pgl'])
    initial_view = fm.state_view(chem.parse_pgl_literal(case['initial_state_pgl']))
    view = fm.state_view(target)
    best = None
    for comp in fm._components(view):
        single = {**view, 'component': {n: comp for n in comp}}
        if fm.same_state(initial_view, {**view, 'component': {n: comp for n in comp},
                                         'atoms': {n: view['atoms'][n] for n in comp}}):
            continue  # unchanged spectator
        size = sum(1 for n in comp if view['atoms'][n][0] != 'H')
        if best is None or size > best[0]:
            best = (size, comp)
    if best is None:
        return None
    comp = best[1]
    return {'atoms': {n: view['atoms'][n] for n in comp}, 'hcount': {n: view['hcount'][n] for n in comp},
            'hcharge': {n: v for n, v in view['hcharge'].items() if n in comp},
            'adjacency': {n: view['adjacency'][n] for n in comp},
            'orders': {p: o for p, o in view['orders'].items() if p <= comp},
            'component': {n: comp for n in comp}}


def graph_to_json(graph, root, mapping):
    nodes = list(graph.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    return {'root': index[root], 'nodes': nodes, 'mapping': {str(k): v for k, v in mapping.items()},
            'edges': [[index[u], index[v], int(d.get('count', 0)), int(d.get('rank', 0))] for u, v, d in graph.edges(data=True)]}


def graph_from_json(data):
    graph = nx.DiGraph()
    nodes = data['nodes']
    graph.add_nodes_from(nodes)
    for u, v, count, rank in data['edges']:
        graph.add_edge(nodes[u], nodes[v], count=count, rank=rank)
    return graph, nodes[data['root']], {int(k): v for k, v in data['mapping'].items()}


def score_graph(case, graph, root, mapping, sample_size=64, per_node=20, max_paths=3000):
    """Score every root-to-node path of FlowER's search graph.

    top1   highest-probability path ending at a node FlowER itself marked as
           finished (self-loop); pruned beam leaves are not completions.
    target highest-probability path whose end state contains the principal
           product (fallback: top1). Comparable to the target-given LLM arms.
    any_*  whether ANY path to ANY node scores correct (upper bound).
    """
    main_view = main_product_view(case)
    finished = {n for n in graph.nodes if graph.has_edge(n, n)}
    scored, count = [], 0
    for node in graph.nodes:
        if node == root:
            continue
        for path in itertools.islice(nx.all_simple_paths(graph, root, node), per_node):
            prob = float(np.prod([graph[u][v]['count'] / sample_size for u, v in zip(path, path[1:])]))
            states = []
            for smi in path[1:]:
                st = mapped_smiles_to_state(smi, mapping)
                if not states or chem.state_signature(st) != chem.state_signature(states[-1]):
                    states.append(st)
            scored.append({'prob': prob, 'finished': node in finished, **score_states(case, states, main_view)})
            count += 1
            if count >= max_paths:
                break
        if count >= max_paths:
            break
    done = sorted([s for s in scored if s['finished']], key=lambda x: -x['prob'])
    product = sorted([s for s in scored if s['reaches_main_product']], key=lambda x: -x['prob'])
    top1 = done[0] if done else (product[0] if product else None)
    target = product[0] if product else top1
    return {'graph_nodes': graph.number_of_nodes(), 'n_paths': len(scored), 'n_finished_paths': len(done),
            'any_reaches_main_product': bool(product), 'any_reaches_target': any(s['reaches_target'] for s in scored),
            'any_sigma_L1': any(s['sigma_L1'] for s in scored),
            'any_in_order': any(s['intermediates_in_order'] for s in scored),
            'max_intermediates_hit': max((s['intermediates_hit'] for s in scored), default=0),
            'top1': top1, 'target': target, 'main_product_defined': main_view is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', choices=['dev', 'heldout'], required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='cpu')  # MPS lacks float64, which FlowER's BE matrices use
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'flower_compare_v2_20261031')
    ap.add_argument('--shard', default='0/1', help='i/n: process every n-th case starting at i')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--ids-file', type=Path, default=None, help='JSON with an "ids" list restricting the held-out cases')
    ap.add_argument('--case-timeout', type=int, default=1200,
                    help='wall-clock seconds per case; a timed-out case is recorded as FlowER finding no path')
    cli = ap.parse_args()
    if cli.threads:
        torch.set_num_threads(cli.threads)
    cli.output.mkdir(parents=True, exist_ok=True)
    if cli.set == 'dev':
        ids = json.loads((ROOT / 'outputs/p23_planning_probe_20261024/cases.json').read_text())
        pool = {c['case_id']: c for c in map(json.loads, (ROOT / 'outputs/fukuyama/clean.jsonl').read_text().splitlines())}
    else:
        reg = json.loads((ROOT / 'outputs/heldout_fukuyama_20261024/REGISTRATION.json').read_text())
        ids = reg['sets']['all_218']
        pool = {c['case_id']: c for c in map(json.loads, (ROOT / 'outputs/heldout_fukuyama_20261024/heldout.jsonl').read_text().splitlines())}
    if cli.ids_file:
        keep = set(json.loads(cli.ids_file.read_text())['ids'])
        ids = [i for i in ids if i in keep]
    ids = ids[: cli.limit or None]
    shard, shards = map(int, cli.shard.split('/'))
    ids = ids[shard::shards]

    args = Args
    args.local_rank = -1
    args.device = torch.device(cli.device)
    state = torch.load(os.path.join(args.model_path, args.model_name), weights_only=False, map_location='cpu')
    pre = state['args']
    pre.load_from, pre.device, pre.local_rank = None, args.device, -1
    model, flow, _ = init_model(pre)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['state_dict'].items()})
    model.eval()

    for cid in ids:
        dest = cli.output / cli.set / f'{cid}.json'
        if dest.exists():
            continue
        case = pool[cid]
        t0 = time.time()
        record = {'case_id': cid, 'partition': case['partition']}
        try:
            smiles, mapping = pgl_to_mapped_smiles(chem.parse_pgl_literal(case['initial_state_pgl']))
            graph = nx.DiGraph()
            graph.add_node(smiles, depth=1)
            graph_list = [(graph, smiles, (smiles, []))]
            def on_alarm(signum, frame):
                raise TimeoutError(f'beam search exceeded {cli.case_timeout}s')
            signal.signal(signal.SIGALRM, on_alarm)
            signal.alarm(cli.case_timeout)
            try:
                BP.beam_search(args, model, flow, {0: [smiles]}, graph_list)
                record['timed_out'] = False
            except TimeoutError:
                # Score whatever the graph holds so far; flag the case.
                record['timed_out'] = True
            finally:
                signal.alarm(0)
            record['graph'] = graph_to_json(graph, smiles, mapping)
            record.update(score_graph(case, graph, smiles, mapping, sample_size=Args.sample_size))
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {str(exc)[:300]}'
        record['seconds'] = round(time.time() - t0, 1)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(record, ensure_ascii=False, indent=1))
        print(json.dumps({k: record.get(k) for k in ('case_id', 'n_paths', 'any_reaches_main_product', 'any_sigma_L1', 'error', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
