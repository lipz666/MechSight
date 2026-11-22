"""Score the specialised models and the LLM direct baseline on the same cases and metrics.

ChRIMP (MechSMILES T5, equ_min_n, FlowER-trained): top-1 = first returned mechanism; its
  intermediate states are the reactant sides of steps 2..n, then the target product.
FlowER: from the saved beam-search graph, the highest-probability root path whose last node
  contains the main product; else the best finished path; else the best path (top-1 of the
  same rule as the Fukuyama driver). Nodes are full mapped states, unmapped here.
Metrics are species-level (official oMeS needs step subtypes, which neither model predicts):
oMe-Gold in_order_full / all_present; T3 all GT intermediates present; uspto31k in-order full.
Usage: spec_score.py <chrimp_dir> <flower_fuku_dir> <flower_smiles_dir>
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import networkx as nx
from rdkit import Chem, RDLogger

import common4 as K

RDLogger.DisableLog('rdApp.*')


def unmap(smi):
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return ''
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    try:
        Chem.SanitizeMol(m)
        m = Chem.RemoveHs(m)
    except Exception:
        return ''
    return Chem.MolToSmiles(m)


def cases():
    out = {('gold', c): v for c, v in K.load_gold().items()}
    for w in ('T3_dev', 'T3_confirm'):
        out.update({(w, c): v for c, v in K.load_fukuyama(w).items()})
    out.update({('uspto31k', c): v for c, v in K.load_uspto().items()})
    return out


def main_keys(case):
    from spec_inputs import main_product
    return K.species_keys(main_product(case['products'], case['reactants']))


def flower_path(graph, case, max_paths=3000, per_node=20):
    nodes = graph['nodes']
    g = nx.DiGraph()
    g.add_nodes_from(range(len(nodes)))
    for e in graph['edges']:
        g.add_edge(e[0], e[1], count=e[2])
    root, target = graph['root'], main_keys(case)
    finished = {n for n in g.nodes if g.has_edge(n, n)}
    best = {'product': None, 'finished': None, 'any': None}
    count = 0
    for node in g.nodes:
        if node == root:
            continue
        for path in itertools.islice(nx.all_simple_paths(g, root, node), per_node):
            prob = 1.0
            for u, v in zip(path, path[1:]):
                prob *= g[u][v]['count'] / 64
            last = K.species_keys(unmap(nodes[path[-1]]))
            kinds = ['any'] + (['finished'] if node in finished else []) + (['product'] if target and target <= last else [])
            for k in kinds:
                if best[k] is None or prob > best[k][0]:
                    best[k] = (prob, path)
            count += 1
            if count >= max_paths:
                break
        if count >= max_paths:
            break
    pick = best['product'] or best['finished'] or best['any']
    return [unmap(nodes[i]) for i in pick[1][1:]] if pick else []


def chrimp_states(rec, case):
    if not rec.get('mechanisms'):
        return []
    from spec_inputs import main_product
    return rec['mechanisms'][0] + [main_product(case['products'], case['reactants'])]


def bench_of(ds):
    return 'T3' if ds.startswith('T3') else ds


def main():
    chrimp_dir, fuku_dir, smiles_dir = map(Path, sys.argv[1:4])
    cs = cases()
    rows = {}
    for (ds, cid), case in cs.items():
        b = bench_of(ds)
        f = chrimp_dir / f'{ds}__{cid}.json'
        if f.exists():
            rows.setdefault((b, 'ChRIMP'), []).append((cid, K.score(case, [{'intermediate_smiles': s} for s in chrimp_states(json.loads(f.read_text()), case)])))
        f = (fuku_dir / f'{cid}.json') if b == 'T3' else (smiles_dir / f'{ds}__{cid}.json')
        if f.exists():
            rec = json.loads(f.read_text())
            states = flower_path(rec['graph'], case) if rec.get('graph') else []
            rows.setdefault((b, 'FlowER'), []).append((cid, K.score(case, [{'intermediate_smiles': s} for s in states])))
        a = K.cached(f'{ds}/base_default', cid)
        if a is not None:
            rows.setdefault((b, 'LLM_direct'), []).append((cid, K.score(case, K.parse_mech(a))))
    rep = {}
    for (b, m), r in sorted(rows.items()):
        keys = {'gold': ['in_order_full', 'all_present', 'in_order_frac', 'S_partial'], 'T3': ['all_present', 'in_order_frac'],
                'uspto31k': ['in_order_full', 'all_present', 'in_order_frac']}[b]
        rep.setdefault(b, {})[m] = K.summarize(r, keys)
    (K.OUT / 'SUMMARY_SPEC.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
