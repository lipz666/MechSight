"""Diagnose why FlowER scores low on the in-template (mech-USPTO-31k) set.
Run from MechAgent/experiments/pilots_v4. Read-only on project outputs."""
import itertools, json, sys
from collections import Counter
sys.path.insert(0, '.')
import networkx as nx
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
import common4 as K
import spec_score as SS

SPEC = K.OUT / 'spec_v5' / 'flower'
cases = K.load_uspto()

def all_paths(graph, max_paths=3000, per_node=20):
    nodes = graph['nodes']; g = nx.DiGraph(); g.add_nodes_from(range(len(nodes)))
    for e in graph['edges']: g.add_edge(e[0], e[1], count=e[2])
    root = graph['root']; out = []
    for node in g.nodes:
        if node == root: continue
        for p in itertools.islice(nx.all_simple_paths(g, root, node), per_node):
            out.append([SS.unmap(nodes[i]) for i in p[1:]])
            if len(out) >= max_paths: return out
    return out

def inchi_tol(smi):
    out = set()
    for s in (smi or '').split('.'):
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        try: out.add(Chem.MolToInchiKey(m).split('-')[0])
        except Exception: pass
    return out

def tol_all_present(case, smis):
    start = set().union(*[inchi_tol(x) for x in case['reactants']])
    final = inchi_tol(case['states'][-1])
    gk = [g for st in case['states'][:-1] if (g := inchi_tol(st) - start - final)]
    allp = set().union(*[inchi_tol(x) for x in smis]) if smis else set()
    return float(bool(smis) and all(k <= allp for k in gk))

rows = []
for cid, case in cases.items():
    f = SPEC / f'uspto31k__{cid}.json'
    if not f.exists(): continue
    rec = json.loads(f.read_text())
    if 'CUDA' in (rec.get('error') or ''): continue
    sel = SS.flower_path(rec['graph'], case) if rec.get('graph') else []
    s = K.score(case, [{'intermediate_smiles': x} for x in sel])
    tgt = SS.main_keys(case)
    reached = float(bool(tgt) and any(tgt <= K.species_keys(x) for x in sel if x))
    paths = all_paths(rec['graph']) if rec.get('graph') else []
    any_strict = any(K.score(case, [{'intermediate_smiles': x} for x in p])['all_present'] for p in paths)
    any_tol = any(tol_all_present(case, p) for p in paths)
    rows.append(dict(id=cid, timeout=bool(rec.get('timed_out')), reached=reached, strict=s['all_present'],
                     tol=tol_all_present(case, sel), any_strict=float(any_strict), any_tol=float(any_tol),
                     n_steps=case.get('n_steps'), n_gt=len(case['states']) - 1))
n = len(rows)
mean = lambda k, rs=rows: round(100 * sum(r[k] for r in rs) / max(1, len(rs)), 1)
print('n', n)
for k in ('timeout', 'reached', 'strict', 'tol', 'any_strict', 'any_tol'): print(k, mean(k))
fin = [r for r in rows if not r['timeout']]; to = [r for r in rows if r['timeout']]
print('finished n', len(fin), {k: mean(k, fin) for k in ('reached', 'strict', 'tol', 'any_tol')})
print('timeout  n', len(to), {k: mean(k, to) for k in ('reached', 'strict', 'tol', 'any_tol')})
rp = [r for r in rows if r['reached']]
print('reached-product n', len(rp), {k: mean(k, rp) for k in ('strict', 'tol')})
print('steps dist', Counter(r['n_steps'] for r in rows))
for L in sorted({r['n_steps'] for r in rows}):
    sub = [r for r in rows if r['n_steps'] == L]
    print(' steps', L, 'n', len(sub), 'tol', mean('tol', sub), 'reached', mean('reached', sub), 'timeout', mean('timeout', sub))
json.dump(rows, open(sys.argv[1], 'w'), indent=0)
