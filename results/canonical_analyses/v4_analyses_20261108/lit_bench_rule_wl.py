"""Literature recall/complete scored with exactly the Table 1 rules.
strict: every sigma-graph component of the reference intermediate appears among predicted species (K.score style).
equiv : standard-InChIKey skeleton components, predicted species plus reactants/products (p40 style)."""
import json, sys
sys.path.insert(0, '.')
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
import common4 as K, p11_mechagent as M, p30_lit172_gold as P, p31_guided71 as T
which, dfile, out = sys.argv[1:4]; T.configure(which); DS = P.DS
cs = P.cases(); P.CS = cs; dates = json.load(open(dfile))
def ek(smi):
    o = set()
    for s in (smi or '').split('.'):
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        try: o.add(Chem.MolToInchiKey(m).split('-')[0])
        except Exception: pass
    return o
def gfull(c):
    a = K.cached(f'{DS}/A1_place', c); return a if a else K.cached(f'{DS}/G71_rev', c)
ARMS = {'Direct': lambda c: K.cached(f'{DS}/base_default', c), 'Auto': lambda c: M.final(DS, c, 2),
        'Direct+Clues': lambda c: K.cached(f'{DS}/O_llm', c), 'Guided': gfull}
NDS = 'litnew_g_nodft'
def _g(ds, c):
    a = K.cached(f'{ds}/A1_place', c); return a if a is not None else K.cached(f'{ds}/G71_rev', c)
if which == 'test':
    ARMS['DC_wl'] = lambda c: K.cached(f'{NDS}/O_llm', c) if K.cached(f'{NDS}/O_llm', c) is not None else K.cached(f'{DS}/O_llm', c)
    ARMS['Guided_wl'] = lambda c: _g(NDS, c) if K.cached(f'{NDS}/O_llm', c) is not None else gfull(c)
POOL = ['base_default', 'samp1', 'samp2', 'samp3', 'samp4', 'enum3']
rows = []
for cid, c in cs.items():
    if not c['clues'] or not any(K.species_keys(s) for s in c['author']): continue
    if K.cached(f'{DS}/G71_rev', cid) is None: continue
    ends_e = set().union(*[ek(x) for x in c.get('reactants', []) + c.get('products', [])])
    smis = {a: [s.get('intermediate_smiles') or '' for s in (K.parse_mech(f(cid)) or [])] for a, f in ARMS.items()}
    sup = T.G.proposals(c); mvs = [s for s, n in sup.items() if n >= 4]
    smis['MV7'] = mvs
    U = {a: set().union(*[K.species_keys(x) or set() for x in v]) if v else set() for a, v in smis.items()}
    UE = {a: (set().union(*[ek(x) for x in v]) if v else set()) | ends_e for a, v in smis.items()}
    pool_s = []
    for t in POOL: pool_s += [s.get('intermediate_smiles') or '' for s in (K.parse_mech(K.cached(f'{DS}/{t}', cid)) or [])]
    for t in ('V5_comp1', 'V5_comp2'): pool_s += [s for _, s in M.checked_states(c, K.cached(f'{DS}/{t}', cid) or '')]
    PU = set().union(*[K.species_keys(x) or set() for x in pool_s]) if pool_s else set()
    PUE = (set().union(*[ek(x) for x in pool_s]) if pool_s else set()) | ends_e
    strata = []
    for s in c['author']:
        g = K.species_keys(s)
        if not g: continue
        ge = ek(s)
        hit = {a: bool(g <= U[a]) or bool(ge and ge <= UE[a]) for a in U}
        hs = {a + '_strict': bool(g <= U[a]) for a in U}
        st = 'direct' if hit['Direct'] else ('inert' if (g <= PU or (ge and ge <= PUE)) else 'blind')
        strata.append({'st': st, **hit, **hs})
    rows.append({'id': cid, 'set': which, 'date': dates.get(cid, 'NA'), 'strata': strata})
json.dump(rows, open(out, 'w')); print(which, len(rows))
