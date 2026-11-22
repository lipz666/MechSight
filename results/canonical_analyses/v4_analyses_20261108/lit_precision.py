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
rows = []
for cid, c in cs.items():
    if not c['clues'] or not any(K.species_keys(s) for s in c['author']): continue
    if K.cached(f'{DS}/G71_rev', cid) is None: continue
    if dates.get(cid, 'NA') < '2026-04': continue
    ends_s = set().union(*[K.species_keys(x) or set() for x in c.get('reactants', []) + c.get('products', [])])
    ends_e = set().union(*[ek(x) for x in c.get('reactants', []) + c.get('products', [])])
    GS = set().union(*[K.species_keys(s) or set() for s in c['author']])
    GE = set().union(*[ek(s) for s in c['author']]) | ends_e
    r = {'id': cid, 'a1': K.cached(f'{DS}/A1_place', cid) is not None, 'n_author': sum(bool(K.species_keys(s)) for s in c['author'])}
    for a, f in ARMS.items():
        smis = [s.get('intermediate_smiles') or '' for s in (K.parse_mech(f(cid)) or [])]
        ps, pe, n = 0, 0, 0
        for x in smis:
            k = K.species_keys(x); ke = ek(x)
            if not k or k <= ends_s: continue
            n += 1; st = k <= GS; ps += st; pe += st or (bool(ke) and ke <= GE)
        r[a] = (n, ps, pe)
    rows.append(r)
json.dump(rows, open(out, 'w')); print(which, len(rows))
