import json, sys, random, collections
sys.path.insert(0, '.')
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
import common4 as K
import p31_m2_exposure as E
def ek(smi):
    o = set()
    for s in (smi or '').split('.'):
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        try: o.add(Chem.MolToInchiKey(m).split('-')[0])
        except Exception: pass
    return o
old = E.load_cases(E.DS_OLD, E.GOLD_OLD, [p['pmcid'] for p in E.SCOPE])
rows = []
for cid, c in old.items():
    smis = [s.get('intermediate_smiles') or '' for s in (K.parse_mech(K.cached(f'{E.DS_OLD}/base_default', cid)) or [])]
    U = set().union(*[K.species_keys(x) or set() for x in smis]) if smis else set()
    ends = set().union(*[ek(x) for x in c.get('reactants', []) + c.get('products', [])])
    UE = (set().union(*[ek(x) for x in smis]) if smis else set()) | ends
    hs, he = [], []
    for s in c['author']:
        g = K.species_keys(s)
        if not g: continue
        ge = ek(s); st = bool(smis) and g <= U
        hs.append(st); he.append(bool(smis) and (st or (bool(ge) and ge <= UE)))
    if hs: rows.append({'tier': E.META[cid]['exposure'], 's': sum(hs)/len(hs), 'e': sum(he)/len(he)})
new = [r for r in json.load(open(sys.argv[1] + '/bench_eq/rows_dev.json')) + json.load(open(sys.argv[1] + '/bench_eq/rows_test.json')) if r['date'] >= '2026-04']
for r in new:
    rows.append({'tier': '2026', 's': sum(x['Direct_strict'] for x in r['strata'])/len(r['strata']), 'e': sum(x['Direct'] for x in r['strata'])/len(r['strata'])})
def ci(v):
    g = random.Random(0); n = len(v); bs = sorted(sum(v[g.randrange(n)] for _ in range(n))/n for _ in range(4000))
    return round(100*sum(v)/n, 1), round(100*bs[100], 1), round(100*bs[3899], 1)
out = {}
for t in ('low', 'mid', 'high', '2026'):
    X = [r for r in rows if r['tier'] == t]
    out[t] = {'n': len(X), 'strict': ci([r['s'] for r in X]), 'equiv': ci([r['e'] for r in X])}
hi = [r for r in rows if r['tier'] == 'high']; nw = [r for r in rows if r['tier'] == '2026']
g = random.Random(1)
for k in ('s', 'e'):
    bs = sorted(sum(r[k] for r in [g.choice(hi) for _ in hi])/len(hi) - sum(r[k] for r in [g.choice(nw) for _ in nw])/len(nw) for _ in range(4000))
    out[f'high-2026_{k}'] = (round(100*(sum(r[k] for r in hi)/len(hi) - sum(r[k] for r in nw)/len(nw)), 1), round(100*bs[100], 1), round(100*bs[3899], 1))
print(json.dumps(out, indent=1)); json.dump(out, open(sys.argv[1] + '/exposure_unified.json', 'w'), indent=1)
