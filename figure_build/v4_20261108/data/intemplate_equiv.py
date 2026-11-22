"""Offline (no model calls): in-template (mech-USPTO, FlowER-covered 196) tolerant complete-pathway for
FlowER, ChRIMP and LLM Direct, using p43's tol_all_present rule; plus strict and reached for cross-check."""
import json, sys, random
sys.path.insert(0, '.')
import common4 as K, spec_score as SS, p18_spec_compare as P18
import importlib.util
spec = importlib.util.spec_from_file_location('p43', 'p43_flower_diag.py')
src = open('p43_flower_diag.py').read().split('rows = []')[0]
ns = {}; exec(src, ns)
tol = ns['tol_all_present']
cases = K.load_uspto()
rows = {a: [] for a in ('FlowER', 'ChRIMP', 'LLM_direct')}
for cid, case in cases.items():
    f = K.OUT / 'spec_v5' / 'flower' / f'uspto31k__{cid}.json'
    if not f.exists(): continue
    rec = json.loads(f.read_text())
    if 'CUDA' in (rec.get('error') or ''): continue
    sel = SS.flower_path(rec['graph'], case) if rec.get('graph') else []
    fc = K.OUT / 'spec_v5' / 'chrimp' / f'uspto31k__{cid}.json'
    ch = SS.chrimp_states(json.loads(fc.read_text()), case) if fc.exists() else []
    ll = [s.get('intermediate_smiles') for s in (K.parse_mech(K.cached('uspto31k/base_default', cid)) or [])]
    for a, smis in (('FlowER', sel), ('ChRIMP', ch), ('LLM_direct', ll)):
        st = [{'intermediate_smiles': x} for x in smis if x]
        rows[a].append(dict(strict=K.score(case, st)['all_present'] if st else 0.0, tol=tol(case, smis)))
def ci(v, B=2000, seed=20261107):
    r = random.Random(seed); n = len(v); m = sum(v) / n
    bs = sorted(sum(r.choice(v) for _ in v) / n for _ in range(B))
    return [round(100 * m, 1), round(100 * bs[int(.025 * B)], 1), round(100 * bs[int(.975 * B) - 1], 1)]
out = {a: dict(n=len(r), strict=ci([x['strict'] for x in r]), equiv=ci([x['tol'] for x in r])) for a, r in rows.items()}
print(json.dumps(out))
json.dump(out, open(sys.argv[1], 'w'), indent=1)
