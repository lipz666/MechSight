"""Fig. 2 (method) worked example J3-B057: pool, per-step heavy-atom bond changes, Stage-2 proposals, revised path.
Reads only the frozen pilots_v4 cache; writes data/method_example.json."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'MechAgent/experiments/pilots_v4'))
import common4 as K
import p11_mechagent as M

ds, cid = 'J3', 'B057'
case = M.sets(('J3',))['J3'][cid]
out = {'reactants': case['reactants'], 'products': case['products'], 'conditions': case['conditions']}
pool = M.P10.pool(ds, cid)
out['pool_members'] = {n: [s.get('intermediate_smiles') for s in m] for n, m in pool.items()}
def prof(m):
    return [M.changes(r) for r in M.step_rxns(case, m)]
out['pool_profiles'] = {n: prof(m) for n, m in pool.items()}
cl = M.cand_list(ds, cid)
out['cands'] = [{'smiles': [s.get('intermediate_smiles') for s in m], 'types': [s.get('subtype') or s.get('type') for s in m],
                 'dbonds': prof(m), 'gaps': M.gaps(case, m)} for m in cl]
comp1 = K.cached(f'{ds}/V5_comp1', cid)
out['comp1_answer'] = comp1
out['comp1_checked'] = M.checked_states(case, comp1)
for tag in ('V5_rev1', 'V5_comp2', 'V5_rev2'):
    a = K.cached(f'{ds}/{tag}', cid)
    out[tag + '_raw'] = a
for r in (1, 2):
    m = K.parse_mech(M.final(ds, cid, r))
    out[f'final_r{r}'] = {'smiles': [s.get('intermediate_smiles') for s in m], 'types': [s.get('subtype') or s.get('type') for s in m],
                          'dbonds': prof(m), 'gaps': M.gaps(case, m), 'score': K.score(case, m)}
base = K.parse_mech(K.cached(f'{ds}/base_default', cid))
out['direct'] = {'smiles': [s.get('intermediate_smiles') for s in base], 'dbonds': prof(base), 'score': K.score(case, base)}
out['gt'] = case.get('intermediates') or case.get('gt') or {k: v for k, v in case.items() if k not in ('reactants', 'products', 'conditions')}
Path(__file__).with_name('method_example.json').write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str)[:9000])
