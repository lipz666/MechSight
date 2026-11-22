"""Re-score Table 1 arms (T3_confirm + oMe-Gold) under chemically-equivalent matching.

Outputs are untouched; only species matching changes. Tiers:
  strict : the paper's pre-registered scorer (K.score)
  equiv  : same constitution up to protonation state, mobile-H tautomer and stereo depiction
           (standard InChIKey skeleton block per component); a reference intermediate that is
           only a (de)protonated reactant/product counts as satisfied (proton-shuttle granularity).
Run from MechAgent/experiments/pilots_v4.
"""
import json, os, sys
sys.path.insert(0, '.')
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
import common4 as K
import p11_mechagent as M
import p12_controls as C
import p14_confirm as P14
import spec_score as SS


def ekeys(smi):
    out = set()
    for s in (smi or '').split('.'):
        m = Chem.MolFromSmiles(s)
        if m is None:
            m = Chem.MolFromSmiles(s, sanitize=False)
            if m is None:
                continue
            try:
                m.UpdatePropertyCache(strict=False)
            except Exception:
                continue
        try:
            k = Chem.MolToInchiKey(m)
        except Exception:
            k = ''
        if k:
            out.add(k.split('-')[0])
    return out


def ref_list(ds, case):
    if 'pgl' in case:
        hid, vis = C.gt_intermediates(case)
        return hid + vis
    return [g['intermediate_smiles'] for g in case['gold']]


def equiv(case, ref, steps):
    smis = [s.get('intermediate_smiles') or '' for s in (steps or [])]
    if not smis:
        return 0.0, 0.0
    endpoints = set().union(*[ekeys(x) for x in case['reactants'] + case['products']])
    pred = set().union(*[ekeys(x) for x in smis]) | endpoints
    need = [k for r in ref if (k := ekeys(r))]
    hits = [k <= pred for k in need]
    return float(all(hits)), (sum(hits) / len(hits) if hits else 1.0)


def specialist(ds, cid, case, kind):
    f = K.OUT / os.environ.get('SPEC_DIR', 'spec_v5') / kind / f'{ds}__{cid}.json'  # spec_v6_4090 = 2026-11-08 rerun
    if not f.exists():
        return None
    rec = json.loads(f.read_text())
    if 'CUDA' in (rec.get('error') or ''):
        return None
    st = (SS.flower_path(rec['graph'], case) if rec.get('graph') else []) if kind == 'flower' else SS.chrimp_states(rec, case)
    return [{'intermediate_smiles': s} for s in st]


ARMS = ['Direct', 'CoT', 'SelfRefine', 'SelfConsistency', 'MechAgent', 'Oracle_visible', 'Oracle_hidden', 'FlowER', 'ChRIMP']
rows = {}
for ds, cases in M.sets(('T3_confirm', 'gold')).items():
    for cid, case in cases.items():
        if K.cached(f'{ds}/V5_rev1', cid) is None or K.cached(f'{ds}/base_default', cid) is None:
            continue
        ans = P14.answers(ds, cid, case)
        if ds == 'gold':
            ans['Oracle_hidden'] = K.parse_mech(K.cached('gold/O_hid', cid)) or ans['Direct']
            ans['Oracle_visible'] = K.parse_mech(K.cached('gold/O_vis', cid)) or ans['Direct']
        ans['FlowER'] = specialist(ds, cid, case, 'flower')
        ans['ChRIMP'] = specialist(ds, cid, case, 'chrimp')
        ref = ref_list(ds, case)
        for arm in ARMS:
            if arm not in ans or ans[arm] is None and arm in ('FlowER', 'ChRIMP'):
                continue
            st = ans[arm]
            strict = K.score(case, st)['all_present']
            e_all, e_rec = equiv(case, ref, st)
            rows.setdefault(arm, []).append((f'{ds}:{cid}', {'strict': strict, 'equiv': e_all, 'equiv_recall': e_rec}))

out = {}
for scope in ('T3_confirm', 'gold', 'pooled'):
    blk = {}
    sub = {a: [(c, m) for c, m in r if scope == 'pooled' or c.startswith(scope + ':')] for a, r in rows.items()}
    for a, r in sub.items():
        blk[a] = K.summarize(r, ['strict', 'equiv', 'equiv_recall'])
    for a in ARMS:
        if a != 'Direct' and sub.get(a):
            blk[f'{a}-Direct'] = {k: K.paired(sub[a], sub['Direct'], k) for k in ('strict', 'equiv')}
    for a in ('SelfConsistency', 'SelfRefine', 'CoT'):
        blk[f'MechAgent-{a}'] = {k: K.paired(sub['MechAgent'], sub[a], k) for k in ('strict', 'equiv')}
    out[scope] = blk
json.dump(out, open(sys.argv[1], 'w'), indent=1)

def f(x):
    if not x: return '-'
    if isinstance(x, dict): x = [x.get('mean'), x.get('lo'), x.get('hi')] if 'mean' in x else list(x.values())
    return f'{100*x[0]:.1f} [{100*x[1]:.1f}, {100*x[2]:.1f}]'
for scope, blk in out.items():
    print('==', scope)
    for a in ARMS:
        if a in blk:
            print(f"  {a:16s} n={blk[a]['n']:3d} strict {f(blk[a]['strict'])}  equiv {f(blk[a]['equiv'])}  recall {f(blk[a]['equiv_recall'])}")
    for k, v in blk.items():
        if '-' in k:
            print(f"  {k:28s} strict {f(v['strict'])}  equiv {f(v['equiv'])}")
