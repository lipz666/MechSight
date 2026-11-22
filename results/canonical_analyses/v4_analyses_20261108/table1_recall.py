"""Re-score Table 1 arms (T3_confirm + oMe-Gold) under chemically-equivalent matching.

Outputs are untouched; only species matching changes. Tiers:
  strict : the paper's pre-registered scorer (K.score)
  equiv  : same constitution up to protonation state, mobile-H tautomer and stereo depiction
           (standard InChIKey skeleton block per component); a reference intermediate that is
           only a (de)protonated reactant/product counts as satisfied (proton-shuttle granularity).
Run from MechAgent/experiments/pilots_v4.
"""
import json, sys
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
    f = K.OUT / 'spec_v5' / kind / f'{ds}__{cid}.json'
    if not f.exists():
        return None
    rec = json.loads(f.read_text())
    if 'CUDA' in (rec.get('error') or ''):
        return None
    st = (SS.flower_path(rec['graph'], case) if rec.get('graph') else []) if kind == 'flower' else SS.chrimp_states(rec, case)
    return [{'intermediate_smiles': s} for s in st]



import random
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
        endpoints = set().union(*[ekeys(x) for x in case['reactants'] + case['products']])
        for arm in ARMS:
            if arm not in ans:
                continue
            st = ans[arm] or []
            smis = [s.get('intermediate_smiles') or '' for s in st]
            KF = K.species_keys if ds == 'T3_confirm' else (lambda x: K.j4_score.keys_of_smiles(x) if x else set())
            U = set().union(*[KF(x) or set() for x in smis]) if smis else set()
            UE = (set().union(*[ekeys(x) for x in smis]) if smis else set()) | endpoints
            hs, he = [], []
            for r in ref:
                k = KF(r); ke = ekeys(r)
                if not k and not ke: continue
                s_ = bool(smis) and bool(k) and k <= U
                e_ = bool(smis) and (s_ or (bool(ke) and ke <= UE))
                hs.append(s_); he.append(e_)
            strict = K.score(case, st)['all_present'] if st else 0.0
            e_all, _ = equiv(case, ref, st)
            rows.setdefault(arm, []).append((ds, {'strict': strict, 'equiv': max(strict, e_all),
                 'rec_s': sum(hs)/len(hs) if hs else 1.0, 'rec_e': sum(he)/len(he) if he else 1.0,
                 'all_s_from_rec': float(all(hs)) if hs else 1.0}))
def ci(v, B=4000):
    r = random.Random(0); n = len(v)
    bs = sorted(sum(v[r.randrange(n)] for _ in range(n))/n for _ in range(B))
    return round(100*sum(v)/n,1), round(100*bs[100],1), round(100*bs[3899],1)
out = {}
for scope in ('T3_confirm', 'gold', 'pooled'):
    for arm, r in rows.items():
        sub = [m for d, m in r if scope == 'pooled' or d == scope]
        out.setdefault(scope, {})[arm] = {k: ci([m[k] for m in sub]) for k in ('strict','equiv','rec_s','rec_e','all_s_from_rec')} | {'n': len(sub)}
    d = [a['rec_e'] - b['rec_e'] for (x, a), (y, b) in zip(rows['MechAgent'], rows['Direct']) if scope=='pooled' or x==scope]
    out[scope]['Auto-Direct rec_e'] = ci(d)
    d = [a['rec_s'] - b['rec_s'] for (x, a), (y, b) in zip(rows['MechAgent'], rows['Direct']) if scope=='pooled' or x==scope]
    out[scope]['Auto-Direct rec_s'] = ci(d)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
for scope, blk in out.items():
    for arm, v in blk.items():
        print(scope, arm, v if isinstance(v, tuple) else {k: (x[0] if isinstance(x, tuple) else x) for k, x in v.items()})
