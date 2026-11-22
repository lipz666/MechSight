"""Cross-backbone transfer on FukuyamaBench test (T3_confirm, n=48) under the V4 matching rules.

Backbones: gemini-3.8-flash (main), gemini-4-argon, gpt-6-luna (high / no reasoning), gpt-6.1-sol (high),
claude-sonnet-5-5-high. Arms: Direct, SelfConsistency (MBR vote over the same pool), MechSight-Auto.
Metrics: complete pathway (strict / equivalent) and intermediate recall (strict / equivalent), as in table1_recall.py.
Run from MechAgent/experiments/pilots_v4:  python backbone_rule.py OUT.json
"""
import json, random, sys
from pathlib import Path
sys.path.insert(0, '.')
import common4 as K
import p11_mechagent as M
import p12_controls as C
import p14_confirm as P14
import p17_argon_confirm as A
import p3_selector as P3
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')


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



BB = {
    'gemini-3.8-flash': None,
    'gemini-4-argon': ('argon', 'stepmap_argon_cache.json'),
    'gpt-6-luna (high)': ('luna_high', 'stepmap_luna_cache.json'),
    'gpt-6-luna (no reasoning)': ('luna', 'stepmap_luna_cache.json'),
    'gpt-6.1-sol (high)': ('bb_gpt-6_1-sol_high', 'stepmap_bb_gpt-6_1-sol_high_cache.json'),
    'claude-sonnet-5-5-high': ('bb_claude-sonnet-5-5-high', 'stepmap_bb_claude-sonnet-5-5-high_cache.json'),
}


def metrics(case, st):
    st = st or []
    smis = [s.get('intermediate_smiles') or '' for s in st]
    hid, vis = C.gt_intermediates(case)
    endpoints = set().union(*[ekeys(x) for x in case['reactants'] + case['products']])
    U = set().union(*[K.species_keys(x) or set() for x in smis]) if smis else set()
    UE = (set().union(*[ekeys(x) for x in smis]) if smis else set()) | endpoints
    hs, he = [], []
    for r in hid + vis:
        k, ke = K.species_keys(r), ekeys(r)
        if not k and not ke:
            continue
        s_ = bool(smis) and bool(k) and k <= U
        hs.append(s_)
        he.append(bool(smis) and (s_ or (bool(ke) and ke <= UE)))
    strict = K.score(case, st)['all_present'] if st else 0.0
    eq = float(bool(smis) and all(he)) if he else strict
    sc = K.score(case, st) if st else {'n_hidden_events': None}
    return {'cp_s': strict, 'cp_e': max(strict, eq), 'rec_s': sum(hs) / len(hs) if hs else 1.0,
            'rec_e': sum(he) / len(he) if he else 1.0, 'nh': sc.get('n_hidden_events')}


def ci(v, B=4000):
    r = random.Random(0); n = len(v)
    bs = sorted(sum(v[r.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [round(100 * sum(v) / n, 1), round(100 * bs[int(.025 * B)], 1), round(100 * bs[int(.975 * B) - 1], 1)]


cases = M.sets(('T3_confirm',))['T3_confirm']
out = {}
for name, cfg in BB.items():
    rows = {}
    if cfg:
        A.ARGON_OUT = K.OUT / cfg[0]
        A.ARGON_CACHE = K.OUT / cfg[1]
        A.reload_stepmap_cache()
    for cid, case in cases.items():
        if cfg is None:
            if K.cached('T3_confirm/V5_rev1', cid) is None:
                continue
            ans = P14.answers('T3_confirm', cid, case)
            ans = {'Direct': ans['Direct'], 'SelfConsistency': ans['SelfConsistency'], 'MechSight_Auto': ans['MechAgent']}
        else:
            base = A.acached('T3_confirm/base_default', cid)
            if base is None or A.acached('T3_confirm/V5_rev1', cid) is None:
                continue
            p = A.apool('T3_confirm', cid)
            pick = P3.consensus(p, list(p)) if len(p) >= 2 else None
            ans = {'Direct': K.parse_mech(base), 'SelfConsistency': p.get(pick) if pick else K.parse_mech(base),
                   'MechSight_Auto': K.parse_mech(A.afinal('T3_confirm', cid))}
        for a, st in ans.items():
            rows.setdefault(a, {})[cid] = metrics(case, st)
    if not rows:
        continue
    ids = sorted(set.intersection(*[set(v) for v in rows.values()]))
    blk = {'n': len(ids)}
    for a, r in rows.items():
        blk[a] = {k: ci([r[c][k] for c in ids]) for k in ('cp_s', 'cp_e', 'rec_s', 'rec_e')}
    for ref in ('Direct', 'SelfConsistency'):
        blk[f'Auto-{ref}'] = {k: ci([rows['MechSight_Auto'][c][k] - rows[ref][c][k] for c in ids])
                              for k in ('cp_s', 'cp_e', 'rec_s', 'rec_e')}
    out[name] = blk
    print(name, blk['n'], {a: (blk[a]['cp_s'][0], blk[a]['cp_e'][0], blk[a]['rec_s'][0], blk[a]['rec_e'][0])
                          for a in ('Direct', 'SelfConsistency', 'MechSight_Auto')},
          'Auto-Direct', {k: v for k, v in blk['Auto-Direct'].items()})
json.dump(out, open(sys.argv[1], 'w'), indent=1)
