"""P19: ablation ladder on the confirmation sets (SI; post hoc, not part of the registration).

  L0 Direct                        base_default
  L1 Sampling + revision           pool = base_default + samp1..4 (no hidden-event enumeration),
                                   the same eliminate/repair call as M_ms_none (P10 NO_MEM)
  L2 + hidden-event enumeration    M_ms_none (pool adds the three enum3 paths)
  L3 + gap detection/completion    Mech-Agent v5
Writes SUMMARY_P19_LADDER.json.
"""
from __future__ import annotations

import argparse
import json

import common4 as K
import p3_selector as P3
import p10_memory as P10
import p11_mechagent as M
import p15_relabel as RL


def pool_noenum(ds, cid):
    p = {k: v for k, v in P10.pool(ds, cid).items() if not k.startswith('enum')}
    out, seen = {}, set()
    for n in P3.order(cid, p):
        key = tuple(M.canon(s.get('intermediate_smiles')) for s in p[n])
        if key not in seen:
            seen.add(key)
            out[n] = p[n]
    return out


def job(ds, case):
    cid = case['id']
    p = pool_noenum(ds, cid)
    if len(p) < 2:
        return 'skip'
    text = P3.render(list(p.values()))
    return 'ok' if K.call(f'{ds}/L1_noenum', cid, P10.MS.format(memory='', cands=text, instruction=P10.NO_MEM,
                                                                  **M.fields(case))) else 'error'


def analyze(sets):
    rep, pooled = {}, {}
    for ds, cases in sets.items():
        rows = {}
        for cid, case in cases.items():
            if K.cached(f'{ds}/V5_rev1', cid) is None:
                continue
            l1 = K.parse_mech(K.cached(f'{ds}/L1_noenum', cid)) or K.parse_mech(K.cached(f'{ds}/base_default', cid))
            arms = {'L0_Direct': K.parse_mech(K.cached(f'{ds}/base_default', cid)), 'L1_sampling_revision': l1,
                    'L2_plus_enumeration': K.parse_mech(K.cached(f'{ds}/M_ms_none', cid)),
                    'L3_MechSight': K.parse_mech(M.final(ds, cid, 2))}
            if ds == 'gold':
                if K.cached('gold/RL_M_ms_none', cid):
                    arms['L2_plus_enumeration'] = RL.relabelled('M_ms_none', cid) or arms['L2_plus_enumeration']
                if K.cached('gold/RL_MechAgent', cid):
                    arms['L3_MechSight'] = RL.relabelled('MechAgent', cid) or arms['L3_MechSight']
            for a, m in arms.items():
                s = K.score(case, m)
                rows.setdefault(a, []).append((cid, s))
                pooled.setdefault(a, []).append((f'{ds}:{cid}', {'all_present': s['all_present']}))
        block = {a: K.summarize(r, ['all_present']) for a, r in rows.items()}
        names = list(rows)
        for lo, hi in zip(names, names[1:]):
            block[f'{hi}_minus_{lo}'] = K.paired(rows[hi], rows[lo], 'all_present')
        rep[ds] = block
    names = list(pooled)
    rep['pooled'] = {a: K.summarize(r, ['all_present']) for a, r in pooled.items()}
    for lo, hi in zip(names, names[1:]):
        rep['pooled'][f'{hi}_minus_{lo}'] = K.paired(pooled[hi], pooled[lo], 'all_present')
    (K.OUT / 'SUMMARY_P19_LADDER.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['run', 'analyze'], required=True)
    ap.add_argument('--workers', type=int, default=200)
    cli = ap.parse_args()
    s = M.sets(('T3_confirm', 'gold'))
    if cli.phase == 'run':
        K.run_pool(job, [(ds, c) for ds, cs in s.items() for c in cs.values()], cli.workers)
    analyze(s)
