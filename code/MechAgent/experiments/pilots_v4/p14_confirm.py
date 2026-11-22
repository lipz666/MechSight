"""P14: pre-registered confirmation of Mech-Agent v5 (REGISTRATION_v5_confirm.json).

Sets: T3_confirm (48) + oMe-Gold (196). Arms: Direct (base_default), CoT (base_cot), Self-refine
(SR2), Self-consistency over the same pool (MBR consensus, P3), M_ms_none, Mech-Agent v5;
Oracle-hidden / Oracle-visible on T3_confirm. Writes SUMMARY_P14_confirm.json.
"""
from __future__ import annotations

import json

import common4 as K
import p3_selector as P3
import p10_memory as P10
import p11_mechagent as M
import p12_controls as C
import p15_relabel as RL

ARMS = ['Direct', 'CoT', 'SelfRefine', 'SelfConsistency', 'M_ms_none', 'MechAgent', 'Oracle_hidden', 'Oracle_visible']


def answers(ds, cid, case):
    base = K.cached(f'{ds}/base_default', cid)
    p = P10.pool(ds, cid)
    sc = P3.consensus(p, list(p)) if len(p) >= 2 else None
    out = {'Direct': K.parse_mech(base), 'CoT': K.parse_mech(K.cached(f'{ds}/base_cot', cid)),
           'SelfRefine': K.parse_mech(K.cached(f'{ds}/SR2', cid)) or K.parse_mech(K.cached(f'{ds}/SR1', cid)),
           'SelfConsistency': p.get(sc) if sc else K.parse_mech(base),
           'M_ms_none': K.parse_mech(K.cached(f'{ds}/M_ms_none', cid)),
           'MechAgent': K.parse_mech(M.final(ds, cid, 2))}
    if ds == 'gold':  # format-only relabelling (DEVIATION_v5_confirm.md); species are unchanged
        for arm, rl in (('SelfRefine', 'SR2'), ('M_ms_none', 'M_ms_none'), ('MechAgent', 'MechAgent')):
            if K.cached(f'gold/RL_{rl}', cid) is not None:
                out[arm] = RL.relabelled(rl, cid) or out[arm]
    if 'pgl' in case:
        hid, vis = C.gt_intermediates(case)
        out['Oracle_hidden'] = K.parse_mech(K.cached(f'{ds}/O_hid', cid)) if hid else out['Direct']
        out['Oracle_visible'] = (K.parse_mech(K.cached(f'{ds}/O_vis', cid)) if vis else out['Direct']) if hid else out['Direct']
    return out


def main():
    sets = M.sets(('T3_confirm', 'gold'))
    rep, pooled = {}, {}
    for ds, cases in sets.items():
        rows, strata, missing = {}, {}, 0
        for cid, case in cases.items():
            if K.cached(f'{ds}/V5_rev1', cid) is None or K.cached(f'{ds}/base_default', cid) is None:
                missing += 1
                continue
            for arm, m in answers(ds, cid, case).items():
                s = K.score(case, m)
                rows.setdefault(arm, []).append((cid, s))
                if arm in ('Direct', 'SelfRefine', 'SelfConsistency', 'MechAgent', 'CoT', 'M_ms_none'):
                    pooled.setdefault(arm, []).append((f'{ds}:{cid}', s))
                if 'pgl' in case:
                    h = s['n_hidden_events']
                    b = '0' if h == 0 else '1-2' if h <= 2 else '>=3'
                    strata.setdefault(b, {}).setdefault(arm, []).append(s['all_present'])
        keys = ['all_present', 'in_order_frac'] + (['S_partial', 'S_total', 'in_order_full'] if ds == 'gold' else [])
        block = {'missing': missing, **{a: K.summarize(r, keys) for a, r in rows.items()}}
        for a in rows:
            if a != 'Direct':
                block[f'{a}_minus_Direct'] = K.paired(rows[a], rows['Direct'], 'all_present')
        for a in ('Direct', 'CoT', 'SelfRefine', 'SelfConsistency', 'M_ms_none'):
            block[f'MechAgent_minus_{a}'] = K.paired(rows['MechAgent'], rows[a], 'all_present')
        if ds == 'gold':
            for a in ('Direct', 'CoT', 'SelfRefine', 'SelfConsistency'):
                block[f'MechAgent_minus_{a}_S_partial'] = K.paired(rows['MechAgent'], rows[a], 'S_partial')
        if strata:
            block['strata_all_present'] = {b: {a: [len(v), round(sum(v) / len(v), 3)] for a, v in d.items()}
                                           for b, d in sorted(strata.items())}
        rep[ds] = block
    rep['pooled_primary'] = {a: K.summarize([(c, {'all_present': m['all_present']}) for c, m in r], ['all_present'])
                             for a, r in pooled.items()}
    for a in ('Direct', 'CoT', 'SelfRefine', 'SelfConsistency', 'M_ms_none'):
        rep['pooled_primary'][f'MechAgent_minus_{a}'] = K.paired(pooled['MechAgent'], pooled[a], 'all_present')
    (K.OUT / 'SUMMARY_P14_confirm.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
