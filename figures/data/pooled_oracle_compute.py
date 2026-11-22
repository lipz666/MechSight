import sys, json
sys.path.insert(0, '.')
import common4 as K, p11_mechagent as M, p14_confirm as P, p16_supplement_g38 as S
rows = {a: [] for a in ('Direct', 'MechAgent', 'O_hid', 'O_vis')}
for ds in ('T3_confirm', 'gold'):
    for cid, case in M.sets((ds,))[ds].items():
        if K.cached(f'{ds}/V5_rev1', cid) is None or K.cached(f'{ds}/base_default', cid) is None:
            continue
        if ds == 'T3_confirm':
            a = P.answers(ds, cid, case)
            ans = {'Direct': a['Direct'], 'MechAgent': a['MechAgent'], 'O_hid': a['Oracle_hidden'], 'O_vis': a['Oracle_visible']}
        else:
            hid, vis = S.gold_gt_intermediates(case)
            base = K.parse_mech(K.cached('gold/base_default', cid))
            ma = P.answers(ds, cid, case)['MechAgent']
            oh = K.parse_mech(K.cached('gold/O_hid', cid)) if hid else base
            ov = (K.parse_mech(K.cached('gold/O_vis', cid)) if vis else base) if hid else base
            ans = {'Direct': base, 'MechAgent': ma, 'O_hid': oh, 'O_vis': ov}
        for k, m in ans.items():
            rows[k].append((f'{ds}:{cid}', {'all_present': K.score(case, m)['all_present']}))
out = {k: K.summarize(v, ['all_present']) for k, v in rows.items()}
for k in ('O_hid', 'O_vis', 'MechAgent'):
    out[f'{k}_minus_Direct'] = K.paired(rows[k], rows['Direct'], 'all_present')
out['MechAgent_minus_O_hid'] = K.paired(rows['MechAgent'], rows['O_hid'], 'all_present')
print(json.dumps(out))
