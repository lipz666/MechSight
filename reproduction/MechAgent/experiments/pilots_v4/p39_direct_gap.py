"""P39: single-component ablation (review request, 2026-11-06): gap completion on the Direct path alone.

For each of the 244 confirmation reactions whose Direct path contains a flagged gap (M.gaps: >=3 heavy-atom bond
changes, atom-conservation failure or unparsable step), the frozen Stage-2 completion prompt and Stage-3 revision
prompt are applied to the Direct path only (no candidate pool, one round). Reactions without a flagged gap keep
the Direct answer. gemini-3.8-flash on the 117 gateway. Post hoc, not pre-registered.

python p39_direct_gap.py --phase run|analyze [--workers 200]
"""
from __future__ import annotations

import argparse
import json
import os

import common4 as K
import failover as F
import p11_mechagent as M

for ep in F.ENDPOINTS:
    ep['model'] = 'gemini-3.8-flash-high' if 'vps' in ep['base_url'] else 'gemini-3.8-flash'


def direct(ds, cid):
    return K.parse_mech(K.cached(f'{ds}/base_default', cid))


def flagged(ds, case):
    m = direct(ds, case['id'])
    return bool(m) and bool(M.gaps(case, m))


def job(ds, case):
    cid = case['id']
    m = direct(ds, cid)
    text = M.render(case, [m])
    comp = K.call(f'{ds}/DG_comp', cid, M.COMPLETE.format(cands=text, **M.fields(case)), temperature=0.2)
    if comp is None:
        return 'error'
    st = M.checked_states(case, comp)
    if not st:
        return 'nostates'
    states = '\n'.join(f'- ({loc}) {s}' for loc, s in st)
    rev = K.call(f'{ds}/DG_rev', cid, M.REVISE.format(cands=text, states=states, **M.fields(case)), temperature=0.2)
    return 'ok' if rev else 'error'


def analyze(sets):
    rep, pooled = {}, {}
    for ds, cases in sets.items():
        rows, sub = {}, {}
        for cid, case in cases.items():
            if K.cached(f'{ds}/V5_rev1', cid) is None or K.cached(f'{ds}/base_default', cid) is None:
                continue
            fl = flagged(ds, case)
            dg = K.parse_mech(K.cached(f'{ds}/DG_rev', cid)) if fl else None
            ans = {'Direct': direct(ds, cid), 'Direct_gap_only': dg or direct(ds, cid),
                   'Pool_revision_no_gap': K.parse_mech(K.cached(f'{ds}/M_ms_none', cid)),
                   'MechSight_Auto': K.parse_mech(M.final(ds, cid, 2))}
            for a, m in ans.items():
                s = {'all_present': K.score(case, m)['all_present']}
                rows.setdefault(a, []).append((cid, s))
                pooled.setdefault(a, []).append((f'{ds}:{cid}', s))
                if fl:
                    sub.setdefault(a, []).append((cid, s))
                    pooled.setdefault(f'{a}@flagged', []).append((f'{ds}:{cid}', s))
        blk = lambda r: {a: K.summarize(v, ['all_present']) for a, v in r.items()}
        dif = lambda r: {f'{h}-{l}': K.paired(r[h], r[l], 'all_present') for h, l in
                         [('Direct_gap_only', 'Direct'), ('MechSight_Auto', 'Direct_gap_only'),
                          ('Pool_revision_no_gap', 'Direct'), ('MechSight_Auto', 'Direct')]}
        rep[ds] = {'n': len(rows['Direct']), 'n_flagged': len(sub.get('Direct', [])), 'all': blk(rows), 'all_diffs': dif(rows),
                   'flagged': blk(sub) if sub else {}, 'flagged_diffs': dif(sub) if sub else {}}
    allp = {a: v for a, v in pooled.items() if '@' not in a}
    flp = {a.split('@')[0]: v for a, v in pooled.items() if '@' in a}
    blk = lambda r: {a: K.summarize(v, ['all_present']) for a, v in r.items()}
    dif = lambda r: {f'{h}-{l}': K.paired(r[h], r[l], 'all_present') for h, l in
                     [('Direct_gap_only', 'Direct'), ('MechSight_Auto', 'Direct_gap_only'),
                      ('Pool_revision_no_gap', 'Direct'), ('MechSight_Auto', 'Direct')]}
    rep['pooled'] = {'n': len(allp['Direct']), 'n_flagged': len(flp.get('Direct', [])), 'all': blk(allp),
                     'all_diffs': dif(allp), 'flagged': blk(flp), 'flagged_diffs': dif(flp)}
    (K.OUT / 'SUMMARY_DIRECT_GAP.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep['pooled'], indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['run', 'analyze'], required=True)
    ap.add_argument('--workers', type=int, default=200)
    cli = ap.parse_args()
    sets = M.sets(('T3_confirm', 'gold'))
    if cli.phase == 'run':
        assert os.environ.get('FALLBACK_API_KEY'), 'FALLBACK_API_KEY required'
        jobs = [(ds, c) for ds, cs in sets.items() for c in cs.values()
                if K.cached(f'{ds}/base_default', c['id']) is not None and flagged(ds, c)]
        print(json.dumps({ds: sum(1 for d, _ in jobs if d == ds) for ds in sets}), flush=True)
        K.run_pool(job, jobs, cli.workers)
    analyze(sets)


if __name__ == '__main__':
    main()
