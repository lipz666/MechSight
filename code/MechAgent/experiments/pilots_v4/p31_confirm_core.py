"""Confirmation run on litnew_g (271 papers): the four Table 2 arms, with MechSight-Guided = A1 (frozen 2026-11-06).

Arms: Direct, MechSight-Auto, Direct+Clues, MechSight-Guided.
MechSight-Guided = v7.1 revision (p31_guided71.job_g71, unchanged) followed by the A1 insertion step
(p34_pilot36.job('A1'), chosen on the lit172g dev set): every proposed intermediate supported by >= 3 of the 7
proposal sources and missing from the revision is placed into the mechanism by one call; papers with nothing
missing keep the revision. Model: gemini-3.8-flash throughout. Primary: Guided - Direct+Clues recall.

python p31_confirm_core.py [--workers 400] [--analyze-only] [--batch 1|2]
"""
import argparse
import os

import common4 as K  # noqa: I001  (puts failover on sys.path)
import failover as F
import p11_mechagent as M
import p25_guided7 as G7
import p28_lit172_v7 as G
import p30_lit172_gold as P
import p31_guided71 as T
import p34_pilot36 as X

for ep in F.ENDPOINTS:  # p34_pilot36 defaults to the 3.6 pilot model; the confirmation is 3.8 only
    ep['model'] = 'gemini-3.8-flash-high' if 'vps' in ep['base_url'] else 'gemini-3.8-flash'
assert all(ep['model'].startswith('gemini-3.8-flash') for ep in F.ENDPOINTS)


def guided(cid):
    a1 = K.cached(f'{P.DS}/A1_place', cid)
    return a1 if a1 is not None else K.cached(f'{P.DS}/G71_rev', cid)


T.ARMS = {'Direct': T.ARMS['Direct'], 'MechSight_Auto': T.ARMS['MechSight_Auto'],
          'Direct_clues': T.ARMS['Direct_clues'], 'MechSight_Guided': guided}
T.PRIMARY = ('MechSight_Guided', 'Direct_clues')
T.COMPARISONS = [T.PRIMARY, ('MechSight_Guided', 'Direct'), ('MechSight_Guided', 'MechSight_Auto'),
                 ('Direct_clues', 'Direct'), ('MechSight_Auto', 'Direct')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=400)
    ap.add_argument('--analyze-only', action='store_true')
    ap.add_argument('--batch', type=int, default=0)
    cli = ap.parse_args()
    run = T.configure('test')
    X.PDS = P.DS  # A1 reads litnew_g/G71_rev and writes litnew_g/A1_place
    cs = P.cases()
    P.CS = cs
    if not cli.analyze_only:
        assert os.environ.get('FALLBACK_API_KEY'), 'FALLBACK_API_KEY required'
        w, lst = cli.workers, [c for c in cs.values() if c['clues']]
        G.phase('auto_pool', P.job_pool, [(a, c) for a in ('base_default', 'samp1', 'samp2', 'samp3', 'samp4',
                                                              'enum3') for c in cs.values()], w)
        G.mapping(cs)
        G.phase('auto_round1', M.job_r1, [(P.DS, c) for c in cs.values()], w)
        G.mapping(cs)
        G.phase('auto_round2', M.job_r2, [(P.DS, c) for c in cs.values()], w)
        G.phase('clue_samples', G.job_clue, [(c, k) for c in lst for k in range(3)], w)  # O_llm + 2 clue paths
        G.phase('guided_hypotheses', G.job_hyp, [(c, k) for c in lst for k in range(G7.NHYP)], w)
        G.phase('guided_v71_revision', T.job_g71, [(c,) for c in lst], w)
        G.phase('guided_a1_insertion', X.job, [('A1', c) for c in lst], w)
    T.analyze(cs, run, cli.batch or None)


if __name__ == '__main__':
    main()
