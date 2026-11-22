"""P37b: wet-lab-only re-run of one development paper used as the running example in the figures (2026-11-08).

PMC13321810 (Ir-sensitised dearomative [2+2], Chem. Sci. 2026) was run with all clue lines, including the authors'
DFT line (classify label: computational, line 9), and the four hypotheses that formed the >=3/7 anchor quote those
DFT numbers. This re-runs every clue-dependent step with the computational line removed (p36 nodft_clues), exactly
as p37_nodft.py did for the confirmation papers: Direct + Clues (O_llm), two further clue-conditioned samples,
four hypotheses, anchored revision (G71) and insertion (A1). Clue-free Auto parts are reused from lit172g.
Outputs under OUT/lit172g_nodft/ and OUT/nodft_case_20261108/SUMMARY.json. Post hoc, not pre-registered.

FALLBACK_API_KEY / OPENAI_API_KEY must be set in the environment.
python p37b_case_nodft.py [--cid PMC13321810] [--analyze-only]
"""
from __future__ import annotations

import argparse
import json
import os

import common4 as K  # noqa: I001
import failover as F
import p11_mechagent as M
import p25_guided7 as G7
import p28_lit172_v7 as G
import p29_gold_rebuild as GOLD
import p30_lit172_gold as P
import p31_guided71 as T
import p34_pilot36 as X
import p36_clue_audit as A

for ep in F.ENDPOINTS:  # same model as the original development run
    ep['model'] = 'gemini-3.8-flash-high' if 'vps' in ep['base_url'] else 'gemini-3.8-flash'

NDS = 'lit172g_nodft'
RUN = K.OUT / 'nodft_case_20261108'


def job_g71n(case):
    cid = case['id']
    auto = K.parse_mech(M.final(P.DS, cid, 2))
    cands = ([auto] if auto else []) + M.cand_list(P.DS, cid)[:4] + G.clue_paths(cid)
    states = [s for tag in ('V5_comp1', 'V5_comp2') for _, s in M.checked_states(case, K.cached(f'{P.DS}/{tag}', cid) or '')]
    p = T.REVISE71.format(obs=G.obs(case), nsrc=G7.NHYP + 3, consensus=T.consensus_lines(G.proposals(case), T.TH71),
                          cand_note=' (candidate 1 is the Auto final path)', cands=M.render(case, cands),
                          states='\n'.join(dict.fromkeys(states)) or '- none', **M.fields(case))
    return 'ok' if K.call(f'{NDS}/G71_rev', cid, p) else 'error'


def guided(ds, cid):
    a1 = K.cached(f'{ds}/A1_place', cid)
    return a1 if a1 is not None else K.cached(f'{ds}/G71_rev', cid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cid', default='PMC13321810')
    ap.add_argument('--analyze-only', action='store_true')
    cli = ap.parse_args()
    T.configure('dev')
    RUN.mkdir(parents=True, exist_ok=True)
    cs = P.cases(); P.CS = cs
    c0 = cs[cli.cid]
    case = dict(c0, clues=A.nodft_clues(cli.cid, c0))
    assert case['clues'] and case['clues'] != c0['clues'], 'no computational line removed'
    (RUN / 'clues_nodft.txt').write_text(case['clues'])
    if not cli.analyze_only:
        assert os.environ.get('FALLBACK_API_KEY') or os.environ.get('OPENAI_API_KEY'), 'gateway key required in env'
        G.DS, G.RUN, G.STATE = NDS, RUN, RUN / 'status.json'
        X.PDS = NDS
        G.phase('clue_samples', G.job_clue, [(case, k) for k in range(3)], 8)
        G.phase('hypotheses', G.job_hyp, [(case, k) for k in range(G7.NHYP)], 8)
        G.phase('revision', job_g71n, [(case,)], 1)
        G.phase('insertion', X.job, [('A1', case)], 1)
        G.DS = P.DS
    # proposals support (what the anchor step saw) and scores
    G.DS = NDS
    support = {s: n for s, n in G.proposals(case).items()} if hasattr(G.proposals(case), 'items') else G.proposals(case)
    G.DS = P.DS
    refs = GOLD.substrate_refs(P.records()[cli.cid]['record'])
    out = {'cid': cli.cid, 'proposal_support_nodft': {k: v for k, v in dict(support).items()},
           'arms': {}}
    for arm, txt in (('Direct', K.cached(f'{P.DS}/base_default', cli.cid)),
                     ('Auto', M.final(P.DS, cli.cid, 2)),
                     ('Direct+Clues_all', K.cached(f'{P.DS}/O_llm', cli.cid)),
                     ('Guided_all', guided(P.DS, cli.cid)),
                     ('Direct+Clues_wetlab', K.cached(f'{NDS}/O_llm', cli.cid)),
                     ('Guided_wetlab', guided(NDS, cli.cid))):
        mech = K.parse_mech(txt)
        s = T.score(mech, c0, refs)
        out['arms'][arm] = {**{k: s.get(k) for k in ('recall', 'n_pred', 'hits')},
                            'states': [st.get('intermediate_smiles') for st in (mech or [])]}
    (RUN / 'SUMMARY.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
