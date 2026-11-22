"""P37: confirmation papers re-run with computational content removed from the clues (review request, 2026-11-06).

Papers: the confirmation papers whose clues contain computational content according to the gemini-4-argon-build
line classification (p36_clue_audit.py classify). Their clues are replaced by the experimental-only version
(computational lines dropped, mixed lines rewritten). Everything that depends on the clues is re-run with the
frozen prompts: Direct + Clues (O_llm), the two further clue-conditioned samples, the four hypotheses, the anchored
revision and the insertion step. The clue-free MechSight-Auto parts are reused from the original run.
gemini-3.8-flash on the 117 gateway; outputs under OUT/litnew_g_nodft/. Post hoc, not pre-registered.

python p37_nodft.py [--workers 300] [--analyze-only]
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

for ep in F.ENDPOINTS:  # p34_pilot36 sets the 3.6 pilot model on import; this run is 3.8 only
    ep['model'] = 'gemini-3.8-flash-high' if 'vps' in ep['base_url'] else 'gemini-3.8-flash'
assert all(ep['model'].startswith('gemini-3.8-flash') for ep in F.ENDPOINTS)

NDS = 'litnew_g_nodft'
RUN = K.OUT / 'nodft_confirm_20261106'


def nodft_cases():
    T.configure('test')
    cs = P.cases()
    P.CS = cs
    out = {}
    for c in cs.values():
        if not c['clues'] or not any(K.species_keys(s) for s in c['author']):
            continue
        f = A.OUTD / 'classify' / f"{c['id']}.json"
        lab = A.parse(json.loads(f.read_text())['answer']) if f.exists() else None
        if lab is None:
            continue
        if any(isinstance(x, dict) and x.get('label') in ('computational', 'mixed') for x in lab):
            out[c['id']] = dict(c, clues=A.nodft_clues(c['id'], c))
    return cs, out


def job_g71n(case):
    """T.job_g71 with clue-dependent inputs from NDS and the output written to NDS (Auto parts from P.DS)."""
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


def analyze(cs, nd):
    rec = P.records()
    rows = {}
    for cid, c in cs.items():
        if not c['clues'] or not any(K.species_keys(s) for s in c['author']):
            continue
        refs = GOLD.substrate_refs(rec[cid]['record'])
        inset = cid in nd
        arms = {'Direct': K.cached(f'{P.DS}/base_default', cid),
                'Direct_clues': K.cached(f'{P.DS}/O_llm', cid), 'MechSight_Guided': guided(P.DS, cid),
                'Direct_clues_nodft': K.cached(f'{NDS}/O_llm', cid) if inset else K.cached(f'{P.DS}/O_llm', cid),
                'MechSight_Guided_nodft': guided(NDS, cid) if inset else guided(P.DS, cid)}
        for a, txt in arms.items():
            s = T.score(K.parse_mech(txt), c, refs)
            s['complete'] = float(s['recall'] >= 0.999)
            rows.setdefault(a, []).append((cid, s))
            if inset:
                rows.setdefault(a + '@nodft_papers', []).append((cid, s))
    keys = ['recall', 'complete', 'f1', 'relaxed_recall', 'n_pred']
    pairs = [('MechSight_Guided_nodft', 'Direct_clues_nodft'), ('MechSight_Guided', 'Direct_clues'),
             ('Direct_clues_nodft', 'Direct'), ('MechSight_Guided_nodft', 'MechSight_Guided'),
             ('Direct_clues_nodft', 'Direct_clues')]
    out = {}
    for scope, suffix in (('all_271', ''), ('papers_with_computational_clues', '@nodft_papers')):
        r = {a.replace(suffix, ''): v for a, v in rows.items() if (a.endswith(suffix) if suffix else '@' not in a)}
        out[scope] = {'n': len(r['Direct']), 'arms': {a: K.summarize(v, keys) for a, v in r.items()},
                      'diffs': {f'{h}-{l}': {k: K.paired(r[h], r[l], k) for k in ('recall', 'complete', 'f1')}
                                for h, l in pairs}}
    (RUN / 'SUMMARY_NODFT.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=300)
    ap.add_argument('--analyze-only', action='store_true')
    cli = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    cs, nd = nodft_cases()
    print(json.dumps({'papers_rerun': len(nd)}), flush=True)
    if not cli.analyze_only:
        assert os.environ.get('FALLBACK_API_KEY'), 'FALLBACK_API_KEY required'
        G.DS = NDS  # clue-dependent caches (O_llm, G7L_clue*, G7L_hyp*) are read and written under NDS
        G.RUN, G.STATE = RUN, RUN / 'status.json'
        X.PDS = NDS  # A1 reads NDS/G71_rev and writes NDS/A1_place
        lst = list(nd.values())
        w = cli.workers
        G.phase('nodft_clue_samples', G.job_clue, [(c, k) for c in lst for k in range(3)], w)
        G.phase('nodft_hypotheses', G.job_hyp, [(c, k) for c in lst for k in range(G7.NHYP)], w)
        G.phase('nodft_revision', job_g71n, [(c,) for c in lst], w)
        G.phase('nodft_insertion', X.job, [('A1', c) for c in lst], w)
        G.DS = P.DS
    analyze(cs, nd)


if __name__ == '__main__':
    main()
