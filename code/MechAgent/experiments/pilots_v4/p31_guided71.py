"""P31: Guided v7.1 = v7 literature adaptation with consensus-anchored revision, plus compute-matched baselines.

Diagnosis on lit172g (dev): the 4 hypotheses + 3 clue paths jointly contain 65% of the gold intermediates, but the
v7 revision only received the single voted species (22.9% recall) and ended at 52.5%.
v7.1 change (one prompt, no new information):
  - no vote; the revision receives every proposed intermediate supported by >= 3 of the 7 independent proposal
    sources (4 hypothesis sets + 3 observation-guided mechanisms), ranked by support, and must include each one
    unless an elementary-step or observation argument rules it out.
Compute-matched baselines (same revision prompt, same aggregation rule, only Direct+clues samples):
  SC7   O_llm + G7L_clue1..6       consensus support >= 3/7,  revision over the 7 samples
  SC15  O_llm + G7L_clue1..14      consensus support >= 7/15, revision over the first 7 samples
Metrics per paper (species de-duplicated by sigma-graph keys): recall, precision, F1 (strict); relaxed recall.

python p31_guided71.py --set dev  --workers 400     (lit172g, 165 papers; development)
python p31_guided71.py --set test --workers 400     (litnew_g: batch 1 = 58 unused lit150 PDFs, batch 2 = new
                                                     2026 papers; pre-registered confirmation, REGISTRATION_g71_confirm.json)
python p31_guided71.py --set test --analyze-only --batch 1   (interim, descriptive only)
python p31_guided71.py --set dev --analyze-only
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import common4 as K
import p11_mechagent as M
import p24_v6 as V
import p25_guided7 as G7
import p28_lit172_v7 as G
import p29_gold_rebuild as GOLD
import p30_lit172_gold as P
import stepmap as SMAP

NSC = (7, 15)
TH71 = 3
TH_SC = {7: 3, 15: 7}

REVISE71 = (
    'Reactants and reagents (SMILES): {reactants}\nProducts (SMILES): {products}\n'
    'Reaction conditions: {conditions}\n\nExperimental observations:\n{obs}\n\n'
    'Intermediates corroborated by several independent proposals (support = number of the {nsrc} independent '
    'proposal sets that contain the structure; calculated formula in brackets). These are inferences to assess '
    'against the observations, not experimental facts:\n{consensus}\n\n'
    'Candidate mechanisms{cand_note} (flagged steps change several bonds):\n{cands}\n\n'
    'Checked intermediates proposed for flagged steps:\n{states}\n\n'
    'Write the final mechanism consistent with the actual observations. Include every corroborated intermediate '
    'in its proper place unless an elementary-step or observation argument rules it out; for each one you exclude, '
    'state the reason in one sentence before the answer. Preserve the other sound intermediates, add the species '
    'needed so that every step is elementary, and do not treat a trapped adduct as a native intermediate unless '
    'justified. Return the final mechanism between [ANSWER] and [/ANSWER] as a JSON list of steps, each '
    '{{"step": n, "type": ..., "subtype": ..., "intermediate_smiles": ...}}.'
)


def configure(which):
    if which == 'test':
        P.DS, P.RUN = 'litnew_g', K.OUT / 'litnew_confirm_g71'
        P.IDS = [p for b in json.loads((P.RUN / 'MANIFEST.json').read_text())['batches'] for p in b['ids']]
    run = P.RUN if which == 'test' else K.OUT / 'g71_dev_20261106'
    G.DS, G.RUN, G.STATE = P.DS, run, run / 'status.json'
    os.environ['STEPMAP_REQ'] = str(K.OUT / f'stepmap_{P.DS}_requests.jsonl')
    os.environ['STEPMAP_CACHE'] = str(K.OUT / f'stepmap_{P.DS}_cache.json')
    SMAP.REQ, SMAP.CACHE = Path(os.environ['STEPMAP_REQ']), Path(os.environ['STEPMAP_CACHE'])
    run.mkdir(parents=True, exist_ok=True)
    return run


def sc_tags(n):
    return ['O_llm'] + [f'G7L_clue{k}' for k in range(1, n)]


def consensus_lines(support, th):
    return '\n'.join(f'- {s} [support {n}; {V._formula(s)}]' for s, n in support.most_common() if n >= th) or '- none'


def job_g71(case):
    cid = case['id']
    auto = K.parse_mech(M.final(P.DS, cid, 2))
    cands = ([auto] if auto else []) + M.cand_list(P.DS, cid)[:4] + G.clue_paths(cid)
    states = [s for tag in ('V5_comp1', 'V5_comp2')
              for _, s in M.checked_states(case, K.cached(f'{P.DS}/{tag}', cid) or '')]
    p = REVISE71.format(obs=G.obs(case), nsrc=G7.NHYP + 3, consensus=consensus_lines(G.proposals(case), TH71),
                        cand_note=' (candidate 1 is the Auto final path)', cands=M.render(case, cands),
                        states='\n'.join(dict.fromkeys(states)) or '- none', **M.fields(case))
    return 'ok' if K.call(f'{P.DS}/G71_rev', cid, p) else 'error'


def sc_support(case, n):
    support = Counter()
    for t in sc_tags(n):
        m = K.parse_mech(K.cached(f'{P.DS}/{t}', case['id']))
        if m:
            text = '<states>\n' + '\n'.join(dict.fromkeys(s['intermediate_smiles'] for s in m)) + '\n</states>'
            for _, smi in M.checked_states(case, text):
                support[smi] += 1
    return support


def job_sc(case, n):
    cid = case['id']
    paths = [m for t in sc_tags(7) if (m := K.parse_mech(K.cached(f'{P.DS}/{t}', cid)))]
    p = REVISE71.format(obs=G.obs(case), nsrc=n, consensus=consensus_lines(sc_support(case, n), TH_SC[n]),
                        cand_note='', cands=M.render(case, paths), states='- none', **M.fields(case))
    return 'ok' if K.call(f'{P.DS}/SC{n}_rev', cid, p) else 'error'


ARMS = {'Direct': lambda c: K.cached(f'{P.DS}/base_default', c),
        'Direct_clues': lambda c: K.cached(f'{P.DS}/O_llm', c),
        'MechSight_Auto': lambda c: M.final(P.DS, c, 2),
        'Guided_v7': G.guided_final,
        'SC7': lambda c: K.cached(f'{P.DS}/SC7_rev', c),
        'SC15': lambda c: K.cached(f'{P.DS}/SC15_rev', c),
        'Guided_v71': lambda c: K.cached(f'{P.DS}/G71_rev', c)}
PRIMARY = ('Guided_v71', 'Direct_clues')
COMPARISONS = [PRIMARY, ('Guided_v71', 'SC15'), ('Guided_v71', 'SC7'), ('Guided_v71', 'Guided_v7'),
               ('Guided_v71', 'Direct'), ('SC15', 'Direct_clues'), ('Guided_v7', 'Direct_clues'),
               ('MechSight_Auto', 'Direct')]
METRICS = ['recall', 'precision', 'f1', 'relaxed_recall', 'n_pred']


def species(mech):
    out = []
    for s in mech or []:
        k = K.species_keys(s.get('intermediate_smiles') or '')
        if k and not any(k & o for o in out):
            out.append(k)
    return out


def score(mech, case, refs):
    gs = [k for s in case['author'] if (k := K.species_keys(s))]
    pred = species(mech)
    r = sum(any(g & p for p in pred) for g in gs) / len(gs)
    pr = sum(any(p & g for g in gs) for p in pred) / len(pred) if pred else 0.0
    gr = [P._canon(x) if (x := P._largest(f)) else None for f in case['author_frags']]
    gr = [x for x in gr if x]
    rel = P.relaxed_pred(mech, refs)
    return {'recall': r, 'precision': pr, 'f1': 2 * r * pr / (r + pr) if r + pr else 0.0,
            'relaxed_recall': sum(x in rel for x in gr) / len(gr) if gr else r, 'n_pred': float(len(pred)),
            'parsed': float(bool(mech))}


def analyze(cs, run, batch=None):
    rec = P.records()
    if batch:
        keep = set(json.loads((run / 'MANIFEST.json').read_text())['batches'][batch - 1]['ids'])
        cs = {k: v for k, v in cs.items() if k in keep}
    rows = {a: [] for a in ARMS}
    for cid, c in cs.items():
        if not c['clues'] or not any(K.species_keys(s) for s in c['author']):
            continue
        refs = GOLD.substrate_refs(rec[cid]['record'])
        for a, f in ARMS.items():
            rows[a].append((cid, score(K.parse_mech(f(cid)), c, refs)))
    rep = {'n_papers': len(rows['Direct']), 'batch': batch or 'all', 'primary': f'{PRIMARY[0]} - {PRIMARY[1]} recall',
           'arms': {a: K.summarize(r, METRICS + ['parsed']) for a, r in rows.items()},
           'diffs': {f'{hi}_minus_{lo}': {k: K.paired(rows[hi], rows[lo], k) for k in METRICS}
                     for hi, lo in COMPARISONS}}
    (run / f"SUMMARY_G71{f'_batch{batch}' if batch else ''}.json").write_text(json.dumps({'summary': rep, 'per_paper': {a: r for a, r in rows.items()}},
                                                     indent=1))
    print(json.dumps(rep, indent=1))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', choices=['dev', 'test'], required=True)
    ap.add_argument('--workers', type=int, default=400)
    ap.add_argument('--analyze-only', action='store_true')
    ap.add_argument('--batch', type=int, default=0, help='analyze one manifest batch only (interim)')
    cli = ap.parse_args()
    run = configure(cli.set)
    cs = P.cases()
    P.CS = cs
    print(json.dumps({'set': cli.set, 'ds': P.DS, 'gold_records': len(P.records()), 'usable_cases': len(cs)}),
          flush=True)
    if not cli.analyze_only:
        assert os.environ.get('FALLBACK_API_KEY'), 'FALLBACK_API_KEY required'
        w, lst = cli.workers, [c for c in cs.values() if c['clues']]
        if cli.set == 'test':
            G.phase('auto_pool', P.job_pool, [(a, c) for a in ('base_default', 'samp1', 'samp2', 'samp3', 'samp4',
                                                                  'enum3') for c in cs.values()], w)
            G.mapping(cs)
            G.phase('auto_round1', M.job_r1, [(P.DS, c) for c in cs.values()], w)
            G.mapping(cs)
            G.phase('auto_round2', M.job_r2, [(P.DS, c) for c in cs.values()], w)
        G.phase('clue_samples', G.job_clue, [(c, k) for c in lst for k in range(max(NSC))], w)
        G.phase('guided_hypotheses', G.job_hyp, [(c, k) for c in lst for k in range(G7.NHYP)], w)
        if cli.set == 'test':
            G.phase('guided_v7_votes', G.job_vote, [(c, k) for c in lst for k in range(G7.NVOTE)], w)
            G.phase('guided_v7_revision', G.job_revise, [(c,) for c in lst], w)
        G.phase('sc_revisions', job_sc, [(c, n) for c in lst for n in NSC], w)
        G.phase('guided_v71_revision', job_g71, [(c,) for c in lst], w)
    analyze(cs, run, cli.batch or None)


if __name__ == '__main__':
    main()
