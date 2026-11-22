"""P30: the fixed 172-paper literature set re-evaluated on the argon-build gold standard (p29, PDF incl. schemes).

The model reaction itself is re-extracted, so every arm reruns on the new inputs under data set 'lit172g':
  Direct (base_default) | pool samp1-4 + enum3 | Auto = frozen v5 (r1, r2) | Direct+clues (O_llm)
  | Guided v7 literature adaptation (p28 functions: clue paths, 4 hypotheses, 3 votes, revision + retry).
Inputs to prediction: reactants + catalysts, products, conditions, observed clues. Gold structures enter scoring only.

Scoring (per paper, recall of gold substrate intermediates):
  strict   sigma-graph keys (K.species_keys), as in all earlier literature tables
  relaxed  metal atoms deleted, ligand-only fragments dropped (p29.substrate_fragments), on gold and prediction alike;
           a gold species counts when its largest relaxed fragment is predicted
  ceiling  union of all outputs/candidates/hypotheses (what any selection step could reach)

python p30_lit172_gold.py --workers 200            (run all phases, then analyze)
python p30_lit172_gold.py --analyze-only
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import common4 as K
import p11_mechagent as M
import p24_v6 as V
import p25_guided7 as G7
import p28_lit172_v7 as G
import p29_gold_rebuild as GOLD
import p2_enumerate as P2
import p6_baselines as P6
import stepmap as SMAP

DS = 'lit172g'
RUN = K.OUT / 'lit172_gold_argon_20261105'
IDS = json.loads((K.OUT / 'lit172_v7_real_20261105' / 'MANIFEST.json').read_text())['ids']

# p28's Guided functions read their module globals: point them at the new data set and run directory.
G.DS, G.RUN, G.STATE = DS, RUN, RUN / 'status.json'


def records():
    out = {}
    for pid in IDS:
        f = GOLD.OUTD / 'gold2' / f'{pid}.json'
        if f.exists():
            out[pid] = json.loads(f.read_text())
    return out


def cases():
    cs = {}
    for pid, r in records().items():
        mr = r['record'].get('model_reaction') or {}
        lst = GOLD._lst
        ok = lambda s: GOLD._mol(s) is not None  # noqa: E731
        reac = [s for s in lst(mr.get('reactants')) + lst(mr.get('catalysts')) if ok(s)]
        prod = [s for s in lst(mr.get('products')) if ok(s)]
        clues = r['record'].get('clues') or []
        clues = [c if isinstance(c, str) else json.dumps(c, ensure_ascii=False) for c in lst(clues)]
        if not (r['reaction_ok'] and reac and prod and r['gold_intermediates']):
            continue
        cs[pid] = {'id': pid, 'src': 'lit', 'reactants': reac, 'products': prod,
                   'conditions': mr.get('conditions') or '', 'author': r['gold_intermediates'],
                   'author_frags': r['gold_fragments'],
                   'clues': '\n'.join(f'- {c}' for c in clues if c and c != 'null')}
    return cs


def job_pool(arm, case):
    return P2.job(DS, case) if arm == 'enum3' else P6.job(DS, arm, case)


def _largest(frags):
    from rdkit import Chem
    best, n = None, -1
    for f in frags:
        for part in f.split('.'):
            m = Chem.MolFromSmiles(part) or Chem.MolFromSmiles(part, sanitize=False)
            if m is not None and m.GetNumHeavyAtoms() > n:
                best, n = part, m.GetNumHeavyAtoms()
    return best


def _canon(s):
    from rdkit import Chem
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else s


def relaxed_pred(mech, refs):
    out = set()
    for st in mech or []:
        for f in GOLD.substrate_fragments(st.get('intermediate_smiles') or '', refs):
            out.update(_canon(p) for p in f.split('.'))
    return out


def outputs(cid):
    """Every mechanism produced for a paper (for the ceiling)."""
    tags = ['base_default', 'samp1', 'samp2', 'samp3', 'samp4', 'enum3', 'O_llm', 'G7L_clue1', 'G7L_clue2']
    ms = [K.parse_mech(K.cached(f'{DS}/{t}', cid)) for t in tags]
    ms += [K.parse_mech(M.final(DS, cid, 2)), K.parse_mech(G.guided_final(cid))]
    states = []
    for t in ['V5_comp1', 'V5_comp2'] + [f'G7L_hyp{k}' for k in range(G7.NHYP)]:
        a = K.cached(f'{DS}/{t}', cid) or ''
        states += [{'intermediate_smiles': s} for s in dict.fromkeys(s for _, s in M.checked_states(CS[cid], a))]
    return [m for m in ms if m] + ([states] if states else [])


def analyze(cases_):
    arms = {'Direct': lambda c: K.cached(f'{DS}/base_default', c), 'MechSight_Auto': lambda c: M.final(DS, c, 2),
            'Direct_clues': lambda c: K.cached(f'{DS}/O_llm', c), 'MechSight_Guided': G.guided_final}
    rec = records()
    rows = {lvl: {a: [] for a in arms} for lvl in ('strict', 'relaxed')}
    ceil = {'strict': [0, 0], 'relaxed': [0, 0]}
    per = []
    for cid, c in cases_.items():
        refs = GOLD.substrate_refs(rec[cid]['record'])
        gs = [K.species_keys(s) for s in c['author']]
        gr = [_canon(x) if (x := _largest(f)) else None for f in c['author_frags']]
        allm = outputs(cid)
        u_s = set().union(*[V.keys_of(m) for m in allm]) if allm else set()
        u_r = set().union(*[relaxed_pred(m, refs) for m in allm]) if allm else set()
        ceil['strict'][0] += sum(bool(k & u_s) for k in gs if k)
        ceil['strict'][1] += sum(1 for k in gs if k)
        ceil['relaxed'][0] += sum(x in u_r for x in gr if x)
        ceil['relaxed'][1] += sum(1 for x in gr if x)
        row = {'id': cid, 'n_gold': len(gs), 'has_clues': bool(c['clues'])}
        for a, f in arms.items():
            if a in ('Direct_clues', 'MechSight_Guided') and not c['clues']:
                continue
            m = K.parse_mech(f(cid))
            ps, pr = V.keys_of(m), relaxed_pred(m, refs)
            ks = [k for k in gs if k]
            hs = sum(bool(k & ps) for k in ks)
            rr = [x for x in gr if x]
            hr = sum(x in pr for x in rr)
            rows['strict'][a].append((cid, {'recall': hs / len(ks) if ks else 0.0, 'all_present': float(bool(ks) and hs == len(ks)),
                                            'parsed': float(bool(m))}))
            if rr:
                rows['relaxed'][a].append((cid, {'recall': hr / len(rr), 'all_present': float(hr == len(rr))}))
            row[a] = {'strict': hs, 'relaxed': hr}
        per.append(row)
    rep = {'n_papers': len(cases_), 'n_with_clues': sum(bool(c['clues']) for c in cases_.values()),
           'n_gold_species': sum(len(c['author']) for c in cases_.values()),
           'ceiling_any_output': {k: [v[0], v[1], round(v[0] / max(v[1], 1), 3)] for k, v in ceil.items()}}
    for lvl, rs in rows.items():
        clue = {cid for cid, _ in rs['MechSight_Guided']}
        sub = {a: [x for x in r if x[0] in clue] for a, r in rs.items()}
        met = ['recall', 'all_present']
        rep[lvl] = {'all_papers': {a: K.summarize(rs[a], met) for a in ('Direct', 'MechSight_Auto')},
                    'all_Auto_minus_Direct': {k: K.paired(rs['MechSight_Auto'], rs['Direct'], k) for k in met},
                    'paired_with_clues': {a: K.summarize(r, met) for a, r in sub.items()}}
        for hi, lo in [('MechSight_Auto', 'Direct'), ('Direct_clues', 'Direct'), ('MechSight_Guided', 'Direct'),
                       ('MechSight_Guided', 'Direct_clues'), ('MechSight_Guided', 'MechSight_Auto')]:
            rep[lvl]['paired_with_clues'][f'{hi}_minus_{lo}'] = {k: K.paired(sub[hi], sub[lo], k) for k in met}
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / 'SUMMARY_LIT172_GOLD_ARGON.json').write_text(json.dumps({'summary': rep, 'per_paper': per}, indent=1))
    print(json.dumps(rep, indent=1))
    return rep


CS = {}


def main():
    global CS
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=200)
    ap.add_argument('--analyze-only', action='store_true')
    cli = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    os.environ['STEPMAP_REQ'] = str(K.OUT / 'stepmap_lit172g_requests.jsonl')
    os.environ['STEPMAP_CACHE'] = str(K.OUT / 'stepmap_lit172g_cache.json')
    SMAP.REQ, SMAP.CACHE = Path(os.environ['STEPMAP_REQ']), Path(os.environ['STEPMAP_CACHE'])
    CS = cases()
    print(json.dumps({'gold_records': len(records()), 'usable_cases': len(CS),
                      'with_clues': sum(bool(c['clues']) for c in CS.values())}), flush=True)
    if not cli.analyze_only:
        assert os.environ.get('FALLBACK_API_KEY'), 'FALLBACK_API_KEY required'
        cs = list(CS.values())
        w = cli.workers
        G.phase('auto_pool', job_pool, [(a, c) for a in ('base_default', 'samp1', 'samp2', 'samp3', 'samp4', 'enum3')
                                         for c in cs], w)
        G.mapping(CS)
        G.phase('auto_round1', M.job_r1, [(DS, c) for c in cs], w)
        G.mapping(CS)
        G.phase('auto_round2', M.job_r2, [(DS, c) for c in cs], w)
        guided = [c for c in cs if c['clues']]
        G.phase('guided_clue_paths', G.job_clue, [(c, k) for c in guided for k in range(3)], w)
        G.phase('guided_hypotheses', G.job_hyp, [(c, k) for c in guided for k in range(G7.NHYP)], w)
        G.phase('guided_votes', G.job_vote, [(c, k) for c in guided for k in range(G7.NVOTE)], w)
        G.phase('guided_revision', G.job_revise, [(c,) for c in guided], w)
    analyze(CS)


if __name__ == '__main__':
    main()
