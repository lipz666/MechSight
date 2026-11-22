"""P31 / M2: does mechanism ability track literature exposure?

Same arms, prompts and gold pipeline as P30 (lit172g, 2026 papers, after the training cut-off), applied to the 244
in-scope old papers (2008-2019, lit_old/gold_scope_old.json, exposure tertiles by citations):
  Direct (base_default; inputs reactants+catalysts, products, conditions) and MechSight-Auto (frozen v5, r1+r2).
Primary metric: per-paper strict recall of gold substrate intermediates (sigma-graph keys). Relaxed is secondary.

Pre-registered analyses (manuscript/机理能力来源_预注册_v1.md):
  P1  old-high minus recent, Direct strict recall, regression-adjusted for difficulty covariates; TOST +/-10 pp
  P2  slope of Direct strict recall on log10(citations) within the 244 old papers, adjusted for year and covariates
  S1  same for MechSight-Auto, and Auto-Direct per stratum
  S2  Direct averaged over 5 samples (base_default + samp1-4)
CIs: 2000 paper-level bootstrap resamples, seed 20261106.

python p31_m2_exposure.py --workers 200
python p31_m2_exposure.py --analyze-only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

import common4 as K
import p11_mechagent as M
import p30_lit172_gold as P30
import stepmap as SMAP

LIT_OLD = K.HERE.parent / 'lit_old'
SCOPE = json.loads((LIT_OLD / 'gold_scope_old.json').read_text())
META = {p['pmcid']: p for p in SCOPE}
DS_OLD, DS_NEW = 'litoldg', 'lit172g'
GOLD_OLD, GOLD_NEW = K.OUT / 'lit_old_argon', K.OUT / 'lit_gold_argon'
RUN = K.OUT / 'm2_exposure_20261106'
LIT172_IDS = list(P30.IDS)
SEED = 20261106
TOST = 0.10


def load_cases(ds, gold_dir, ids):
    P30.DS, P30.IDS = ds, ids
    P30.GOLD.OUTD = gold_dir
    return P30.cases()


def covariates(case, rec):
    from rdkit import Chem
    heavy = 0
    for s in case['reactants']:
        m = Chem.MolFromSmiles(s)
        if m:
            heavy = max(heavy, m.GetNumHeavyAtoms())
    metal = any(a in ''.join(case['reactants']) for a in
                ['Pd', 'Ni', 'Cu', 'Co', 'Fe', 'Rh', 'Ir', 'Ru', 'Au', 'Ag', 'Pt', 'Mn', 'Cr', 'Ti', 'Zn', 'Sc', 'Ln'])
    return {'n_gold': len(case['author']), 'heavy': heavy, 'metal': float(metal),
            'n_reactants': len(case['reactants'])}


def recall(ds, arm_answer, case, rec):
    m = K.parse_mech(arm_answer)
    ps = P30.V.keys_of(m)
    ks = [k for k in (K.species_keys(s) for s in case['author']) if k]
    refs = P30.GOLD.substrate_refs(rec['record'])
    pr = P30.relaxed_pred(m, refs)
    rr = [x for f in case['author_frags'] if (x := P30._largest(f))]
    rr = [P30._canon(x) for x in rr]
    return (sum(bool(k & ps) for k in ks) / len(ks) if ks else 0.0,
            sum(x in pr for x in rr) / len(rr) if rr else float('nan'))


def rows_for(ds, gold_dir, cases, stratum_of):
    P30.GOLD.OUTD = gold_dir
    out = []
    for cid, c in cases.items():
        rec = json.loads((gold_dir / 'gold2' / f'{cid}.json').read_text())
        d = covariates(c, rec)
        d.update(id=cid, stratum=stratum_of(cid))
        for arm, ans in [('Direct', K.cached(f'{ds}/base_default', cid)), ('Auto', M.final(ds, cid, 2))]:
            d[f'{arm}_strict'], d[f'{arm}_relaxed'] = recall(ds, ans, c, rec)
        d['Direct5_strict'] = float(np.mean([recall(ds, K.cached(f'{ds}/{t}', cid), c, rec)[0]
                                             for t in ('base_default', 'samp1', 'samp2', 'samp3', 'samp4')]))
        if cid in META:
            d['log_cites'] = math.log10(1 + META[cid]['cites_gm'])
            d['year'] = META[cid]['year']
        out.append(d)
    return out


COV = ['n_gold', 'heavy', 'metal', 'n_reactants']


def ols(rows, y, x_names):
    rows = [r for r in rows if not math.isnan(r[y])]
    X = np.array([[1.0] + [float(r[n]) for n in x_names] for r in rows])
    Y = np.array([r[y] for r in rows])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return beta


def boot(rows, fn, n=2000):
    rng = random.Random(SEED)
    est = fn(rows)
    bs = []
    for _ in range(n):
        s = [rows[rng.randrange(len(rows))] for _ in rows]
        try:
            bs.append(fn(s))
        except Exception:
            continue
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return [round(100 * est, 2), round(100 * lo, 2), round(100 * hi, 2)]


def adj_diff(y, a='high', b='recent'):
    """Coefficient of the stratum-a indicator in OLS on {a, b} rows with difficulty covariates."""
    def fn(rows):
        sub = [dict(r, ind=float(r['stratum'] == a)) for r in rows if r['stratum'] in (a, b)]
        return ols(sub, y, ['ind'] + COV)[1]
    return fn


def slope(y):
    """Change in recall per 10x citations among old papers, adjusted for year and covariates."""
    def fn(rows):
        sub = [r for r in rows if r['stratum'] != 'recent']
        return ols(sub, y, ['log_cites', 'year'] + COV)[1]
    return fn


def verdict(ci):
    lo, hi = ci[1] / 100, ci[2] / 100
    if -TOST < lo and hi < TOST:
        return 'equivalent within +/-10 pp'
    if lo > 0 or hi < 0:
        return 'difference'
    return 'no difference detected (not equivalent)'


def analyze(rows):
    rep = {'n': {s: sum(r['stratum'] == s for r in rows) for s in ('low', 'mid', 'high', 'recent')}}
    rep['means'] = {s: {k: round(100 * float(np.nanmean([r[k] for r in rows if r['stratum'] == s])), 2)
                        for k in ('Direct_strict', 'Direct_relaxed', 'Direct5_strict', 'Auto_strict', 'Auto_relaxed')}
                    for s in ('low', 'mid', 'high', 'recent')}
    rep['balance_means'] = {s: {k: round(float(np.mean([r[k] for r in rows if r['stratum'] == s])), 2) for k in COV}
                            for s in ('low', 'mid', 'high', 'recent')}
    p1 = boot(rows, adj_diff('Direct_strict'))
    rep['P1_high_minus_recent_Direct_strict_adj'] = {'est_ci_pp': p1, 'verdict': verdict(p1)}
    p2 = boot(rows, slope('Direct_strict'))
    rep['P2_slope_per_10x_cites_Direct_strict'] = {'est_ci_pp': p2, 'verdict': verdict(p2)}
    rep['S1'] = {
        'high_minus_recent_Auto_strict_adj': boot(rows, adj_diff('Auto_strict')),
        'slope_Auto_strict': boot(rows, slope('Auto_strict')),
        'Auto_minus_Direct_by_stratum': {s: boot([r for r in rows if r['stratum'] == s],
                                                 lambda rs: float(np.mean([r['Auto_strict'] - r['Direct_strict']
                                                                           for r in rs])))
                                         for s in ('low', 'mid', 'high', 'recent')}}
    rep['S2'] = {'high_minus_recent_Direct5_adj': boot(rows, adj_diff('Direct5_strict')),
                 'slope_Direct5': boot(rows, slope('Direct5_strict'))}
    rep['secondary_relaxed'] = {'high_minus_recent_Direct_relaxed_adj': boot(rows, adj_diff('Direct_relaxed')),
                                'slope_Direct_relaxed': boot(rows, slope('Direct_relaxed'))}
    rep['low_minus_recent_Direct_strict_adj'] = boot(rows, adj_diff('Direct_strict', 'low'))
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / 'SUMMARY_M2.json').write_text(json.dumps({'summary': rep, 'rows': rows}, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=200)
    ap.add_argument('--analyze-only', action='store_true')
    cli = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    P30.G.DS, P30.G.RUN, P30.G.STATE = DS_OLD, RUN, RUN / 'status.json'
    os.environ['STEPMAP_REQ'] = str(K.OUT / 'stepmap_litoldg_requests.jsonl')
    os.environ['STEPMAP_CACHE'] = str(K.OUT / 'stepmap_litoldg_cache.json')
    SMAP.REQ, SMAP.CACHE = Path(os.environ['STEPMAP_REQ']), Path(os.environ['STEPMAP_CACHE'])
    old = load_cases(DS_OLD, GOLD_OLD, [p['pmcid'] for p in SCOPE])
    print(json.dumps({'old_cases': len(old)}), flush=True)
    if not cli.analyze_only:
        assert os.environ.get('FALLBACK_API_KEY'), 'FALLBACK_API_KEY required'
        cs, w = list(old.values()), cli.workers
        P30.G.phase('m2_pool', P30.job_pool, [(a, c) for a in ('base_default', 'samp1', 'samp2', 'samp3', 'samp4',
                                                                'enum3') for c in cs], w)
        P30.G.mapping(old)
        P30.G.phase('m2_round1', M.job_r1, [(DS_OLD, c) for c in cs], w)
        P30.G.mapping(old)
        P30.G.phase('m2_round2', M.job_r2, [(DS_OLD, c) for c in cs], w)
    new = load_cases(DS_NEW, GOLD_NEW, LIT172_IDS)
    rows = rows_for(DS_OLD, GOLD_OLD, old, lambda c: META[c]['exposure'])
    rows += rows_for(DS_NEW, GOLD_NEW, new, lambda c: 'recent')
    analyze(rows)


if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    main()
