"""Literature-memory scaling study: pre-registered analysis of source recall vs exposure.

Exposure c = OpenAlex citations up to 2023 (primary); total citations (secondary). x = ln(1 + c).
Per paper k hits out of n parsed R1 samples (binomial). Models (maximum likelihood, numpy Nelder-Mead):
  const | loglin p = a + b x | logistic | POWER p = A (1+c)^b | power + year | free threshold step |
  fixed threshold at c >= 1000 | saturating p = pmax (1+c)^b / ((1+c)^b + K)
Primary: power-law exponent b for gemini-3.8-flash, paper-level bootstrap (1000) 95% CI; "clearly better" = dAIC >= 4.
Hold-out: fit on a random half (seed), predict hit rate per citation bin on the other half; pass if >= 80 % of bins
have the observed rate inside its Wilson 95% CI around the prediction... (observed Wilson CI contains prediction).
Secondary: exponents for argon and flash-lite; year, OA and input-source covariates; swap control; calibration.

python analyze_scale.py
"""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'pilots_v4'))
import common4 as K  # noqa: E402
import p33_m1_m4 as P33  # noqa: E402
import r1_scale as R  # noqa: E402

SEED = 20261107
OUT = K.OUT / 'lit_scale'


def minimize(f, x0, iters=6000):
    x0 = np.asarray(x0, float)
    n = len(x0)
    sim = [x0] + [x0 + np.eye(n)[i] * (0.5 if x0[i] == 0 else 0.25 * abs(x0[i]) + 0.1) for i in range(n)]
    fs = [f(v) for v in sim]
    for _ in range(iters * n):
        o = np.argsort(fs)
        sim, fs = [sim[i] for i in o], [fs[i] for i in o]
        if abs(fs[-1] - fs[0]) < 1e-10 and np.max(np.abs(np.array(sim) - sim[0])) < 1e-8:
            break
        c = np.mean(sim[:-1], axis=0)
        xr = c + (c - sim[-1])
        fr = f(xr)
        if fr < fs[0]:
            xe = c + 2 * (c - sim[-1])
            fe = f(xe)
            sim[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fs[-2]:
            sim[-1], fs[-1] = xr, fr
        else:
            xc = c + 0.5 * (sim[-1] - c)
            fc = f(xc)
            if fc < fs[-1]:
                sim[-1], fs[-1] = xc, fc
            else:
                sim = [sim[0]] + [sim[0] + 0.5 * (v - sim[0]) for v in sim[1:]]
                fs = [f(v) for v in sim]
    i = int(np.argmin(fs))
    return sim[i], fs[i]


def nll(p, k, n):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -float(np.sum(k * np.log(p) + (n - k) * np.log(1 - p)))


MODELS = {
    'const': (lambda t, d: np.full(len(d['x']), 1 / (1 + math.exp(-t[0]))), [-2.0]),
    'loglin': (lambda t, d: t[0] + t[1] * d['x'], [0.02, 0.02]),
    'logistic': (lambda t, d: 1 / (1 + np.exp(-(t[0] + t[1] * d['x']))), [-4.0, 0.5]),
    'power': (lambda t, d: np.exp(t[0] + t[1] * d['x']), [-4.0, 0.5]),
    'power_year': (lambda t, d: np.exp(t[0] + t[1] * d['x'] + t[2] * d['y']), [-4.0, 0.5, 0.0]),
    'power_year_oa_src': (lambda t, d: np.exp(t[0] + t[1] * d['x'] + t[2] * d['y'] + t[3] * d['oa'] + t[4] * d['pdf']),
                          [-4.0, 0.5, 0.0, 0.0, 0.0]),
    'threshold_1000': (lambda t, d: np.where(d['c'] >= 1000, 1 / (1 + math.exp(-t[1])), 1 / (1 + math.exp(-t[0]))),
                       [-2.5, -1.0]),
    'saturating': (lambda t, d: (1 / (1 + math.exp(-t[0]))) * np.exp(t[1] * d['x']) / (np.exp(t[1] * d['x'])
                                                                                       + math.exp(t[2])), [0.0, 0.6, 4.0]),
}


def fit(name, d, t0=None):
    f, init = MODELS[name]
    t, v = minimize(lambda t: nll(f(t, d), d['k'], d['n']), init if t0 is None else t0)
    return t, v


def free_threshold(d):
    best = None
    for c0 in np.unique(np.round(d['c'])):
        hi = d['c'] >= c0
        if hi.sum() < 20 or (~hi).sum() < 20:
            continue
        p1 = d['k'][hi].sum() / d['n'][hi].sum()
        p0 = d['k'][~hi].sum() / d['n'][~hi].sum()
        v = nll(np.where(hi, p1, p0), d['k'], d['n'])
        if best is None or v < best[0]:
            best = (v, float(c0), p0, p1)
    return best


def sub(d, idx):
    return {k: v[idx] for k, v in d.items()}


def wilson(h, n, z=1.96):
    p = h / n
    den = 1 + z * z / n
    ce = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return ce - hw, ce + hw


def data(model, inp='main', exposure='oa_cites_to2023'):
    sample = {p['id']: p for p in json.loads((HERE / 'sample.json').read_text())['papers']}
    ins = R.inputs()
    rows = []
    for key, d in ins.items():
        if inp not in d:
            continue
        p = sample[key]
        meta = {'doi': p['doi'], 'year': p['year'], 'authors': p['authors']}
        hits = [P33.r1_hit(P33._refs(a), meta) for s in range(1, R.NS + 1)
                if (a := R.answer(model, inp, s, key)) is not None]
        if not hits or not p['authors']:
            continue
        rows.append({'id': key, 'c': p[exposure] or 0, 'year': p['year'], 'oa': float(p['oa']),
                     'pdf': float(p['source'] == 'pdf'), 'k': sum(hits), 'n': len(hits), 'bin': p['bin']})
    d = {'c': np.array([r['c'] for r in rows], float), 'k': np.array([r['k'] for r in rows], float),
         'n': np.array([r['n'] for r in rows], float), 'y': np.array([r['year'] - 2012 for r in rows], float),
         'oa': np.array([r['oa'] for r in rows]), 'pdf': np.array([r['pdf'] for r in rows]),
         'bin': np.array([r['bin'] for r in rows])}
    d['x'] = np.log1p(d['c'])
    return d, rows


def scaling(model, inp='main', exposure='oa_cites_to2023', nboot=1000):
    d, rows = data(model, inp, exposure)
    rep = {'papers': len(rows), 'samples': int(d['n'].sum()), 'hits': int(d['k'].sum()),
           'papers_with_hit': int((d['k'] > 0).sum()), 'max_c': float(d['c'].max())}
    fits = {}
    for name in MODELS:
        t, v = fit(name, d)
        fits[name] = t
        rep.setdefault('fits', {})[name] = {'params': [round(float(x), 4) for x in t], 'nll': round(v, 2),
                                            'aic': round(2 * v + 2 * len(t), 2)}
    ft = free_threshold(d)
    rep['fits']['free_threshold'] = {'c0': ft[1], 'p_low': round(ft[2], 4), 'p_high': round(ft[3], 4),
                                     'aic': round(2 * ft[0] + 6, 2)}
    rng = random.Random(SEED)
    bs = {'power': [], 'power_year': []}
    idx = np.arange(len(rows))
    for _ in range(nboot):
        s = np.array([rng.randrange(len(rows)) for _ in idx])
        ds = sub(d, s)
        for name in bs:
            bs[name].append(fit(name, ds, fits[name])[0][1])
    rep['exponent_power'] = [round(float(fits['power'][1]), 3)] + [round(float(x), 3)
                                                                  for x in np.percentile(bs['power'], [2.5, 97.5])]
    rep['exponent_power_year'] = [round(float(fits['power_year'][1]), 3)] + [
        round(float(x), 3) for x in np.percentile(bs['power_year'], [2.5, 97.5])]
    rep['year_factor_per_year'] = round(math.exp(float(fits['power_year'][2])), 3)
    # hold-out
    perm = list(idx)
    random.Random(SEED + 2).shuffle(perm)
    a, b = np.array(perm[:len(perm) // 2]), np.array(perm[len(perm) // 2:])
    ta, _ = fit('power', sub(d, a))
    db = sub(d, b)
    ok, tot, per_bin = 0, 0, []
    for bn in sorted(set(db['bin'].tolist())):
        m = db['bin'] == bn
        if db['n'][m].sum() < 20:
            continue
        pred = float(np.sum(np.exp(ta[0] + ta[1] * db['x'][m]) * db['n'][m]) / db['n'][m].sum())
        h, n = db['k'][m].sum(), db['n'][m].sum()
        lo, hi = wilson(h, n)
        tot += 1
        ok += lo <= pred <= hi
        per_bin.append([int(bn), round(100 * pred, 2), round(100 * h / n, 2), int(n)])
    rep['holdout'] = {'bins_within_ci': ok, 'bins': tot, 'pass': tot > 0 and ok / tot >= 0.8, 'per_bin': per_bin,
                      'fit_half_A_exponent': round(float(ta[1]), 3)}
    rep['binned'] = []
    for bn in sorted(set(d['bin'].tolist())):
        m = d['bin'] == bn
        h, n = d['k'][m].sum(), d['n'][m].sum()
        lo, hi = wilson(h, n)
        rep['binned'].append([int(bn), round(float(np.exp(np.mean(np.log1p(d['c'][m])))) - 1, 1), int(m.sum()),
                              round(100 * h / n, 2), round(100 * lo, 2), round(100 * hi, 2)])
    return rep


def paired(model_inp_a, model_inp_b):
    da, ra = data(*model_inp_a)
    db, rb = data(*model_inp_b)
    ka = {r['id']: r['k'] / r['n'] for r in ra}
    kb = {r['id']: r['k'] / r['n'] for r in rb}
    common = sorted(set(ka) & set(kb))
    diff = [ka[i] - kb[i] for i in common]
    rng = random.Random(SEED)
    bs = [np.mean([diff[rng.randrange(len(diff))] for _ in diff]) for _ in range(2000)]
    return {'n': len(common), 'a': round(100 * float(np.mean([ka[i] for i in common])), 2),
            'b': round(100 * float(np.mean([kb[i] for i in common])), 2),
            'a_minus_b': [round(100 * float(np.mean(diff)), 2)] + [round(100 * float(x), 2)
                                                                   for x in np.percentile(bs, [2.5, 97.5])]}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--primary', default='flash36')
    ap.add_argument('--others', default='')
    cli = ap.parse_args()
    m = cli.primary
    rep = {f'primary_{m}': scaling(m), f'{m}_total_citations': scaling(m, exposure='oa_cites', nboot=300),
           'swap_control_main_minus_swap': paired((m, 'main'), (m, 'swap')),
           'calibration_pdf_minus_abstract': paired((m, 'main'), (m, 'cal'))}
    for o in [x for x in cli.others.split(',') if x]:
        rep[o] = scaling(o, nboot=300)
    (OUT / 'SUMMARY_SCALE_R1.json').write_text(json.dumps(rep, indent=1, default=lambda o: o.item()))
    print(json.dumps(rep, indent=1, default=lambda o: o.item()))


if __name__ == '__main__':
    main()
