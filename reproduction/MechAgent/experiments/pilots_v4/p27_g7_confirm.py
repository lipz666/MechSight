"""P27: pre-registered held-out test of frozen Guided v7 (REGISTRATION_g7_confirm.json).

Held-out cases: blind spots of T3_confirm (Fukuyama PGL GT) and oMe-Gold (gold GT, P16 hidden/visible split),
same blind-spot definition as P22 (a GT hidden intermediate missed by Direct, absent from every pool candidate and
every checked v5 completion state); the first such species per reaction gets the simulated HRMS clue (P22 wording).
Arms (identical code to dev):
  Direct_clue  official prompt + clue (P22 BS_llm)
  Guided_v5    v5 completion + clue revision (P22 BS_comp -> BS_ms)
  Guided_v7    frozen p25 pipeline: clue paths (BS_llm, G6_clue1..2) + 4 bookkeeping hypotheses + 3 isomer votes
               + revision with the Auto final and v5 states + formula retry
Primary: blind_species_written, Guided_v7 - Direct_clue (paired, reaction-clustered bootstrap 95% CI).
Secondary: all_present; Guided_v7 - Guided_v5.
"""
from __future__ import annotations

import argparse
import json

import common4 as K
import p10_memory as P10
import p11_mechagent as M
import p12_controls as C
import p16_supplement_g38 as P16
import p22_extras as X
import p24_v6 as V
import p25_guided7 as G

SETS = ('T3_confirm', 'gold')


FUKU = ('J3', 'T3_dev', 'T3_confirm')


def load(ds):
    if ds in FUKU:
        return K.load_fukuyama(ds)
    if ds == 'gold':
        return K.load_gold()
    import p21_scale as S  # template / silver_conf (scale-up sets)
    return S.sets()[ds]


def hidden(ds, case):
    return C.gt_intermediates(case)[0] if ds in FUKU else P16.gold_gt_intermediates(case)[0]


def blind(ds, cid, case):
    if K.cached(f'{ds}/V5_rev1', cid) is None or K.cached(f'{ds}/base_default', cid) is None:
        return []
    keys = lambda m: set().union(*[K.species_keys(s.get('intermediate_smiles') or '') for s in (m or [])] or [set()])  # noqa: E731
    direct = keys(K.parse_mech(K.cached(f'{ds}/base_default', cid)))
    pool = set()
    for m in P10.pool(ds, cid).values():
        pool |= keys(m)
    states = set()
    for tag in ('V5_comp1', 'V5_comp2'):
        for _, smi in M.checked_states(case, K.cached(f'{ds}/{tag}', cid) or ''):
            states |= K.species_keys(smi)
    out = []
    for smi in hidden(ds, case):
        k = K.species_keys(smi)
        if k and not (k & direct) and not (k & pool) and not (k & states) and X.Chem.MolFromSmiles(smi) is not None:
            out.append((smi, k))
    return out


_LOADED = {}


def cases():
    out = []
    for ds in SETS:
        if ds not in _LOADED:
            _LOADED[ds] = load(ds)
        for cid, case in _LOADED[ds].items():
            b = blind(ds, cid, case)
            if b:
                out.append((ds, case, b[0][0], b[0][1]))
    return out


def job_clue(ds, case, target, k):
    """Clue-conditioned paths: k=0 is BS_llm (Direct_clue arm, T=0.2), k=1,2 are G6_clue (T=0.7)."""
    cid = case['id']
    if k == 0:
        p = K.official_prompt(case).replace('\n\n## Input', X.clue(target) + '\n\n## Input', 1)
        return 'ok' if K.call(f'{ds}/BS_llm', cid, p) else 'error'
    base = K.official_prompt(case).replace('\n\n## Input', '\n\n' + V._obs(target) + '\n\n## Input', 1)
    return 'ok' if K.call(f'{ds}/G6_clue{k}', cid, base, temperature=0.7) else 'error'


def job_v5guided(ds, case, target):
    """P22 Guided (v5) arm, same calls as X.job_blind but for the given target."""
    cid = case['id']
    obs = X.clue(target)
    cands = M.cand_list(ds, cid)
    if len(cands) < 2:
        return 'skip'
    text = M.render(case, cands)
    comp = K.call(f'{ds}/BS_comp', cid, M.COMPLETE.format(cands=text, **M.fields(case)) + obs)
    st = M.checked_states(case, comp) if comp else []
    states = '\n'.join(f'- ({loc}) {s}' for loc, s in st) or '- none'
    p = P16.V5_OBS_REVISE.format(cands=text, states=states, obs=obs.strip(), **M.fields(case))
    return 'ok' if K.call(f'{ds}/BS_ms', cid, p) else 'error'


def analyze():
    arms = ('Direct', 'Auto_V5', 'Direct_clue', 'Guided_v5', 'Guided_v7')
    rows = {a: [] for a in arms}
    per_set = {}
    for ds, case, target, key in cases():
        cid = case['id']
        if K.cached(f'{ds}/G7_rev', cid) is None:
            continue
        ans = {'Direct': K.cached(f'{ds}/base_default', cid), 'Auto_V5': M.final(ds, cid, 2),
               'Direct_clue': K.cached(f'{ds}/BS_llm', cid), 'Guided_v5': K.cached(f'{ds}/BS_ms', cid),
               'Guided_v7': G.g7_final(ds, cid)}
        for a, t in ans.items():
            m = K.parse_mech(t)
            s = {'all_present': K.score(case, m)['all_present'], 'blind_species_written': float(bool(key & V.keys_of(m)))}
            rows[a].append((f'{ds}:{cid}', s))
            per_set.setdefault(ds, {}).setdefault(a, []).append((f'{ds}:{cid}', s))
    met = ['blind_species_written', 'all_present']
    rep = {'n': len(rows['Guided_v7']), 'pooled': {a: K.summarize(r, met) for a, r in rows.items()}}
    for lo in ('Direct_clue', 'Guided_v5', 'Auto_V5'):
        rep['pooled'][f'Guided_v7_minus_{lo}'] = {k: K.paired(rows['Guided_v7'], rows[lo], k) for k in met}
    for ds, d in per_set.items():
        rep[ds] = {a: K.summarize(r, met) for a, r in d.items()}
        if len(d['Guided_v7']) >= 2:
            rep[ds]['Guided_v7_minus_Direct_clue'] = {k: K.paired(d['Guided_v7'], d['Direct_clue'], k) for k in met}
    name = 'SUMMARY_P27_G7_CONFIRM.json' if SETS == ('T3_confirm', 'gold') else f'SUMMARY_P27_G7_{"_".join(SETS)}.json'
    (K.OUT / name).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['count', 'run', 'analyze'], required=True)
    ap.add_argument('--workers', type=int, default=300)
    ap.add_argument('--sets', default='T3_confirm,gold')
    cli = ap.parse_args()
    SETS = tuple(cli.sets.split(','))
    if cli.phase == 'count':
        from collections import Counter
        print(Counter(ds for ds, *_ in cases()))
    elif cli.phase == 'run':
        bc = [(ds, c, t) for ds, c, t, _ in cases()]
        K.run_pool(job_clue, [(ds, c, t, k) for ds, c, t in bc for k in range(3)], cli.workers)
        K.run_pool(job_v5guided, bc, cli.workers)
        K.run_pool(G.job_hyp, [(ds, c, t, k) for ds, c, t in bc for k in range(G.NHYP)], cli.workers)
        K.run_pool(G.job_vote, [(ds, c, t, k) for ds, c, t in bc for k in range(G.NVOTE)], cli.workers)
        K.run_pool(G.job_g7, bc, cli.workers)
        analyze()
    else:
        analyze()
