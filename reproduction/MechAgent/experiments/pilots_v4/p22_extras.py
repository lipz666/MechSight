"""P22: reviewer extras (2026-11-04).

blind    Guided mode on true blind spots. Fukuyama J3 + T3_dev + T3_confirm: a blind-spot species is a
         GT hidden intermediate missed by Direct that appears in no pool candidate and no checked v5
         state. Per case the first such species (heavy hidden first) gets a simulated HRMS clue
         (formula + exact mass, no structure; P8 wording). Arms:
           BS_llm   official prompt + clue (plain LLM with the same clue)
           BS_ms    Mech-Agent-Guided: v5 completion prompt + clue -> checked states -> P16 clue revision
         Metric: the blind species written (clue_species_present), all_present.
randsmi  Randomised (non-canonical) SMILES. oMe-Gold product prediction on the 133 P5 pairs with two
         random atom orderings of the original reactants; J3 mechanism (official prompt) with random
         SMILES of reactants and products. A string-memorising model should drop.
curve    Compute curve: samp5..samp8 (T=0.7, official prompt) on T3_confirm + oMe-Gold, so that
         self-consistency (MBR) and best-of-N (oracle pick) can be scored for N = 1..8 calls (Mech-Agent uses ~8).
"""
from __future__ import annotations

import argparse
import json
import random

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

import common4 as K
import p3_selector as P3
import p5_contamination as P5
import p10_memory as P10
import p11_mechagent as M
import p12_controls as C
import p16_supplement_g38 as P16

RDLogger.DisableLog('rdApp.*')
OBS = ('\n\nExperimental observation: HRMS of the reaction mixture shows a transient species of molecular formula '
       '{formula} (exact mass {mass:.4f}). Its structure was not determined. The mechanism must account for it.')


# ------------------------------------------------------------------ blind spots
def fuku_sets():
    return {ds: K.load_fukuyama(ds) for ds in ('J3', 'T3_dev', 'T3_confirm')}


def blind_species(ds, cid, case):
    hid, _ = C.gt_intermediates(case)
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
    for smi in hid:
        k = K.species_keys(smi)
        if k and not (k & direct) and not (k & pool) and not (k & states) and Chem.MolFromSmiles(smi) is not None:
            out.append((smi, k))
    return out


def clue(smi):
    mol = Chem.MolFromSmiles(smi)
    return OBS.format(formula=rdMolDescriptors.CalcMolFormula(mol), mass=Descriptors.ExactMolWt(mol))


def job_blind(ds, case):
    cid = case['id']
    bs = blind_species(ds, cid, case)
    if not bs:
        return 'none'
    obs = clue(bs[0][0])
    K.call(f'{ds}/BS_llm', cid, K.official_prompt(case).replace('\n\n## Input', obs + '\n\n## Input', 1))
    cands = M.cand_list(ds, cid)
    if len(cands) < 2:
        return 'skip'
    text = M.render(case, cands)
    comp = K.call(f'{ds}/BS_comp', cid, M.COMPLETE.format(cands=text, **M.fields(case)) + obs)
    st = M.checked_states(case, comp) if comp else []
    states = '\n'.join(f'- ({loc}) {s}' for loc, s in st) or '- none'
    p = P16.V5_OBS_REVISE.format(cands=text, states=states, obs=obs.strip(), **M.fields(case))
    return 'ok' if K.call(f'{ds}/BS_ms', cid, p) else 'error'


def analyze_blind():
    rows = {a: [] for a in ('Direct', 'Auto', 'Direct_clue', 'Guided')}
    n_species = 0
    for ds, cases in fuku_sets().items():
        for cid, case in cases.items():
            if K.cached(f'{ds}/BS_ms', cid) is None:
                continue
            bs = blind_species(ds, cid, case)
            if not bs:
                continue
            n_species += len(bs)
            target = bs[0][1]
            ans = {'Direct': K.cached(f'{ds}/base_default', cid), 'Auto': M.final(ds, cid, 2),
                   'Direct_clue': K.cached(f'{ds}/BS_llm', cid), 'Guided': K.cached(f'{ds}/BS_ms', cid)}
            for a, t in ans.items():
                m = K.parse_mech(t)
                s = K.score(case, m)
                keys = set().union(*[K.species_keys(x.get('intermediate_smiles') or '') for x in (m or [])] or [set()])
                s['blind_species_written'] = float(bool(target & keys))
                rows[a].append((f'{ds}:{cid}', s))
    rep = {'cases': len(rows['Guided']), 'blind_species_total': n_species}
    rep.update({a: K.summarize(r, ['blind_species_written', 'all_present']) for a, r in rows.items()})
    for hi, lo in (('Guided', 'Direct_clue'), ('Guided', 'Auto'), ('Direct_clue', 'Direct')):
        rep[f'{hi}_minus_{lo}'] = {k: K.paired(rows[hi], rows[lo], k) for k in ('blind_species_written', 'all_present')}
    (K.OUT / 'SUMMARY_P22_BLIND.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


# ------------------------------------------------------------------ randomised SMILES
def rand_smiles(smi, seed):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    order = list(range(mol.GetNumAtoms()))
    random.Random(seed).shuffle(order)
    return Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)


def job_rand_product(h, k):
    rs = [rand_smiles(s, 1000 * k + i) for i, s in enumerate(h['orig_reactants'])]
    p = P5.TASK.format(reactants='.'.join(rs), conditions=h['conditions'])
    return 'ok' if K.call(f'gold/p5_rand{k}', h['reaction_id'], p) else 'error'


def j3_random():
    out = {}
    for cid, case in K.load_fukuyama('J3').items():
        c = dict(case)
        c['reactants'] = [rand_smiles(s, 7 + i) for i, s in enumerate(case['reactants'])]
        c['products'] = [rand_smiles(s, 70 + i) for i, s in enumerate(case['products'])]
        out[cid] = c
    return out


def job_rand_mech(case):
    return 'ok' if K.call('J3/p22_rand_mech', case['id'], K.official_prompt(case)) else 'error'


def analyze_rand():
    rows = []
    for rid, h in P5.homologs().items():
        vals = [P5.correct(K.cached(f'gold/p5_{w}', rid), h['orig_products'] if w != 'hom' else h['products'],
                           h['orig_reactants'] if w != 'hom' else h['reactants']) for w in ('orig', 'rand1', 'rand2', 'hom')]
        if all(v is not None for v in vals):
            rows.append((rid, vals))
    ids = [r for r, _ in rows]
    rep = {'product_prediction_pairs': len(rows)}
    for i, w in enumerate(('canonical', 'random_1', 'random_2', 'homologue')):
        rep[w] = K.phase2.ci([v[i] for _, v in rows], ids)
    for i, w in ((1, 'random_1'), (2, 'random_2'), (3, 'homologue')):
        rep[f'canonical_minus_{w}'] = K.phase2.ci([v[0] - v[i] for _, v in rows], ids)
    orig, ra, rb = K.load_fukuyama('J3'), [], []
    for cid, c in j3_random().items():
        a, b = K.cached('J3/base_default', cid), K.cached('J3/p22_rand_mech', cid)
        if a and b:
            ra.append((cid, K.score(orig[cid], K.parse_mech(a))))
            rb.append((cid, K.score(orig[cid], K.parse_mech(b))))
    rep['J3_mechanism'] = {'canonical': K.summarize(ra, ['all_present']), 'random': K.summarize(rb, ['all_present']),
                           'canonical_minus_random': K.paired(ra, rb, 'all_present')}
    (K.OUT / 'SUMMARY_P22_RANDSMI.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


# ------------------------------------------------------------------ compute curve
def job_curve(ds, i, case):
    return 'ok' if K.call(f'{ds}/samp{i}', case['id'], K.official_prompt(case), temperature=0.7) else 'error'


def analyze_curve():
    sets = M.sets(('T3_confirm', 'gold'))
    ns = (1, 2, 4, 8)
    rep = {}
    pooled = {f'SC@{n}': [] for n in ns} | {f'BestOfN_oracle@{n}': [] for n in ns} | {'MechSight': [], 'Direct': []}
    calls = []
    for ds, cases in sets.items():
        for cid, case in cases.items():
            names = ['base_default'] + [f'samp{i}' for i in range(1, 8)]
            mechs = [K.parse_mech(K.cached(f'{ds}/{n}', cid)) for n in names]
            if any(K.cached(f'{ds}/{n}', cid) is None for n in names) or K.cached(f'{ds}/V5_rev1', cid) is None:
                continue
            key = f'{ds}:{cid}'
            for n in ns:
                sub = {names[j]: m for j, m in enumerate(mechs[:n]) if m}
                pick = P3.consensus(sub, list(sub)) if len(sub) >= 2 else (next(iter(sub)) if sub else None)
                pooled[f'SC@{n}'].append((key, {'all_present': K.score(case, sub.get(pick))['all_present'] if pick else 0.0}))
                best = max((K.score(case, m)['all_present'] for m in sub.values()), default=0.0)
                pooled[f'BestOfN_oracle@{n}'].append((key, {'all_present': best}))
            pooled['Direct'].append((key, {'all_present': K.score(case, mechs[0])['all_present']}))
            pooled['MechSight'].append((key, {'all_present': K.score(case, K.parse_mech(M.final(ds, cid, 2)))['all_present']}))
            calls.append(7 + 2 * bool(K.cached(f'{ds}/V5_rev2', cid)) + 1)
    rep = {a: K.summarize(r, ['all_present']) for a, r in pooled.items() if r}
    rep['MechSight_calls_per_reaction'] = round(sum(calls) / len(calls), 2) if calls else None
    rep['note'] = 'SC@n = MBR consensus over the first n samples (base_default + samp1..); BestOfN_oracle = best sample by GT (upper bound of any selector)'
    (K.OUT / 'SUMMARY_P22_CURVE.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['blind', 'randsmi', 'curve', 'analyze'], required=True)
    ap.add_argument('--workers', type=int, default=100)
    cli = ap.parse_args()
    if cli.task == 'blind':
        K.run_pool(job_blind, [(ds, c) for ds, cs in fuku_sets().items() for c in cs.values()], cli.workers)
        analyze_blind()
    elif cli.task == 'randsmi':
        K.run_pool(job_rand_product, [(h, k) for h in P5.homologs().values() for k in (1, 2)], cli.workers)
        K.run_pool(job_rand_mech, [(c,) for c in j3_random().values()], cli.workers)
        analyze_rand()
    elif cli.task == 'curve':
        sets = M.sets(('T3_confirm', 'gold'))
        K.run_pool(job_curve, [(ds, i, c) for ds, cs in sets.items() for i in range(5, 8) for c in cs.values()], cli.workers)
        analyze_curve()
    else:
        analyze_blind(); analyze_rand(); analyze_curve()
