"""P25: Guided v7 (dev blind spots: J3 + T3_dev; T3_confirm blind spots held out).

Three fixes to Guided v6, each aimed at one measured shortfall (diag_g6, 41 dev cases):
  generation (16/41 right isomer never proposed)
      formula bookkeeping: the observed formula is decomposed programmatically into the reaction's
      species (reactants, reagents, products; up to 3, repeats allowed) plus/minus small molecules
      (H2O, HCl, CO2, N2, ...) and protons; the decompositions go into the hypothesis prompt;
      4 hypothesis calls instead of 2. Isomers are kept only if RDKit confirms the exact formula.
  selection (8/41 right isomer proposed but not chosen)
      isomer vote: 3 independent judgements (T=0.7), each must say how every isomer forms from the
      species before it and goes on to the next; majority (ties by proposal support).
  other intermediates (21/41 miss another hidden species)
      the Auto (V5) final path is candidate 1 and the V5 completion states are given to the revision.
Final check as v6: the chosen isomer must be in the final path (else one retry with feedback).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from itertools import combinations_with_replacement

from rdkit import Chem

import common4 as K
import p11_mechagent as M
import p24_v6 as V

SMALL = {'H2O': 'O', 'HCl': 'Cl', 'HBr': 'Br', 'HI': 'I', 'HF': 'F', 'H2': '[H][H]', 'CO2': 'O=C=O', 'CO': '[C-]#[O+]',
         'N2': 'N#N', 'NH3': 'N', 'MeOH': 'CO', 'EtOH': 'CCO', 'AcOH': 'CC(=O)O', 'O2': 'O=O', 'SO2': 'O=S=O'}
NHYP = 4
NVOTE = 3


def comp(smi):
    m = Chem.MolFromSmiles(smi or '')
    if m is None:
        return None
    m = Chem.AddHs(m)
    c = Counter(a.GetSymbol() for a in m.GetAtoms())
    c['+'] = Chem.GetFormalCharge(m)
    return c


def _sub(a, b):
    out = Counter(a)
    out.subtract(b)
    return out


def _zero(c):
    return all(v == 0 for v in c.values())


def bookkeeping(case, target, limit=4):
    """Decompositions 'target = A + B - H2O (+ H+)' over the reaction species; fewest terms first."""
    goal = comp(target)
    names = {}
    for s in dict.fromkeys(case['reactants'] + case['products']):
        if comp(s) is not None and Chem.MolFromSmiles(s).GetNumHeavyAtoms() > 0:
            names[s] = s
    sp = list(names)
    small = [(k, comp(v)) for k, v in SMALL.items()]
    proton = Counter({'H': 1, '+': 1})
    found = []
    for r in (1, 2, 3):
        for combo in combinations_with_replacement(sp, r):
            tot = Counter()
            for s in combo:
                tot.update(comp(s))
            rest = _sub(goal, tot)  # what still has to be added (+) or removed (-)
            for p in (0, 1, -1):
                rp = _sub(rest, Counter({k: p * v for k, v in proton.items()}))
                if _zero(rp):
                    found.append((r + abs(p), combo, [], p))
                    continue
                for k, c in small:  # one small molecule added or removed
                    for sign in (1, -1):
                        if _zero(_sub(rp, Counter({e: sign * v for e, v in c.items()}))):
                            found.append((r + 1 + abs(p), combo, [(sign, k)], p))
        if len(found) >= limit:
            break
    found.sort(key=lambda x: x[0])
    out = []
    for _, combo, sm, p in found[:limit]:
        terms = ' + '.join(combo) + ''.join(f" {'+' if s > 0 else '-'} {k}" for s, k in sm)
        terms += ' + H+' if p == 1 else ' - H+' if p == -1 else ''
        out.append(terms)
    return out


HYP7 = ('Reactants and reagents (SMILES): {reactants}\nProducts (SMILES): {products}\nReaction conditions: {conditions}\n'
        '{obs}\n\nFormula bookkeeping (computed; which species of this reaction add up to the observed formula):\n'
        '{book}\n\nPropose up to four distinct structures of exactly this molecular formula that could be this '
        'intermediate on the way from the reactants to the products, most likely first. One SMILES per line between '
        '<states> and </states>.')
VOTE = ('Reactants and reagents (SMILES): {reactants}\nProducts (SMILES): {products}\nReaction conditions: {conditions}\n'
        '{obs}\n\nA mechanism proposed without this observation:\n{path}\n\nStructures that match the observed formula '
        '(checked):\n{isomers}\n\nFor each structure, say in one or two sentences from which species it would form by '
        'an elementary step and to which species it would go on, under these conditions. Then choose the structure '
        'that is the observed intermediate. End with <choice>k</choice>, k being its number.')
G7_REVISE = ('Reactants and reagents (SMILES): {reactants}\nProducts (SMILES): {products}\nReaction conditions: {conditions}\n'
             '{obs}\n\nThe observed species is most likely\n  {chosen}\n(other structures of this formula: {others})\n\n'
             'Candidate mechanisms (candidate 1 was revised for elementary-step completeness; others were written with '
             'the observation in mind; flagged steps change several bonds at once):\n\n{cands}\n\nIntermediates '
             'proposed for flagged steps (valid structures; not all of them need to be right):\n{states}\n\nWrite the '
             'final mechanism: it must pass through the observed species, and every step must be elementary (insert the '
             'proposed intermediates where they are chemically justified). Return the final mechanism between [ANSWER] '
             'and [/ANSWER] as a JSON list of steps, each {{"step": n, "type": ..., "subtype": ..., '
             '"intermediate_smiles": ...}}.')
RETRY7 = ('\n\nPrevious answer:\n{prev}\n\nThe previous answer does not pass through the observed species {chosen}. '
          'Revise it so that it does.')


def isomers7(ds, case, target):
    f = V._formula(target)
    sup = Counter()
    for k in range(NHYP):
        for _, smi in M.checked_states(case, K.cached(f'{ds}/G7_hyp{k}', case['id']) or ''):
            if V._formula(smi) == f:
                sup[smi] += 1
    for m in V.clue_paths(ds, case['id']):
        for s in {M.canon(x.get('intermediate_smiles')) for x in m}:
            if s and V._formula(s) == f:
                sup[s] += 1
    return sup


def chosen(ds, case, iso):
    ranked = [s for s, _ in iso.most_common()]
    votes = Counter()
    for k in range(NVOTE):
        m = re.search(r'<choice>\s*(\d+)', K.cached(f'{ds}/G7_vote{k}', case['id']) or '')
        if m and 0 < int(m.group(1)) <= len(ranked):
            votes[ranked[int(m.group(1)) - 1]] += 1
    if not ranked:
        return None
    return max(ranked, key=lambda s: (votes[s], iso[s]))


def _book(case, target):
    return '\n'.join(f'- {V._formula(target)} = {b}' for b in bookkeeping(case, target)) or '- no simple decomposition'


def job_hyp(ds, case, target, k):
    """One hypothesis call (parallel phase 1)."""
    p = HYP7.format(obs=V._obs(target), book=_book(case, target), **M.fields(case))
    return 'ok' if K.call(f'{ds}/G7_hyp{k}', case['id'], p, temperature=0.7) else 'error'


def _vote_prompt(ds, case, target):
    iso = isomers7(ds, case, target)
    if not iso:
        return None
    ranked = [s for s, _ in iso.most_common()]
    v5 = K.parse_mech(M.final(ds, case['id'], 2)) or []
    path = '\n'.join(f"  step {j}: {s.get('intermediate_smiles')}" for j, s in enumerate(v5, 1)) or '  (none)'
    return VOTE.format(obs=V._obs(target), path=path, isomers='\n'.join(f'{i}. {s}' for i, s in enumerate(ranked, 1)),
                       **M.fields(case))


def job_vote(ds, case, target, k):
    """One isomer vote (parallel phase 2)."""
    p = _vote_prompt(ds, case, target)
    if p is None:
        return 'noiso'
    return 'ok' if K.call(f'{ds}/G7_vote{k}', case['id'], p, temperature=0.7) else 'error'


def job_g7(ds, case, target):
    cid = case['id']
    obs = V._obs(target)
    f = dict(M.fields(case))
    for k in range(NHYP):
        job_hyp(ds, case, target, k)
    iso = isomers7(ds, case, target)
    if not iso:
        return 'noiso'
    ranked = [s for s, _ in iso.most_common()]
    v5 = K.parse_mech(M.final(ds, cid, 2)) or []
    path = '\n'.join(f"  step {j}: {s.get('intermediate_smiles')}" for j, s in enumerate(v5, 1)) or '  (none)'
    for k in range(NVOTE):
        job_vote(ds, case, target, k)
    ch = chosen(ds, case, iso)
    cands = ([v5] if v5 else []) + M.cand_list(ds, cid)[:4] + V.clue_paths(ds, cid)
    st = []
    for t in ('V5_comp1', 'V5_comp2'):
        st += M.checked_states(case, K.cached(f'{ds}/{t}', cid) or '')
    states = '\n'.join(f'- {s}' for s in dict.fromkeys(s for _, s in st)) or '- none'
    p = G7_REVISE.format(obs=obs, chosen=ch, others=', '.join(s for s in ranked if s != ch)[:600] or 'none',
                         cands=M.render(case, cands), states=states, **f)
    ans = K.call(f'{ds}/G7_rev', cid, p)
    if ans is None:
        return 'error'
    m = K.parse_mech(ans)
    if m and not (K.species_keys(ch) & V.keys_of(m)):
        K.call(f'{ds}/G7_retry', cid, p + RETRY7.format(prev=ans, chosen=ch))
        return 'retried'
    return 'ok'


def g7_final(ds, cid):
    r = K.cached(f'{ds}/G7_retry', cid)
    return r if r is not None and K.parse_mech(r) else K.cached(f'{ds}/G7_rev', cid)


def analyze(names):
    arms = ('Direct_clue', 'Guided_v5', 'Guided_v6', 'Guided_v7', 'Auto_V5')
    rows = {a: [] for a in arms}
    diag = Counter()
    for ds, case, target, key in V.blind_cases(names):
        cid = case['id']
        if K.cached(f'{ds}/G7_hyp0', cid) is None:
            continue
        ans = {'Direct_clue': K.cached(f'{ds}/BS_llm', cid), 'Guided_v5': K.cached(f'{ds}/BS_ms', cid),
               'Guided_v6': V.g_final(ds, cid), 'Guided_v7': g7_final(ds, cid), 'Auto_V5': M.final(ds, cid, 2)}
        for a, t in ans.items():
            m = K.parse_mech(t)
            s = K.score(case, m)
            s['blind_species_written'] = float(bool(key & V.keys_of(m)))
            rows[a].append((f'{ds}:{cid}', s))
        iso = isomers7(ds, case, target)
        inl = any(key & K.species_keys(s) for s in iso)
        ch = chosen(ds, case, iso) if iso else None
        diag['right_isomer_in_list'] += inl
        diag['right_isomer_chosen'] += bool(ch and key & K.species_keys(ch))
        diag['bookkeeping_found'] += bool(bookkeeping(case, target))
        diag['n'] += 1
    rep = {'diagnostics': dict(diag)}
    rep.update({a: K.summarize(r, ['blind_species_written', 'all_present']) for a, r in rows.items()})
    for lo in ('Direct_clue', 'Guided_v6', 'Auto_V5'):
        rep[f'Guided_v7_minus_{lo}'] = {k: K.paired(rows['Guided_v7'], rows[lo], k) for k in ('blind_species_written', 'all_present')}
    (K.OUT / f'SUMMARY_P25_G7_{"_".join(names)}.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['run', 'analyze', 'book'], required=True)
    ap.add_argument('--sets', default='J3,T3_dev')
    ap.add_argument('--workers', type=int, default=120)
    cli = ap.parse_args()
    names = tuple(cli.sets.split(','))
    if cli.phase == 'run':
        bc = [(ds, c, t) for ds, c, t, _ in V.blind_cases(names)]
        K.run_pool(job_hyp, [(ds, c, t, k) for ds, c, t in bc for k in range(NHYP)], cli.workers)
        K.run_pool(job_vote, [(ds, c, t, k) for ds, c, t in bc for k in range(NVOTE)], cli.workers)
        K.run_pool(job_g7, bc, cli.workers)
        analyze(names)
    elif cli.phase == 'book':
        for ds, c, t, _ in V.blind_cases(names):
            print(V._formula(t), '|', bookkeeping(c, t)[:2])
    else:
        analyze(names)
