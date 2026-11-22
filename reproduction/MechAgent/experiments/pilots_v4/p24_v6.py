"""P24: Mech-Agent v6 development (dev sets only: T3_dev + J3; 2026-11-04 night).

Diagnosis behind v6 (scratchpad diag_auto / diag_blind):
  * v5 completion is one call over all candidates and adds ~1.2 states per reaction; completion-only
    new species were 0.8% of the misses on confirmation. It is the weakest stage.
  * On blind spots a concrete, local target (the HRMS formula) makes the model write new species
    (0 -> 36%), and the remaining error is mostly the wrong isomer, not ignoring the clue.
v6 change (one idea): resolve each flagged step locally.
  zoom     for every distinct flagged step A -> B in the candidates (most frequent first, <= MAXGAPS),
           two independent samples (T=0.7) of "write A -> B as elementary steps"; states RDKit-checked
           as in v5 (parse, not an endpoint, within the heavy-atom budget)
  revise   the v5 revision call, but the proposed intermediates are listed per flagged step with
           their support (how many zoom samples proposed them); one round
Arms: V6 (zoom + grouped revision); compared paired with V5 and Direct on the same reactions.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict

import common4 as K
import p11_mechagent as M
import p12_controls as C
import p10_memory as P10

MAXGAPS = 6
NSAMP = 2
ZOOM = ('Reactants and reagents (SMILES): {reactants}\nProducts (SMILES): {products}\nReaction conditions: {conditions}\n\n'
        'A proposed mechanism for this reaction contains the single step\n  {a}\n  ->  {b}\n'
        'An atom-mapping check finds that several heavy-atom bonds change in this one step, so it may hide short-lived '
        'intermediates (a tetrahedral adduct, an onium or ylide, a zwitterion, a cation before rearrangement, a radical '
        'or radical ion, a catalyst- or reagent-bound species, ...).\n\nWrite this transformation as a sequence of '
        'elementary steps. List every intermediate between the two species, in order, one SMILES per line between '
        '<states> and </states> (show the bound reagent or catalyst fragment where it is part of the species). If the '
        'step is genuinely concerted (e.g. pericyclic), leave <states></states> empty.')
REVISE6 = ('Reactants and reagents (SMILES): {reactants}\nProducts (SMILES): {products}\nReaction conditions: {conditions}\n\n'
           'Candidate mechanisms proposed for this reaction (intermediate SMILES after each step; steps flagged by an '
           'atom-mapping check change several bonds at once):\n\n{cands}\n\nEach flagged step was re-examined on its own; '
           'the intermediates proposed for it (valid structures, with how many of {n} independent attempts proposed '
           'each; not all of them need to be right):\n{groups}\n\nFirst judge each candidate: discard those with unsound '
           'elementary steps or missing intermediates. Then write the final mechanism: keep the most plausible candidate '
           'or repair it. For every flagged step that your final mechanism keeps, decide explicitly whether it is '
           'concerted or hides intermediates, and if it hides them, insert the intermediates where they belong. Return '
           'the final mechanism between [ANSWER] and [/ANSWER] as a JSON list of steps, each '
           '{{"step": n, "type": ..., "subtype": ..., "intermediate_smiles": ...}}.')


def flagged_steps(case, cands):
    """Distinct flagged steps (a, b) over all candidates, most frequent first."""
    cnt, where = Counter(), {}
    start = '.'.join(case['reactants'])
    for ci, m in enumerate(cands, 1):
        for i, _ in M.gaps(case, m):
            a = m[i - 1].get('intermediate_smiles') if i > 0 else start
            b = m[i].get('intermediate_smiles')
            key = (M.canon(a) or a, M.canon(b) or b)
            if not key[1]:
                continue
            cnt[key] += 1
            where.setdefault(key, []).append(f'C{ci}.S{i + 1}')
    order = sorted(cnt, key=lambda k: (-cnt[k], where[k][0]))
    return [(k, where[k]) for k in order[:MAXGAPS]]


def job_zoom(ds, case):
    cid = case['id']
    cands = M.cand_list(ds, cid)
    if len(cands) < 2:
        return 'skip'
    fs = flagged_steps(case, cands)
    for j, ((a, b), _) in enumerate(fs):
        for s in range(NSAMP):
            K.call(f'{ds}/V6_zoom{s}', f'{cid}__g{j}', ZOOM.format(a=a, b=b, **M.fields(case)), temperature=0.7)
    return f'{len(fs)} gaps'


def zoom_states(ds, case, cands=None):
    """OrderedDict gap -> {'where': [...], 'states': Counter(canon smiles -> support)}."""
    cid = case['id']
    cands = cands or M.cand_list(ds, cid)
    out = OrderedDict()
    for j, ((a, b), where) in enumerate(flagged_steps(case, cands)):
        sup = Counter()
        for s in range(NSAMP):
            for _, smi in M.checked_states(case, K.cached(f'{ds}/V6_zoom{s}', f'{cid}__g{j}') or ''):
                if smi not in (a, b):
                    sup[smi] += 1
        out[(a, b)] = {'where': where, 'states': sup}
    return out


def render_groups(zs):
    lines = []
    for (a, b), g in zs.items():
        head = f"- flagged step {', '.join(g['where'][:4])} ({b[:60]}{'...' if len(b) > 60 else ''}):"
        body = [f'    {smi}   [{n}/{NSAMP}]' for smi, n in g['states'].most_common()] or ['    (judged concerted)']
        lines.append('\n'.join([head] + body))
    return '\n'.join(lines) or '- none'


def job_revise(ds, case):
    cid = case['id']
    cands = M.cand_list(ds, cid)
    if len(cands) < 2:
        return 'skip'
    zs = zoom_states(ds, case, cands)
    p = REVISE6.format(cands=M.render(case, cands), groups=render_groups(zs), n=NSAMP, **M.fields(case))
    return 'ok' if K.call(f'{ds}/V6_rev', cid, p) else 'error'


def keys_of(m):
    return set().union(*[K.species_keys(s.get('intermediate_smiles') or '') for s in (m or [])] or [set()])


def coverage(all_sets):
    """Offline: where the GT hidden species missed by V5 go, and whether zoom states contain them."""
    tab = Counter()
    nstates = []
    for ds, cases in all_sets.items():
        for cid, case in cases.items():
            fin = K.parse_mech(M.final(ds, cid, 2))
            if fin is None:
                continue
            cands = M.cand_list(ds, cid)
            if len(cands) < 2:
                continue
            zs = zoom_states(ds, case, cands)
            zk = set()
            for g in zs.values():
                for smi in g['states']:
                    zk |= K.species_keys(smi)
            nstates.append(sum(len(g['states']) for g in zs.values()))
            v5st = set()
            for t in ('V5_comp1', 'V5_comp2'):
                for _, s in M.checked_states(case, K.cached(f'{ds}/{t}', cid) or ''):
                    v5st |= K.species_keys(s)
            pool = keys_of([s for m in P10.pool(ds, cid).values() if m for s in m])
            hid, _ = C.gt_intermediates(case)
            for h in hid:
                k = K.species_keys(h)
                if not k:
                    continue
                cat = 'v5_hit' if k & keys_of(fin) else 'pool_dropped' if k & pool else 'blind'
                tab[(cat, 'in_zoom' if k & zk else '-', 'in_v5_states' if k & v5st else '-')] += 1
    print(json.dumps({' | '.join(k): v for k, v in sorted(tab.items())}, indent=1))
    print('zoom states per reaction', round(sum(nstates) / max(1, len(nstates)), 2), 'n', len(nstates))


def analyze(all_sets):
    rows = {a: [] for a in ('Direct', 'M_ms_none', 'V5', 'V6')}
    for ds, cases in all_sets.items():
        for cid, case in cases.items():
            if K.cached(f'{ds}/V6_rev', cid) is None:
                continue
            ans = {'Direct': K.cached(f'{ds}/base_default', cid), 'M_ms_none': K.cached(f'{ds}/M_ms_none', cid),
                   'V5': M.final(ds, cid, 2), 'V6': K.cached(f'{ds}/V6_rev', cid)}
            for a, t in ans.items():
                rows[a].append((f'{ds}:{cid}', K.score(case, K.parse_mech(t))))
    rep = {a: K.summarize(r, ['all_present', 'in_order_frac']) for a, r in rows.items()}
    for hi, lo in (('V6', 'Direct'), ('V6', 'V5'), ('V6', 'M_ms_none'), ('V5', 'Direct')):
        rep[f'{hi}_minus_{lo}'] = K.paired(rows[hi], rows[lo], 'all_present')
    (K.OUT / 'SUMMARY_P24_V6_DEV.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


# ------------------------------------------------------------------ Guided v6 (blind spots, dev: J3 + T3_dev)
HYP = ('Reactants and reagents (SMILES): {reactants}\nProducts (SMILES): {products}\nReaction conditions: {conditions}\n'
       '{obs}\n\nPropose up to four distinct structures of exactly this molecular formula that could be this '
       'intermediate on the way from the reactants to the products, most likely first. One SMILES per line between '
       '<states> and </states>.')
G_REVISE = ('Reactants and reagents (SMILES): {reactants}\nProducts (SMILES): {products}\nReaction conditions: {conditions}\n'
            '{obs}\n\nCandidate mechanisms (some were written with this observation in mind; steps flagged by an '
            'atom-mapping check change several bonds at once):\n\n{cands}\n\nStructures that match the observed formula '
            'exactly (checked; with how many of the independent proposals contained each):\n{isomers}\n\nFirst decide '
            'which structure of the observed formula fits the mechanism: it must form from the species before it and '
            'lead to the species after it by elementary steps. Then write the final mechanism: keep the most plausible '
            'candidate or repair it so that it passes through that intermediate. Return the final mechanism between '
            '[ANSWER] and [/ANSWER] as a JSON list of steps, each {{"step": n, "type": ..., "subtype": ..., '
            '"intermediate_smiles": ...}}.')
G_RETRY = ('\n\nYour previous answer contained no intermediate of molecular formula {formula}, although the experiment '
           'observed it. Revise the mechanism so that one intermediate has exactly this formula.')
NCLUE = 3


def _formula(smi):
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    m = Chem.MolFromSmiles(smi or '')
    return rdMolDescriptors.CalcMolFormula(m) if m is not None else None


def blind_cases(names):
    import p22_extras as X
    out = []
    for ds in names:
        for cid, case in K.load_fukuyama(ds).items():
            if K.cached(f'{ds}/BS_ms', cid) is None:
                continue
            bs = X.blind_species(ds, cid, case)
            if bs:
                out.append((ds, case, bs[0][0], bs[0][1]))
    return out


def _obs(target):
    import p22_extras as X
    return X.clue(target).strip()


def clue_paths(ds, cid):
    tags = ['BS_llm'] + [f'G6_clue{k}' for k in range(1, NCLUE)]
    return [m for t in tags if (m := K.parse_mech(K.cached(f'{ds}/{t}', cid)))]


def isomers(ds, case, target):
    """Formula-checked structures with support across the hypothesis calls and the clue-conditioned paths."""
    f = _formula(target)
    sup = Counter()
    for k in range(2):
        for _, smi in M.checked_states(case, K.cached(f'{ds}/G6_hyp{k}', case['id']) or ''):
            if _formula(smi) == f:
                sup[smi] += 1
    for m in clue_paths(ds, case['id']):
        for s in {M.canon(x.get('intermediate_smiles')) for x in m}:
            if s and _formula(s) == f:
                sup[s] += 1
    return sup


def job_g(ds, case, target):
    cid = case['id']
    obs = _obs(target)
    base = K.official_prompt(case).replace('\n\n## Input', '\n\n' + obs + '\n\n## Input', 1)
    for k in range(1, NCLUE):
        K.call(f'{ds}/G6_clue{k}', cid, base, temperature=0.7)
    for k in range(2):
        K.call(f'{ds}/G6_hyp{k}', cid, HYP.format(obs=obs, **M.fields(case)), temperature=0.7)
    cands = M.cand_list(ds, cid)[:5] + clue_paths(ds, cid)
    iso = isomers(ds, case, target)
    iso_txt = '\n'.join(f'- {s}   [{n}]' for s, n in iso.most_common()) or '- none'
    p = G_REVISE.format(obs=obs, cands=M.render(case, cands), isomers=iso_txt, **M.fields(case))
    ans = K.call(f'{ds}/G6_rev', cid, p)
    if ans is None:
        return 'error'
    f = _formula(target)
    m = K.parse_mech(ans)
    if m and not any(_formula(x.get('intermediate_smiles')) == f for x in m):
        K.call(f'{ds}/G6_retry', cid, p + '\n\nPrevious answer:\n' + ans + G_RETRY.format(formula=f))
        return 'retried'
    return 'ok'


def g_final(ds, cid):
    r = K.cached(f'{ds}/G6_retry', cid)
    if r is not None and K.parse_mech(r):
        return r
    return K.cached(f'{ds}/G6_rev', cid)


def g_analyze(names):
    rows = {a: [] for a in ('Direct_clue', 'Guided_v5', 'Guided_v6', 'Iso_top1')}
    for ds, case, target, key in blind_cases(names):
        cid = case['id']
        if K.cached(f'{ds}/G6_rev', cid) is None:
            continue
        ans = {'Direct_clue': K.cached(f'{ds}/BS_llm', cid), 'Guided_v5': K.cached(f'{ds}/BS_ms', cid),
               'Guided_v6': g_final(ds, cid)}
        for a, t in ans.items():
            m = K.parse_mech(t)
            s = K.score(case, m)
            s['blind_species_written'] = float(bool(key & keys_of(m)))
            s['formula_written'] = float(any(_formula(x.get('intermediate_smiles')) == _formula(target) for x in (m or [])))
            rows[a].append((f'{ds}:{cid}', s))
        iso = isomers(ds, case, target)
        top = iso.most_common(1)[0][0] if iso else ''
        rows['Iso_top1'].append((f'{ds}:{cid}', {'blind_species_written': float(bool(key & K.species_keys(top))),
                                                  'all_present': 0.0, 'formula_written': float(bool(top))}))
        rows.setdefault('Iso_any', []).append((f'{ds}:{cid}', {'blind_species_written': float(any(key & K.species_keys(s) for s in iso)),
                                                               'all_present': 0.0, 'formula_written': float(bool(iso))}))
    rep = {a: K.summarize(r, ['blind_species_written', 'formula_written', 'all_present']) for a, r in rows.items()}
    for lo in ('Direct_clue', 'Guided_v5'):
        rep[f'Guided_v6_minus_{lo}'] = {k: K.paired(rows['Guided_v6'], rows[lo], k) for k in ('blind_species_written', 'all_present')}
    (K.OUT / f'SUMMARY_P24_G6_{"_".join(names)}.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['zoom', 'coverage', 'revise', 'analyze', 'guided', 'g_analyze'], required=True)
    ap.add_argument('--workers', type=int, default=120)
    ap.add_argument('--sets', default='T3_dev,J3')
    cli = ap.parse_args()
    names = tuple(cli.sets.split(','))
    s = M.sets(names)
    if cli.phase in ('zoom', 'revise'):
        fn = job_zoom if cli.phase == 'zoom' else job_revise
        K.run_pool(fn, [(ds, c) for ds, cs in s.items() for c in cs.values()], cli.workers)
    elif cli.phase == 'coverage':
        coverage(s)
    elif cli.phase == 'guided':
        K.run_pool(job_g, [(ds, c, t) for ds, c, t, _ in blind_cases(names)], cli.workers)
        g_analyze(names)
    elif cli.phase == 'g_analyze':
        g_analyze(names)
    else:
        analyze(s)
