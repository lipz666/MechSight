"""MechSight v3 (development on T3_dev, 47 cases; gemini with two-endpoint failover).

Agent loop per case:
  1 elicit      five free-text narratives (S prompt, temperature 0.7)
  2 consolidate consensus set C = candidates in >= 3/5 narratives; broad set B = >= 2/5
  3 activate    P1 = executable path with C listed; P2 = with B listed
  4 gap-fill    in P1, every step with >= GAP_SIGMA sigma events is a suspected skipped stage:
                three local narratives (before/after species, temperature 0.7) propose the
                intermediates between them; those in >= 2/3 join C, and P3 is rebuilt with them
  5 select      among E, P1, P2, P3 the path most consistent with the narratives:
                score = |path species in C'| - LAMBDA * |path intermediates never narrated|
                (ties: earlier in the list E, P1, P3, P2)
E is the T4 direct baseline. Nothing in steps 1-5 sees the GT.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import baselines as B  # noqa: E402
import failover as F  # noqa: E402
import pilot as PL  # noqa: E402
from pilot import A, C, P, chem, fm, phase2  # noqa: E402

OUT = P.ROOT / 'outputs' / 'mechsight_v3_dev_20261103'
N_NARR, GAP_SIGMA, LAMBDA = 5, 3, 0.5
GAP_TASK = ('In the reaction below, one proposed elementary step converts the species on the left into the species '
            'on the right.\nReaction conditions: {conditions}\nBefore: {before}\nAfter: {after}\n\n'
            'Can this happen in a single elementary step? If short-lived intermediates (for example carbocations, '
            'ylides, onium ions, carbenes, tetrahedral adducts) lie between them, list their SMILES in order inside '
            '<intermediates> and </intermediates>; if it is genuinely one step, write <intermediates></intermediates>.')


def model(folder, name, temperature=0.2):
    return F.FailoverModel(OUT / folder / 'traces' / f'{name}.jsonl', max_tokens=P.MAX_TOKENS, temperature=temperature)


def save(folder, name, rec):
    dest = OUT / folder / f'{name}.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1))
    return rec


def intermediates_from(answer):
    m = re.search(r'<intermediates>(.*?)</intermediates>', answer, re.S)
    return [x.strip().strip('`').strip() for x in (m.group(1) if m else '').splitlines() if x.strip()]


def endpoint_keys(case):
    i = chem.parse_pgl_literal(case['initial_state_pgl'])
    t = chem.parse_pgl_literal(case['target_state_pgl'])
    return {P.state_component_keys(s, c)[0] for s in (i, t) for c in C._components(s)}


def vote(lists, endpoint):
    votes, first = Counter(), {}
    for lst in lists:
        seen = set()
        for smi in lst:
            keys = tuple(sorted(k for k, _ in P.smiles_keys(smi)))
            if not keys or all(k in endpoint for k in keys) or keys in seen:
                continue
            seen.add(keys)
            votes[keys] += 1
            first.setdefault(keys, smi)
    return votes, first


def path_species(case, actions):
    states, _ = fm.replay(case['initial_state_pgl'], actions)
    return states, {P.state_component_keys(s, c)[0] for s in states for c in C._components(s)}


def activated(case, cid, tag, cands):
    rec = PL.load(OUT / tag, cid)
    if rec:
        return rec
    hint = PL.CAND_HINT.format(lines='\n'.join(cands)) if cands else None
    m = model(tag, cid)
    actions, errors = phase2.generate(m, A.user_prompt(case, hint=hint), tag)
    return save(tag, cid, {'candidates': cands, 'actions_pgl': actions, 'parse_errors': errors, 'endpoints': m.endpoints_used})


def run_case(case):
    cid = case['case_id']
    if (OUT / 'final' / f'{cid}.json').exists():
        return 'cached'
    try:
        if not (B.OUT / 'T3' / 'gemini' / 'E' / f'{cid}.json').exists():
            B.job('T3', 'gemini', case)
        e = PL.load(B.OUT / 'T3' / 'gemini' / 'E', cid)
        initial = chem.parse_pgl_literal(case['initial_state_pgl'])
        target = chem.parse_pgl_literal(case['target_state_pgl'])
        prompt = P.FREE_TASK.format(reactants=P.endpoint_smiles(initial), products=P.endpoint_smiles(target),
                                    conditions=A.conditions_text(case), cue='')
        def narrate(k):
            rec = PL.load(OUT / f'N{k}', cid)
            if not rec:
                m = model(f'N{k}', cid, 0.7)
                ans = m.complete([{'role': 'system', 'content': P.FREE_SYSTEM}, {'role': 'user', 'content': prompt}], f'N{k}')
                rec = save(f'N{k}', cid, {'intermediates': intermediates_from(ans), 'endpoints': m.endpoints_used})
            return rec['intermediates']
        with ThreadPoolExecutor(N_NARR) as ex:
            lists = list(ex.map(narrate, range(N_NARR)))
        endpoint = endpoint_keys(case)
        votes, first = vote(lists, endpoint)
        cons = [first[k] for k, v in votes.items() if v >= 3]
        broad = [first[k] for k, v in votes.items() if v >= 2]
        p1 = activated(case, cid, 'P1', cons)
        p2 = activated(case, cid, 'P2', broad) if set(broad) != set(cons) else p1
        # gap filling on P1
        states, _ = path_species(case, p1['actions_pgl'])
        gap_cands, gaps = [], []
        prev = initial
        for k, st in enumerate(states):
            step = fm.step_from_states(prev, st)
            if len(step['form']) + len(step['break']) >= GAP_SIGMA:
                before = C.step_text(st, prev)[1]  # species of the earlier state touched by the step
                after = C.step_text(prev, st)[1]
                gaps.append(k)
                def local(r, k=k, before=before, after=after):
                    name = f'{cid}_s{k}_{r}'
                    rec = PL.load(OUT / 'gap', name)
                    if not rec:
                        m = model('gap', name, 0.7)
                        ans = m.complete([{'role': 'system', 'content': P.FREE_SYSTEM}, {'role': 'user', 'content': GAP_TASK.format(
                            conditions=A.conditions_text(case), before=before, after=after)}], 'gap')
                        rec = save('gap', name, {'intermediates': intermediates_from(ans), 'endpoints': m.endpoints_used})
                    return rec['intermediates']
                with ThreadPoolExecutor(3) as ex:
                    g_lists = list(ex.map(local, range(3)))
                gv, gf = vote(g_lists, endpoint)
                gap_cands += [gf[x] for x, v in gv.items() if v >= 2]
            prev = st
        cons_plus = cons + [s for s in gap_cands if s not in cons]
        p3 = activated(case, cid, 'P3', cons_plus) if gap_cands else p1
        # selection by consistency with the narratives
        ckeys = {k for s in cons_plus for k, _ in P.smiles_keys(s)}
        narrated = {k for keys in votes for k in keys} | {k for s in gap_cands for k, _ in P.smiles_keys(s)}
        scores = {}
        for tag, rec in (('E', e), ('P1', p1), ('P3', p3), ('P2', p2)):
            _, sp = path_species(case, rec['actions_pgl'])
            inter = sp - endpoint - {''}
            scores[tag] = len(inter & ckeys) - LAMBDA * len(inter - narrated)
        chosen = max(scores, key=lambda t: (scores[t], -['E', 'P1', 'P3', 'P2'].index(t)))
        paths = {'E': e, 'P1': p1, 'P2': p2, 'P3': p3}
        save('final', cid, {'case_id': cid, 'consensus': cons, 'broad': broad, 'gap_steps': gaps, 'gap_candidates': gap_cands,
                            'scores': scores, 'chosen': chosen, 'actions_pgl': paths[chosen]['actions_pgl']})
    except Exception as exc:
        return f'error:{type(exc).__name__}:{str(exc)[:80]}'
    return 'ok'


def analyze(cases):
    rows, events = [], []
    for cid, case in cases.items():
        fin = PL.load(OUT / 'final', cid)
        if not fin:
            continue
        paths = {'E': PL.load(B.OUT / 'T3' / 'gemini' / 'E', cid), 'P1': PL.load(OUT / 'P1', cid),
                 'P2': PL.load(OUT / 'P2', cid) or PL.load(OUT / 'P1', cid), 'P3': PL.load(OUT / 'P3', cid) or PL.load(OUT / 'P1', cid),
                 'MS3': fin}
        sc = {k: fm.score_case(case, v['actions_pgl']) for k, v in paths.items()}
        row = {'case': cid, 'chosen': fin['chosen'], 'n_gap': len(fin['gap_steps'])}
        for k, s in sc.items():
            row[f'{k}_sigma'], row[f'{k}_order'] = float(s['sigma_L1']), float(s['intermediates_in_order'])
            row[f'{k}_spur'] = len(set(map(tuple, s['predicted_transient'])) - set(map(tuple, s['gold_transient'])))
        row['oracle'] = max(row[f'{k}_sigma'] for k in ('E', 'P1', 'P2', 'P3'))
        rows.append(row)
        for t, info in P.hidden_events(case).items():
            if info['keys']:
                events.append({'case': cid, 'heavy': info['heavy'], **{k: t in set(map(tuple, s['predicted_transient'])) for k, s in sc.items()}})
    (OUT / 'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    ids = [r['case'] for r in rows]
    arms = ['E', 'P1', 'P2', 'P3', 'MS3']
    rep = {'cases': len(rows), 'chosen': dict(Counter(r['chosen'] for r in rows)),
           'cases_with_gap_steps': sum(r['n_gap'] > 0 for r in rows),
           'oracle_best_of_paths': round(statistics.mean(r['oracle'] for r in rows), 3) if rows else None}
    for metric in ('sigma', 'order'):
        rep[metric] = {a: phase2.ci([r[f'{a}_{metric}'] for r in rows], ids) for a in arms}
        for a in ('P1', 'P3', 'MS3'):
            d = [r[f'{a}_{metric}'] - r[f'E_{metric}'] for r in rows]
            rep[metric][f'{a} - E'] = phase2.ci(d, ids) + [sum(x > 0 for x in d), sum(x < 0 for x in d)]
    rep['spurious_per_case'] = {a: round(statistics.mean(r[f'{a}_spur'] for r in rows), 3) for a in arms}
    for name, sel in (('hidden_all', lambda e: True), ('hidden_heavy', lambda e: e['heavy'])):
        es = [e for e in events if sel(e)]
        rep[name] = {'events': len(es), **{a: phase2.ci([float(e[a]) for e in es], [e['case'] for e in es]) for a in arms}} if es else {'events': 0}
    (OUT / 'SUMMARY.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=60)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--analyze-only', action='store_true')
    cli = ap.parse_args()
    sets = B.load_sets()
    dev = json.loads((B.OUT / 'splits.json').read_text())['T3_dev']
    cases = {cid: sets['T3'][cid] for cid in dev}
    if cli.limit:
        cases = dict(list(cases.items())[:cli.limit])
    OUT.mkdir(parents=True, exist_ok=True)
    if not cli.analyze_only:
        PL.pool(run_case, [(c,) for c in cases.values()], cli.workers)
    analyze(cases)


if __name__ == '__main__':
    main()
