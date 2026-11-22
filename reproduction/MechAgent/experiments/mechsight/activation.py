"""Activation pilot v2 (development set, last iteration): activate inert knowledge by consensus.

  1 elicit       five independent free-text narratives per case: S and G from the
                 inert-knowledge probe plus N1-N3 (S prompt, temperature 0.7)
  2 consolidate  keep candidate intermediates (sigma-graph key, endpoints removed)
                 asserted in >= 3 of the 5 narratives
  3 activate     executable PGL mechanism with the consensus candidates listed, using
                 the same hint wording as the MechSight pilot (arm A); cases without
                 consensus candidates keep the direct path E
  4 judge        E vs A in both orders, A taken only if preferred in both (arm AS)
The MechSight pilot arm M (union of S and G, same wording) is the no-consensus ablation.
Registration: outputs/activation_pilot_20261103/REGISTRATION.json.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pilot as PL  # noqa: E402
from pilot import A, C, P, chem, fm, phase2  # noqa: E402
import failover as F  # noqa: E402

OUT = P.ROOT / 'outputs' / 'activation_pilot_20261103'
NEW = ['N1', 'N2', 'N3']
THRESHOLD = 3


def narrative_job(name, case):
    dest = OUT / name / f"{case['case_id']}.json"
    if dest.exists():
        return 'cached'
    initial = chem.parse_pgl_literal(case['initial_state_pgl'])
    target = chem.parse_pgl_literal(case['target_state_pgl'])
    prompt = P.FREE_TASK.format(reactants=P.endpoint_smiles(initial), products=P.endpoint_smiles(target),
                                conditions=A.conditions_text(case), cue='')
    model = F.FailoverModel(OUT / name / 'traces' / f"{case['case_id']}.jsonl", max_tokens=P.MAX_TOKENS, temperature=0.7)
    try:
        answer = model.complete([{'role': 'system', 'content': P.FREE_SYSTEM}, {'role': 'user', 'content': prompt}], name)
    except Exception as exc:
        return f'error:{type(exc).__name__}'
    m = re.search(r'<intermediates>(.*?)</intermediates>', answer, re.S)
    lines = [ln.strip().strip('`').strip() for ln in (m.group(1) if m else '').splitlines()]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({'case_id': case['case_id'], 'intermediates': [x for x in lines if x], 'tag_found': bool(m),
                                'endpoints': model.endpoints_used}, indent=1))
    return 'ok'


def narratives(cid):
    out = []
    for folder in (P.OUT / 'S', P.OUT / 'G', *(OUT / n for n in NEW)):
        rec = PL.load(folder, cid)
        out.append(rec['intermediates'] if rec else None)
    return out


def consensus(case, threshold=THRESHOLD):
    """Candidates asserted in >= threshold narratives; SMILES of the first mention. None if a narrative is missing."""
    lists = narratives(case['case_id'])
    if any(x is None for x in lists):
        return None
    initial = chem.parse_pgl_literal(case['initial_state_pgl'])
    target = chem.parse_pgl_literal(case['target_state_pgl'])
    endpoint = {P.state_component_keys(s, c)[0] for s in (initial, target) for c in C._components(s)}
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
    return [first[k] for k, v in votes.items() if v >= threshold]


def activate_job(case):
    dest = OUT / 'A' / f"{case['case_id']}.json"
    if dest.exists():
        return 'cached'
    cands = consensus(case)
    if cands is None:
        return 'skip:narratives_missing'
    rec = {'case_id': case['case_id'], 'candidates': cands}
    if not cands:
        rec.update({'actions_pgl': PL.load(P.OUT / 'E', case['case_id'])['actions_pgl'], 'fallback_E': True})
    else:
        try:
            model = F.FailoverModel(OUT / 'A' / 'traces' / f"{case['case_id']}.jsonl", max_tokens=P.MAX_TOKENS)
            actions, errors = phase2.generate(model, A.user_prompt(case, hint=PL.CAND_HINT.format(lines='\n'.join(cands))), 'A')
        except Exception as exc:
            return f'error:{type(exc).__name__}'
        rec.update({'actions_pgl': actions, 'parse_errors': errors, 'fallback_E': False, 'endpoints': model.endpoints_used})
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1))
    return 'ok'


def judge_job(case, order):
    cid = case['case_id']
    dest = OUT / 'judge_EA' / f'{cid}_{order}.json'
    if dest.exists():
        return 'cached'
    base = PL.load(P.OUT / 'E', cid)['actions_pgl']
    alt = PL.load(OUT / 'A', cid)['actions_pgl']
    if A.sigma_signature(case, base) == A.sigma_signature(case, alt):
        rec = {'identical': True, 'picked_alt': None}
    else:
        tb, ta = A.render_actions(case, base), A.render_actions(case, alt)
        a, b = (ta, tb) if order == 'alt_first' else (tb, ta)
        try:
            model = F.FailoverModel(OUT / 'judge_EA' / 'traces' / f'{cid}_{order}.jsonl', max_tokens=P.MAX_TOKENS)
            choice = A.judge(model, case, a, b)
        except Exception as exc:
            return f'error:{type(exc).__name__}'
        rec = {'identical': False, 'choice': choice,
               'picked_alt': ((choice == 'A') if order == 'alt_first' else (choice == 'B')) if choice else None,
               'endpoints': model.endpoints_used}
    rec.update({'case_id': cid, 'order': order})
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1))
    return 'ok'


def candidate_quality(cases):
    """Offline: precision / heavy hidden coverage of consensus sets at thresholds 1..5."""
    out = {}
    for th in range(1, 6):
        n_c = hit = hev = hcov = contaminated = 0
        for cid, case in cases.items():
            cs = consensus(case, th)
            if cs is None:
                continue
            init = chem.parse_pgl_literal(case['initial_state_pgl'])
            states = C.gt_states(case)
            endpoint = {P.state_component_keys(s, c)[0] for s in (init, states[-1]) for c in C._components(s)}
            gt = {P.state_component_keys(s, c)[0] for s in states[:-1] for c in C._components(s)} - endpoint
            keys = {k for smi in cs for k, _ in P.smiles_keys(smi)} - endpoint
            n_c += len(keys)
            hit += len(keys & gt)
            ev = P.hidden_events(case)
            for i in ev.values():
                if i['keys'] and i['heavy']:
                    hev += 1
                    hcov += bool(i['keys'] & keys)
            if not any(i['keys'] for i in ev.values()):
                contaminated += bool(keys - gt)
        out[f'>={th}/5'] = {'candidates': n_c, 'precision': round(hit / n_c, 3) if n_c else None,
                            'heavy_hidden_coverage': round(hcov / hev, 3) if hev else None,
                            'no_hidden_cases_contaminated': contaminated}
    return out


def analyze(cases):
    rows, events = [], []
    for cid, case in cases.items():
        e, r2 = PL.load(P.OUT / 'E', cid), PL.load(P.OUT / 'E_r2', cid)
        a, m = PL.load(OUT / 'A', cid), PL.load(PL.OUT / 'M', cid)
        js = [PL.load(OUT / 'judge_EA', f'{cid}_{o}') for o in ('alt_first', 'base_first')]
        t_ee = PL.take_alt('EE', cid)
        if None in (e, r2, a, m, t_ee) or any(j is None for j in js):
            continue
        t_ea = (not js[0]['identical']) and all(j.get('picked_alt') for j in js)
        paths = {'E': e['actions_pgl'], 'DS': r2['actions_pgl'] if t_ee else e['actions_pgl'], 'M': m['actions_pgl'],
                 'A': a['actions_pgl'], 'AS': a['actions_pgl'] if t_ea else e['actions_pgl']}
        sc = {k: fm.score_case(case, v) for k, v in paths.items()}
        row = {'case': cid, 'n_cand': len(a['candidates']), 'fallback_E': a['fallback_E'], 'took_A': t_ea}
        for k, s in sc.items():
            row[f'{k}_sigma'] = float(s['sigma_L1'])
            row[f'{k}_order'] = float(s['intermediates_in_order'])
            row[f'{k}_spur'] = len(set(map(tuple, s['predicted_transient'])) - set(map(tuple, s['gold_transient'])))
        rows.append(row)
        for t, info in P.hidden_events(case).items():
            if info['keys']:
                events.append({'case': cid, 'heavy': info['heavy'],
                               **{k: t in set(map(tuple, s['predicted_transient'])) for k, s in sc.items()}})
    (OUT / 'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    ids = [r['case'] for r in rows]
    arms = ['E', 'DS', 'M', 'A', 'AS']
    rep = {'model': P.MODEL, 'cases': len(rows), 'threshold': f'>={THRESHOLD}/5',
           'fallback_E_rate': round(statistics.mean(r['fallback_E'] for r in rows), 3),
           'took_A_rate': round(statistics.mean(r['took_A'] for r in rows), 3),
           'candidates_per_case_median': statistics.median(r['n_cand'] for r in rows)}
    for metric in ('sigma', 'order'):
        rep[metric] = {a: phase2.ci([r[f'{a}_{metric}'] for r in rows], ids) for a in arms}
        for x, y in (('DS', 'A'), ('E', 'A'), ('DS', 'AS'), ('M', 'A')):
            d = [r[f'{y}_{metric}'] - r[f'{x}_{metric}'] for r in rows]
            rep[metric][f'{y} - {x}'] = phase2.ci(d, ids) + [sum(v > 0 for v in d), sum(v < 0 for v in d)]
    rep['best_of_E_A'] = round(statistics.mean(max(r['E_sigma'], r['A_sigma']) for r in rows), 3)
    rep['spurious_transients_per_case'] = {a: round(statistics.mean(r[f'{a}_spur'] for r in rows), 3) for a in arms}
    for name, sel in (('hidden_all', lambda e: True), ('hidden_heavy', lambda e: e['heavy'])):
        es = [e for e in events if sel(e)]
        rep[name] = {'events': len(es), **{a: phase2.ci([float(e[a]) for e in es], [e['case'] for e in es]) for a in arms}} if es else {'events': 0}
    rep['candidate_quality_by_threshold'] = candidate_quality(cases)
    (OUT / 'SUMMARY.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=150)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--analyze-only', action='store_true')
    cli = ap.parse_args()
    sys.path.insert(0, str(HERE.parent / 'agent_v2'))
    import j3_dev
    cases = j3_dev.load_cases()
    if cli.limit:
        cases = dict(list(cases.items())[:cli.limit])
    OUT.mkdir(parents=True, exist_ok=True)
    if not cli.analyze_only:
        PL.pool(narrative_job, [(n, c) for n in NEW for c in cases.values()], cli.workers)
        PL.pool(activate_job, [(c,) for c in cases.values()], cli.workers)
        PL.pool(judge_job, [(c, o) for c in cases.values() for o in ('alt_first', 'base_first')
                            if (OUT / 'A' / f"{c['case_id']}.json").exists()], cli.workers)
    analyze(cases)


if __name__ == '__main__':
    main()
