"""Inert-knowledge phase 2: factor decomposition inside the executable PGL format.

New conditions (same model, cases, and scoring as probe.py):
  E_r2       E resampled (noise floor)
  E_nofacts  E without the endpoint-derived net bond-change facts
  E_cue      E plus the generic cue used in condition G
Registration: outputs/inert_knowledge_20261102/REGISTRATION_phase2.json.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe as P  # noqa: E402
from probe import A, C, fm, metrics  # noqa: E402

CONDS = ['E_r2', 'E_nofacts', 'E_cue']


def prompt_without_facts(case):
    return ('Using the initial and target molecular structures and the reaction conditions, infer a complete stepwise '
            'elementary electron-transfer mechanism. Reaction name, class, GT intermediates and GT step count are hidden.\n'
            f"INITIAL:\n{case['initial_state_pgl']}\n"
            f"TARGET:\n{case['target_state_pgl']}\n"
            f'Reaction conditions: {A.conditions_text(case)}\n')


def generate(model, user, phase):
    """Same two turns as agent_v2.generate (path, then audit), with a given user prompt."""
    base = [{'role': 'system', 'content': A.SYSTEM}, {'role': 'user', 'content': user}]
    first = model.complete(base, f'{phase}.path')
    second = model.complete(base + [{'role': 'assistant', 'content': first}, {'role': 'user', 'content': A.P23.AUDIT}], f'{phase}.audit')
    errors = []
    for text in (second, first):
        try:
            return [repr(a) for a in A.P23.parse_path(text)], errors
        except Exception as exc:
            errors.append(str(exc)[:200])
    return [], errors


def job(cond, case):
    dest = P.OUT / cond / f"{case['case_id']}.json"
    if dest.exists():
        return 'cached'
    model = C.Model(P.MODEL, P.OUT / cond / 'traces' / f"{case['case_id']}.jsonl", base_url=P.BASE_URL, max_tokens=P.MAX_TOKENS)
    if cond == 'E_r2':
        user = A.user_prompt(case)
    elif cond == 'E_nofacts':
        user = prompt_without_facts(case)
    else:
        user = A.user_prompt(case, hint=P.CUE.strip())
    try:
        actions, errors = generate(model, user, cond)
    except Exception as exc:
        return f'error:{type(exc).__name__}'
    rec = {'case_id': case['case_id'], 'cond': cond, 'model': P.MODEL, 'actions_pgl': actions, 'parse_errors': errors,
           'tokens': model.tokens, 'calls': len(model.events)}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1))
    return 'ok'


def run(cases, conds, workers):
    jobs = [(c, case) for c in conds for case in cases.values()]
    random.Random(11).shuffle(jobs)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futs = [pool.submit(job, *j) for j in jobs]
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            status = fut.result()
            if status.startswith('error') or done % 50 == 0:
                print(json.dumps({'done': done, 'of': len(jobs), 'status': status}), flush=True)


def ci(vals, ids):
    lo, hi = metrics.clustered_bootstrap_ci(vals, ids, seed=20261102)
    return [round(statistics.mean(vals), 3), round(lo, 3), round(hi, 3)]


def analyze(cases):
    pgl = ['E', 'E_r2', 'E_nofacts', 'E_cue']
    rows, case_rows = [], []
    for cid, case in cases.items():
        recs = {c: P.load(c, cid) for c in pgl + ['S', 'G']}
        if any(r is None for r in recs.values()):
            continue
        scores = {c: fm.score_case(case, recs[c]['actions_pgl']) for c in pgl}
        case_rows.append({'case': cid, **{c: float(scores[c]['sigma_L1']) for c in pgl}})
        listed = {c: [k for smi in recs[c]['intermediates'] for k in P.smiles_keys(smi)] for c in ('S', 'G')}
        for t, info in P.hidden_events(case).items():
            if not info['keys']:
                continue
            row = {'case': cid, 'heavy': info['heavy']}
            for c in pgl:
                row[c] = t in set(map(tuple, scores[c]['predicted_transient']))
            row['K'] = any(k in info['keys'] for c in ('S', 'G') for k, _ in listed[c])
            rows.append(row)
    (P.OUT / 'rows_phase2.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')

    def block(sel):
        rs = [r for r in rows if sel(r)]
        if not rs:
            return {'events': 0}
        ids = [r['case'] for r in rs]
        out = {'events': len(rs), 'cases': len(set(ids))}
        for c in pgl:
            out[f'recall_{c}'] = ci([float(r[c]) for r in rs], ids)
        base = [(float(r['E']) + float(r['E_r2'])) / 2 for r in rs]
        for c in ('E_nofacts', 'E_cue'):
            out[f'{c}_minus_meanE'] = ci([float(r[c]) - b for r, b in zip(rs, base)], ids)
        out['E_r2_minus_E'] = ci([float(r['E_r2']) - float(r['E']) for r in rs], ids)
        om = [r for r in rs if not r['E']]
        oid = [r['case'] for r in om]
        if om:
            out['omitted_in_E'] = len(om)
            out['omitted_E__reappear_in_E_r2'] = ci([float(r['E_r2']) for r in om], oid)
            out['omitted_E__known_free_text'] = ci([float(r['K']) for r in om], oid)
            both = [r for r in om if not r['E_r2']]
            if both:
                out['omitted_E_and_E_r2'] = len(both)
                out['omitted_both__known_free_text'] = ci([float(r['K']) for r in both], [r['case'] for r in both])
        return out

    cids = [r['case'] for r in case_rows]
    report = {'model': P.MODEL, 'cases_complete': len(case_rows),
              'sigma_L1': {c: ci([r[c] for r in case_rows], cids) for c in pgl},
              'sigma_L1_E_nofacts_minus_meanE': ci([r['E_nofacts'] - (r['E'] + r['E_r2']) / 2 for r in case_rows], cids),
              'sigma_L1_E_cue_minus_meanE': ci([r['E_cue'] - (r['E'] + r['E_r2']) / 2 for r in case_rows], cids),
              'heavy_atom_events': block(lambda r: r['heavy']), 'all_events': block(lambda r: True),
              'h_transfer_events': block(lambda r: not r['heavy'])}
    (P.OUT / 'SUMMARY_phase2.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=300)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--conds', nargs='+', default=CONDS)
    ap.add_argument('--analyze-only', action='store_true')
    ap.add_argument('--model')
    ap.add_argument('--base-url')
    ap.add_argument('--max-tokens', type=int)
    cli = ap.parse_args()
    P.configure(cli.model, cli.base_url, cli.max_tokens)
    sys.path.insert(0, str(HERE.parent / 'agent_v2'))
    import j3_dev
    cases = j3_dev.load_cases()
    if cli.limit:
        cases = dict(list(cases.items())[:cli.limit])
    if not cli.analyze_only:
        run(cases, cli.conds, cli.workers)
    analyze(cases)


if __name__ == '__main__':
    main()
