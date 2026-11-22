"""Visibility manipulation: does showing a hidden intermediate let the model use it?

Within the executable PGL format (same prompt and audit turn as E), one
intermediate structure is added to the prompt as an experimental observation:
  V_plus   a GT hidden species (contains a sigma bond absent from both
           endpoints; heavy-atom event preferred, earliest window first)
  V_zero   a GT visible intermediate (differs from both endpoints but contains
           no transient bond): same amount of information, nothing hidden
Only SMILES are shown: no atom labels, bonds or arrows.
Registration: outputs/inert_knowledge_20261102/REGISTRATION_visibility.json.
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
import phase2  # noqa: E402
import probe as P  # noqa: E402
from probe import A, C, chem, fm  # noqa: E402

CONDS = ['V_plus', 'V_zero']
OBS = ('Experimental observation: the following intermediate was detected during this reaction '
       '(SMILES, atom labels not given): {smiles}')


def shown_species(case):
    """(hidden SMILES, its key, visible SMILES or None, its key or None); None without a hidden species."""
    events = P.hidden_events(case)
    events = {t: i for t, i in events.items() if i['keys']}
    if not events:
        return None
    initial = chem.parse_pgl_literal(case['initial_state_pgl'])
    states = C.gt_states(case)
    endpoint = {P.state_component_keys(s, c)[0] for s in (initial, states[-1]) for c in C._components(s)}
    hidden_keys = set().union(*(i['keys'] for i in events.values()))
    heavy_keys = set().union(*(i['keys'] for i in events.values() if i['heavy'])) if any(i['heavy'] for i in events.values()) else set()
    want = heavy_keys or hidden_keys
    hidden = visible = None
    for st in states[:-1]:
        for comp in C._components(st):
            key = P.state_component_keys(st, comp)[0]
            if not key or key in endpoint:
                continue
            if hidden is None and key in want:
                hidden = (C._molecule_smiles(st, comp), key)
            if visible is None and key not in hidden_keys and len(comp) > 2:
                visible = (C._molecule_smiles(st, comp), key)
    if hidden is None:
        return None
    return hidden[0], hidden[1], visible[0] if visible else None, visible[1] if visible else None


def job(cond, case, shown):
    dest = P.OUT / cond / f"{case['case_id']}.json"
    if dest.exists():
        return 'cached'
    smiles = shown[0] if cond == 'V_plus' else shown[2]
    model = C.Model(P.MODEL, P.OUT / cond / 'traces' / f"{case['case_id']}.jsonl", base_url=P.BASE_URL, max_tokens=P.MAX_TOKENS)
    try:
        actions, errors = phase2.generate(model, A.user_prompt(case, hint=OBS.format(smiles=smiles)), cond)
    except Exception as exc:
        return f'error:{type(exc).__name__}'
    rec = {'case_id': case['case_id'], 'cond': cond, 'model': P.MODEL, 'base_url': P.BASE_URL, 'shown_smiles': smiles,
           'actions_pgl': actions, 'parse_errors': errors, 'tokens': model.tokens, 'calls': len(model.events)}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1))
    return 'ok'


def analyze(cases, shown):
    conds = ['E', 'E_r2', 'V_plus', 'V_zero']
    rows, case_rows = [], []
    for cid, case in cases.items():
        if cid not in shown:
            continue
        recs = {c: P.load(c, cid) for c in conds + ['S', 'G']}
        paired = recs['V_zero'] is not None
        if any(recs[c] is None for c in recs if c != 'V_zero'):
            continue
        sc = {c: fm.score_case(case, recs[c]['actions_pgl']) for c in conds if recs[c] is not None}
        case_rows.append({'case': cid, 'paired': paired, **{c: float(sc[c]['sigma_L1']) for c in sc}})
        listed = [k for c in ('S', 'G') for smi in recs[c]['intermediates'] for k, _ in P.smiles_keys(smi)]
        for t, info in P.hidden_events(case).items():
            if not info['keys']:
                continue
            row = {'case': cid, 'heavy': info['heavy'], 'shown': shown[cid][1] in info['keys'], 'paired': paired,
                   'known': any(k in info['keys'] for k in listed)}
            for c in sc:
                row[c] = t in set(map(tuple, sc[c]['predicted_transient']))
            rows.append(row)
    (P.OUT / 'rows_visibility.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')

    def block(rs, paired=True):
        rs = [r for r in rs if r['paired']] if paired else rs
        if not rs:
            return {'events': 0}
        ids = [r['case'] for r in rs]
        base = [(float(r['E']) + float(r['E_r2'])) / 2 for r in rs]
        out = {'events': len(rs), 'cases': len(set(ids)), 'recall_meanE': phase2.ci(base, ids),
               'recall_V_plus': phase2.ci([float(r['V_plus']) for r in rs], ids),
               'V_plus_minus_meanE': phase2.ci([float(r['V_plus']) - b for r, b in zip(rs, base)], ids)}
        if paired:
            out['recall_V_zero'] = phase2.ci([float(r['V_zero']) for r in rs], ids)
            out['V_plus_minus_V_zero'] = phase2.ci([float(r['V_plus']) - float(r['V_zero']) for r in rs], ids)
            out['V_zero_minus_meanE'] = phase2.ci([float(r['V_zero']) - b for r, b in zip(rs, base)], ids)
        return out

    shown_rows = [r for r in rows if r['shown']]
    other = [r for r in rows if not r['shown']]
    pr = [r for r in case_rows if r['paired']]
    pids, aids = [r['case'] for r in pr], [r['case'] for r in case_rows]
    report = {'model': P.MODEL, 'cases_all': len(case_rows), 'cases_paired': len(pr),
              'primary_shown_events_paired': block(shown_rows),
              'shown_events_known_paired': block([r for r in shown_rows if r['known']]),
              'shown_events_unknown_paired': block([r for r in shown_rows if not r['known']]),
              'shown_heavy_events_paired': block([r for r in shown_rows if r['heavy']]),
              'spillover_unshown_hidden_paired': block(other),
              'secondary_shown_events_all_cases': block(shown_rows, paired=False),
              'secondary_shown_known_all_cases': block([r for r in shown_rows if r['known']], paired=False),
              'secondary_shown_unknown_all_cases': block([r for r in shown_rows if not r['known']], paired=False),
              'sigma_L1_paired': {c: phase2.ci([r[c] for r in pr], pids) for c in ('E', 'E_r2', 'V_plus', 'V_zero')},
              'sigma_L1_V_plus_minus_V_zero': phase2.ci([r['V_plus'] - r['V_zero'] for r in pr], pids),
              'sigma_L1_V_zero_minus_meanE': phase2.ci([r['V_zero'] - (r['E'] + r['E_r2']) / 2 for r in pr], pids),
              'sigma_L1_V_plus_minus_meanE_all_cases': phase2.ci([r['V_plus'] - (r['E'] + r['E_r2']) / 2 for r in case_rows], aids)}
    (P.OUT / 'SUMMARY_visibility.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=300)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--analyze-only', action='store_true')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--base-url', help='switch endpoint for the remaining jobs; outputs stay in the same folder')
    ap.add_argument('--model-name', help='model name at that endpoint (recorded per output file)')
    ap.add_argument('--model', help='another model: outputs go under models/<model>/ (probe.configure)')
    cli = ap.parse_args()
    if cli.model:
        P.configure(cli.model, cli.base_url)
    if cli.base_url:
        P.BASE_URL = cli.base_url
    if cli.model_name:
        P.MODEL = cli.model_name
    sys.path.insert(0, str(HERE.parent / 'agent_v2'))
    import j3_dev
    cases = j3_dev.load_cases()
    shown = {cid: s for cid, c in cases.items() if (s := shown_species(c))}
    if cli.list:
        n_hidden = sum(1 for c in cases.values() if any(i['keys'] for i in P.hidden_events(c).values()))
        print(json.dumps({'cases_with_probeable_hidden': n_hidden, 'cases_with_hidden_shown': len(shown),
                          'cases_with_both_species': sum(1 for v in shown.values() if v[2]),
                          'examples': {k: (v[0], v[2]) for k, v in list(shown.items())[:6]}}, indent=1))
        return
    if cli.limit:
        shown = dict(list(shown.items())[:cli.limit])
    P.OUT.mkdir(parents=True, exist_ok=True)
    (P.OUT / 'visibility_shown.json').write_text(json.dumps(shown, indent=1))
    if not cli.analyze_only:
        jobs = [(c, cases[cid], s) for c in CONDS for cid, s in shown.items() if c == 'V_plus' or s[2]]
        random.Random(13).shuffle(jobs)
        done = 0
        with concurrent.futures.ThreadPoolExecutor(cli.workers) as pool:
            for fut in concurrent.futures.as_completed([pool.submit(job, *j) for j in jobs]):
                done += 1
                status = fut.result()
                if status.startswith('error') or done % 50 == 0:
                    print(json.dumps({'done': done, 'of': len(jobs), 'status': status}), flush=True)
    analyze(cases, shown)


if __name__ == '__main__':
    main()
