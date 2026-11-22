"""MechSight pilot (development set): make hidden intermediates visible before pushing arrows.

Pipeline per case (gemini-3.8-flash):
  1 hypothesize  candidate intermediates = union of the model's own free-text
                 lists (conditions S and G of the inert-knowledge probe), deduplicated
                 by sigma-graph key, endpoint species removed
  2 expose       executable PGL mechanism with the candidates listed as possible
                 intermediates (arm M)
  3 compare      pairwise judge between the direct path E and M, both orders;
                 M is taken only if preferred in both orders (arm MS = MechSight)
Compute-matched baseline: the same judge between E and E_r2 (arm DS).
Registration: outputs/mechsight_pilot_20261103/REGISTRATION.json.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'inert_knowledge'))
import phase2  # noqa: E402
import probe as P  # noqa: E402
from probe import A, C, chem, fm  # noqa: E402

OUT = P.ROOT / 'outputs' / 'mechsight_pilot_20261103'
PROBE = P.OUT  # inert-knowledge outputs: E, E_r2, S, G
CAND_HINT = ('Possible intermediates proposed from chemical knowledge (SMILES, atom labels not given). Some may be '
             'incorrect, and the mechanism may need intermediates that are not listed. Use a listed species only if '
             'it is chemically justified under the conditions:\n{lines}')


def load(folder, cid):
    f = folder / f'{cid}.json'
    return json.loads(f.read_text()) if f.exists() else None


def candidates(case):
    """Model's own free-text intermediates (S then G), unique by sigma key, endpoints removed."""
    initial = chem.parse_pgl_literal(case['initial_state_pgl'])
    target = chem.parse_pgl_literal(case['target_state_pgl'])
    endpoint = {P.state_component_keys(s, c)[0] for s in (initial, target) for c in C._components(s)}
    seen, out = set(), []
    for cond in ('S', 'G'):
        rec = load(PROBE / cond, case['case_id'])
        for smi in (rec or {}).get('intermediates', []):
            keys = [k for k, _ in P.smiles_keys(smi)]
            if not keys or all(k in endpoint for k in keys):
                continue
            key = tuple(sorted(keys))
            if key not in seen:
                seen.add(key)
                out.append(smi)
    return out


def model(folder, cid):
    return C.Model(P.MODEL, folder / 'traces' / f'{cid}.jsonl', base_url=P.BASE_URL, max_tokens=P.MAX_TOKENS)


def gen_job(case):
    dest = OUT / 'M' / f"{case['case_id']}.json"
    if dest.exists():
        return 'cached'
    cands = candidates(case)
    hint = CAND_HINT.format(lines='\n'.join(cands)) if cands else None
    try:
        actions, errors = phase2.generate(model(OUT / 'M', case['case_id']), A.user_prompt(case, hint=hint), 'M')
    except Exception as exc:
        return f'error:{type(exc).__name__}'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({'case_id': case['case_id'], 'candidates': cands, 'actions_pgl': actions,
                                'parse_errors': errors, 'model': P.MODEL, 'base_url': P.BASE_URL}, indent=1))
    return 'ok'


def judge_job(pair, case, order):
    """pair 'EM' (E vs M) or 'EE' (E vs E_r2); order 'alt_first' or 'base_first'."""
    cid = case['case_id']
    dest = OUT / f'judge_{pair}' / f'{cid}_{order}.json'
    if dest.exists():
        return 'cached'
    base = load(PROBE / 'E', cid)['actions_pgl']
    alt = (load(OUT / 'M', cid) if pair == 'EM' else load(PROBE / 'E_r2', cid))['actions_pgl']
    if A.sigma_signature(case, base) == A.sigma_signature(case, alt):
        rec = {'identical': True, 'picked_alt': None}
    else:
        tb, ta = A.render_actions(case, base), A.render_actions(case, alt)
        a, b = (ta, tb) if order == 'alt_first' else (tb, ta)
        try:
            choice = A.judge(model(OUT / f'judge_{pair}', f'{cid}_{order}'), case, a, b)
        except Exception as exc:
            return f'error:{type(exc).__name__}'
        rec = {'identical': False, 'choice': choice,
               'picked_alt': ((choice == 'A') if order == 'alt_first' else (choice == 'B')) if choice else None}
    rec.update({'case_id': cid, 'order': order, 'model': P.MODEL, 'base_url': P.BASE_URL})
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1))
    return 'ok'


def pool(fn, jobs, workers):
    random.Random(17).shuffle(jobs)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(workers) as ex:
        for fut in concurrent.futures.as_completed([ex.submit(fn, *j) for j in jobs]):
            done += 1
            status = fut.result()
            if status.startswith('error') or done % 50 == 0:
                print(json.dumps({'fn': fn.__name__, 'done': done, 'of': len(jobs), 'status': status}), flush=True)


def take_alt(pair, cid):
    js = [load(OUT / f'judge_{pair}', f'{cid}_{o}') for o in ('alt_first', 'base_first')]
    if any(j is None for j in js):
        return None
    return (not js[0]['identical']) and all(j.get('picked_alt') for j in js)


def analyze(cases):
    rows, events = [], []
    for cid, case in cases.items():
        e, r2, m = load(PROBE / 'E', cid), load(PROBE / 'E_r2', cid), load(OUT / 'M', cid)
        t_em, t_ee = take_alt('EM', cid), take_alt('EE', cid)
        if None in (e, r2, m) or t_em is None or t_ee is None:
            continue
        paths = {'E': e['actions_pgl'], 'E_r2': r2['actions_pgl'], 'M': m['actions_pgl'],
                 'MS': m['actions_pgl'] if t_em else e['actions_pgl'], 'DS': r2['actions_pgl'] if t_ee else e['actions_pgl']}
        sc = {k: fm.score_case(case, v) for k, v in paths.items()}
        row = {'case': cid, 'took_M': t_em, 'took_r2': t_ee, 'n_cand': len(m['candidates'])}
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
    arms = ['E', 'E_r2', 'DS', 'M', 'MS']
    rep = {'model': P.MODEL, 'cases': len(rows), 'took_M_rate': round(statistics.mean(r['took_M'] for r in rows), 3),
           'took_r2_rate': round(statistics.mean(r['took_r2'] for r in rows), 3),
           'candidates_per_case_median': statistics.median(r['n_cand'] for r in rows)}
    for metric in ('sigma', 'order'):
        rep[metric] = {a: phase2.ci([r[f'{a}_{metric}'] for r in rows], ids) for a in arms}
        for a, b in (('DS', 'MS'), ('E', 'MS'), ('E', 'M'), ('DS', 'M')):
            d = [r[f'{b}_{metric}'] - r[f'{a}_{metric}'] for r in rows]
            rep[metric][f'{b} - {a}'] = phase2.ci(d, ids) + [sum(x > 0 for x in d), sum(x < 0 for x in d)]
    rep['spurious_transients_per_case'] = {a: round(statistics.mean(r[f'{a}_spur'] for r in rows), 3) for a in arms}
    for name, sel in (('hidden_all', lambda e: True), ('hidden_heavy', lambda e: e['heavy'])):
        es = [e for e in events if sel(e)]
        if not es:
            rep[name] = {'events': 0}
            continue
        rep[name] = {'events': len(es), **{a: phase2.ci([float(e[a]) for e in es], [e['case'] for e in es]) for a in arms}}
    (OUT / 'SUMMARY.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=150)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--analyze-only', action='store_true')
    ap.add_argument('--base-url')
    ap.add_argument('--model-name')
    cli = ap.parse_args()
    if cli.base_url:
        P.BASE_URL = cli.base_url
    if cli.model_name:
        P.MODEL = cli.model_name
    sys.path.insert(0, str(HERE.parent / 'agent_v2'))
    import j3_dev
    cases = j3_dev.load_cases()
    if cli.limit:
        cases = dict(list(cases.items())[:cli.limit])
    OUT.mkdir(parents=True, exist_ok=True)
    if not cli.analyze_only:
        pool(gen_job, [(c,) for c in cases.values()], cli.workers)
        jobs = [(p, c, o) for p in ('EM', 'EE') for c in cases.values() for o in ('alt_first', 'base_first')
                if p == 'EE' or (OUT / 'M' / f"{c['case_id']}.json").exists()]
        pool(judge_job, jobs, cli.workers)
    analyze(cases)


if __name__ == '__main__':
    main()
