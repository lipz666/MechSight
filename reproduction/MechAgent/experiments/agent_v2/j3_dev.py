"""J3: MechAgent-v2 vs plain LLM on FukuyamaBench (development, both models).

Cases: T0+T1+T2 held-out (123) + dev (24) = 147, all previously used, so
this is development. Motif hints come from the J2 cross-validation library
of the fold that never saw the case.

Phase 1 (generation): direct@1..3 and hint, per model and case.
Phase 2 (judging): direct@1 vs hint in both orders.
Analysis: sigma_L1 and intermediates_in_order for
  direct (sample 1), vote@3 (compute-matched), hint alone, v2 (judged), and
  the best-of-two upper bound; paired case-bootstrap CIs vs direct and vote@3.
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
sys.path.insert(0, str(HERE))
import agent_v2 as A
import common as C
import motifs as M
from common import ROOT, chem, fm
from develop import metrics

OUT = ROOT / 'outputs' / 'agent_v2_J3'
J2 = ROOT / 'outputs' / 'agent_v2_J2'
HINT_K = 12


def load_cases():
    pool = {}
    for src in [ROOT / 'outputs/fukuyama/clean.jsonl', ROOT / 'outputs/heldout_fukuyama_20261024/heldout.jsonl']:
        for line in src.read_text().splitlines():
            c = json.loads(line)
            pool[c['case_id']] = c
    reg = json.loads((ROOT / 'outputs/heldout_fukuyama_20261024/REGISTRATION.json').read_text())
    ids = sorted(set(reg['sets']['primary_T0_T1_T2']) | set(json.loads((ROOT / 'outputs/p23_planning_probe_20261024/cases.json').read_text())))
    return {cid: pool[cid] for cid in ids}


def hint_keys(case, libs, folds):
    lib = libs[folds[case['case_id']]]
    return M.select(lib.rank(case), HINT_K, 0.0)


def gen_job(model_name, arm, case, libs, folds):
    dest = OUT / model_name / arm / f"{case['case_id']}.json"
    if dest.exists():
        return 'cached'
    model = C.Model(model_name, OUT / model_name / arm / 'traces' / f"{case['case_id']}.jsonl")
    keys = hint_keys(case, libs, folds) if arm == 'hint' else None
    try:
        actions, errors = A.generate(model, case, A.motif_hint(keys) if keys else None, phase=arm)
    except Exception as exc:
        return f'error:{type(exc).__name__}'
    rec = {'case_id': case['case_id'], 'arm': arm, 'model': model_name, 'actions_pgl': actions, 'parse_errors': errors,
           'hint_keys': [list(k) for k in keys] if keys else None, 'tokens': model.tokens, 'calls': len(model.events)}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1))
    return 'ok'


def judge_job(model_name, case, order):
    dest = OUT / model_name / 'judge' / f"{case['case_id']}_{order}.json"
    if dest.exists():
        return 'cached'
    base = json.loads((OUT / model_name / 'direct@1' / f"{case['case_id']}.json").read_text())['actions_pgl']
    hint = json.loads((OUT / model_name / 'hint' / f"{case['case_id']}.json").read_text())['actions_pgl']
    if sigma_equal(case, base, hint):
        rec = {'case_id': case['case_id'], 'order': order, 'choice': None, 'identical': True, 'tokens': 0}
    else:
        tb, th = A.render_actions(case, base), A.render_actions(case, hint)
        a, b = (th, tb) if order == 'hint_first' else (tb, th)
        model = C.Model(model_name, OUT / model_name / 'judge' / 'traces' / f"{case['case_id']}_{order}.jsonl", max_tokens=4096)
        try:
            choice = A.judge(model, case, a, b)
        except Exception as exc:
            return f'error:{type(exc).__name__}'
        picked_hint = (choice == 'A') if order == 'hint_first' else (choice == 'B')
        rec = {'case_id': case['case_id'], 'order': order, 'choice': choice, 'identical': False,
               'picked_hint': picked_hint if choice else None, 'tokens': model.tokens}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1))
    return 'ok'


def sigma_equal(case, p, q):
    return A.sigma_signature(case, p) == A.sigma_signature(case, q)


def run_pool(jobs, fn, workers):
    done = 0
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futs = [pool.submit(fn, *j) for j in jobs]
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            status = fut.result()
            if status.startswith('error') or done % 50 == 0:
                print(json.dumps({'done': done, 'of': len(jobs), 'status': status}), flush=True)


def analyze(cases, models):
    report = {}
    for m in models:
        rows = {}
        for cid, case in cases.items():
            def load(arm):
                f = OUT / m / arm / f'{cid}.json'
                return json.loads(f.read_text())['actions_pgl'] if f.exists() else None
            d = [load(f'direct@{i}') for i in (1, 2, 3)]
            h = load('hint')
            js = [OUT / m / 'judge' / f'{cid}_{o}.json' for o in ('hint_first', 'base_first')]
            if any(x is None for x in d + [h]) or not all(j.exists() for j in js):
                continue
            jr = [json.loads(j.read_text()) for j in js]
            take_hint = (not jr[0]['identical']) and all(r.get('picked_hint') for r in jr)
            v2 = h if take_hint else d[0]
            score = lambda acts: fm.score_case(case, acts)
            s_d, s_h, s_v2 = score(d[0]), score(h), score(v2)
            s_vote = score(d[A.vote(case, d)])
            rows[cid] = {}
            for metric in ('sigma_L1', 'intermediates_in_order'):
                rows[cid][metric] = {'direct': float(s_d[metric]), 'vote3': float(s_vote[metric]), 'hint': float(s_h[metric]),
                                     'v2': float(s_v2[metric]), 'best_of_two': float(s_d[metric] or s_h[metric])}
            rows[cid]['took_hint'] = take_hint
            rows[cid]['judge_split'] = (not jr[0]['identical']) and len({r.get('picked_hint') for r in jr}) > 1
        ids = sorted(rows)
        block = {'cases': len(ids), 'took_hint_rate': round(statistics.mean(rows[c]['took_hint'] for c in ids), 4) if ids else None,
                 'judge_split_rate': round(statistics.mean(rows[c]['judge_split'] for c in ids), 4) if ids else None}
        for metric in ('sigma_L1', 'intermediates_in_order'):
            block[metric] = {arm: round(statistics.mean(rows[c][metric][arm] for c in ids), 4) for arm in ('direct', 'vote3', 'hint', 'v2', 'best_of_two')} if ids else {}
            for a, b in (('direct', 'v2'), ('vote3', 'v2'), ('direct', 'hint'), ('direct', 'vote3')):
                diff = [rows[c][metric][b] - rows[c][metric][a] for c in ids]
                if diff:
                    lo, hi = metrics.clustered_bootstrap_ci(diff, ids, seed=20261030)
                    block[metric][f'{b} - {a}'] = {'delta': round(statistics.mean(diff), 4), 'ci95': [round(lo, 4), round(hi, 4)],
                                                   'better': sum(x > 0 for x in diff), 'worse': sum(x < 0 for x in diff)}
        report[m] = block
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=C.MODELS)
    ap.add_argument('--workers', type=int, default=96)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--analyze-only', action='store_true')
    cli = ap.parse_args()
    cases = load_cases()
    if cli.limit:
        cases = dict(list(cases.items())[:cli.limit])
    libs = {f: M.MotifLibrary.from_json(json.loads((J2 / f'library_fold{f}.json').read_text())) for f in range(5)}
    folds = json.loads((J2 / 'folds.json').read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps({'cases': len(cases), 'models': cli.models}), flush=True)
    if not cli.analyze_only:
        gen = [(m, arm, c, libs, folds) for m in cli.models for c in cases.values() for arm in ('direct@1', 'direct@2', 'direct@3', 'hint')]
        random.Random(2).shuffle(gen)
        run_pool(gen, gen_job, cli.workers)
        jud = [(m, c, o) for m in cli.models for c in cases.values() for o in ('hint_first', 'base_first')
               if (OUT / m / 'direct@1' / f"{c['case_id']}.json").exists() and (OUT / m / 'hint' / f"{c['case_id']}.json").exists()]
        run_pool(jud, judge_job, cli.workers)
    analyze(cases, cli.models)


if __name__ == '__main__':
    main()
