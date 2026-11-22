"""J1: can an LLM pick the accepted mechanism when shown it next to an LLM-written one?

Cases: FukuyamaBench cases (dev 24 + held-out 218, all already used, so this
is a development probe) whose existing LLM path (gemini-3.8-flash-high,
direct, sample 1) does not hit the annotated intermediates in order. Each
pair (GT path, LLM path) is rendered in one format and shown as A/B in both
orders; the judge answers <choice>A</choice> or <choice>B</choice>.

Reported per judge model: accuracy over calls, per-case outcome (correct in
both orders / wrong in both / order-dependent), and position bias.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import random
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C
from common import ROOT, chem, fm

OUT = ROOT / 'outputs' / 'agent_v2_J1'
SYSTEM = ('You are an expert physical organic chemist who reviews proposed arrow-pushing mechanisms. Atom labels '
          '(e.g. C12, O4, H900011) refer to the INITIAL state; every mechanism below is atom- and charge-balanced and '
          'executable from it.')
TASK = ('Two candidate stepwise mechanisms, A and B, are given for the same transformation. Each step lists the bonds '
        'formed and broken, bond-order and formal-charge changes, and the species formed or changed. Decide which '
        'mechanism is the chemically correct one under the given conditions, judging the plausibility of every '
        'intermediate, the order of events, and whether steps are concerted or stepwise. Reason briefly, then answer '
        'with exactly <choice>A</choice> or <choice>B</choice>.')


def load_pairs():
    pool = {}
    for src in [ROOT / 'outputs/fukuyama/clean.jsonl', ROOT / 'outputs/heldout_fukuyama_20261024/heldout.jsonl']:
        for line in src.read_text().splitlines():
            c = json.loads(line)
            pool[c['case_id']] = c
    dev = set(json.loads((ROOT / 'outputs/p23_planning_probe_20261024/cases.json').read_text()))
    pairs = []
    for cid, case in sorted(pool.items()):
        f = (ROOT / 'outputs/p23_planning_probe_20261024/direct' / f'{cid}.json') if cid in dev else \
            (ROOT / 'outputs/heldout_fukuyama_20261024/confirmatory/direct' / f'{cid}.json')
        if not f.exists():
            continue
        actions = json.loads(f.read_text())['actions_pgl']
        score = fm.score_case(case, actions)
        if score['intermediates_in_order'] or score['executed_steps'] == 0:
            continue
        initial = chem.parse_pgl_literal(case['initial_state_pgl'])
        llm_states, _ = fm.replay(case['initial_state_pgl'], actions)
        pairs.append({'case_id': cid, 'case': case, 'llm_sigma_L1': score['sigma_L1'],
                      'gt_text': C.render_path(initial, C.gt_states(case)),
                      'llm_text': C.render_path(initial, llm_states)})
    return pairs


def ask(model_name, pair, order):
    dest = OUT / model_name / f"{pair['case_id']}_{order}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    case = pair['case']
    a, b = (pair['gt_text'], pair['llm_text']) if order == 'gt_first' else (pair['llm_text'], pair['gt_text'])
    conditions = '; '.join(f"{t['role']}: {t['text']}" for t in case['conditions'] if t['text']) or 'not given'
    user = (f"INITIAL:\n{case['initial_state_pgl']}\nTARGET:\n{case['target_state_pgl']}\nConditions: {conditions}\n\n"
            f"Mechanism A:\n{a}\n\nMechanism B:\n{b}\n\n{TASK}")
    model = C.Model(model_name, OUT / model_name / 'traces' / f"{pair['case_id']}_{order}.jsonl", max_tokens=4096)
    try:
        answer = model.complete([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}], 'judge')
        found = re.findall(r'<choice>\s*([AB])\s*</choice>', answer)
        choice = found[-1] if found else None
        error = None
    except Exception as exc:
        choice, error = None, f'{type(exc).__name__}: {str(exc)[:200]}'
    gt_letter = 'A' if order == 'gt_first' else 'B'
    rec = {'case_id': pair['case_id'], 'model': model_name, 'order': order, 'choice': choice,
           'picked_gt': (choice == gt_letter) if choice else None, 'llm_sigma_L1': pair['llm_sigma_L1'],
           'error': error, 'tokens': model.tokens}
    if error is None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(rec, indent=1))
    return rec


def summarize(pairs, models):
    report = {}
    for m in models:
        recs = [json.loads(p.read_text()) for p in (OUT / m).glob('*_*.json')]
        by_case = collections.defaultdict(dict)
        for r in recs:
            by_case[r['case_id']][r['order']] = r
        calls = [r for r in recs if r['choice']]
        outcome = collections.Counter()
        for cid, d in by_case.items():
            if len(d) < 2 or not all(x['choice'] for x in d.values()):
                outcome['incomplete'] += 1
                continue
            g = [d['gt_first']['picked_gt'], d['llm_first']['picked_gt']]
            outcome['gt_both_orders' if all(g) else 'llm_both_orders' if not any(g) else 'order_dependent'] += 1
        full = sum(v for k, v in outcome.items() if k != 'incomplete')
        report[m] = {'pairs': len(by_case), 'calls_with_choice': len(calls),
                     'accuracy_per_call': round(statistics.mean(r['picked_gt'] for r in calls), 4) if calls else None,
                     'position_A_rate': round(statistics.mean(r['choice'] == 'A' for r in calls), 4) if calls else None,
                     'case_outcomes': dict(outcome),
                     'consistent_correct_rate': round(outcome['gt_both_orders'] / full, 4) if full else None,
                     'order_consistency': round((outcome['gt_both_orders'] + outcome['llm_both_orders']) / full, 4) if full else None,
                     'accuracy_when_llm_sigma_correct': round(statistics.mean(r['picked_gt'] for r in calls if r['llm_sigma_L1']), 4)
                     if any(r['llm_sigma_L1'] for r in calls) else None,
                     'accuracy_when_llm_sigma_wrong': round(statistics.mean(r['picked_gt'] for r in calls if not r['llm_sigma_L1']), 4)
                     if any(not r['llm_sigma_L1'] for r in calls) else None,
                     'mean_tokens_per_call': round(statistics.mean(r['tokens'] for r in calls)) if calls else None}
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=C.MODELS)
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--summarize-only', action='store_true')
    cli = ap.parse_args()
    pairs = load_pairs()
    if cli.limit:
        pairs = pairs[:cli.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'pairs.json').write_text(json.dumps([{'case_id': p['case_id'], 'llm_sigma_L1': p['llm_sigma_L1']} for p in pairs], indent=1))
    print(json.dumps({'pairs': len(pairs), 'llm_sigma_correct_among_them': sum(p['llm_sigma_L1'] for p in pairs)}), flush=True)
    if not cli.summarize_only:
        jobs = [(m, p, o) for m in cli.models for p in pairs for o in ('gt_first', 'llm_first')]
        random.Random(1).shuffle(jobs)
        done = 0
        with concurrent.futures.ThreadPoolExecutor(cli.workers) as pool:
            for fut in concurrent.futures.as_completed([pool.submit(ask, *j) for j in jobs]):
                done += 1
                r = fut.result()
                if r.get('error') or done % 50 == 0:
                    print(json.dumps({'done': done, 'of': len(jobs), 'last': {k: r.get(k) for k in ('case_id', 'model', 'choice', 'error')}}), flush=True)
    summarize(pairs, cli.models)


if __name__ == '__main__':
    main()
