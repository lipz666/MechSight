"""Recognition among the model's OWN proposals (hardest negatives) on the 115 GT-prefix steps.

Candidates = distinct valid states proposed by the model for that step (its 5-item list
plus its 5 independent samples, from recall.py), shuffled; the model then picks with the
same prompt as experiments/scale_probe (direct arm). A step counts as correct only if the
pick equals the GT next state, so steps whose GT is not among the proposals are failures.
Compared per step with: one sample (sample r=0) and the list's own first item.
"""
from __future__ import annotations

import concurrent.futures
import json
import random
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recall as R
from recall import C, MV, P, chem, fm, metrics

OUT = R.OUT / R.MODEL / 'select_own'


def build(item, case):
    cur = chem.parse_pgl_literal(item['current'])
    gt = R.gt_chain(case)
    props = json.loads((R.OUT / R.MODEL / 'list' / f"{item['case_id']}_{item['step']}_0.json").read_text())['proposals']
    for r in range(R.K):
        f = R.OUT / R.MODEL / 'sample' / f"{item['case_id']}_{item['step']}_{r}.json"
        if f.exists():
            props += json.loads(f.read_text())['proposals']
    states, seen = [], set()
    for p in props:
        new = R.apply_changes(cur, p['text'])
        if new is None or not C.step_text(cur, new)[0]:
            continue
        sig = MV.signature(new)
        if sig not in seen:
            seen.add(sig)
            states.append(new)
    random.Random(f"{item['case_id']}:{item['step']}:own").shuffle(states)
    cls = [R.classify(s, gt, item['step']) for s in states]
    gt_indices = [k for k, c in enumerate(cls) if c == 'next']  # several proposals can equal the GT state
    return {**item, 'candidates': [repr(s) for s in states], 'gt_index': gt_indices[0] if gt_indices else None,
            'gt_indices': gt_indices, 'later_indices': [k for k, c in enumerate(cls) if c == 'later']}


def job(case, item):
    dest = OUT / f"{item['case_id']}_{item['step']}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    n = len(item['candidates'])
    if n == 1:
        choice = 0
    else:
        model = C.Model(R.MODEL, OUT / 'traces' / f"{item['case_id']}_{item['step']}.jsonl", base_url=R.BASE_URL, max_tokens=8192)
        try:
            answer = model.complete([{'role': 'system', 'content': P.SYSTEM},
                                     {'role': 'user', 'content': P.context(case, item) + P.DIRECT_ASK}], 'select_own')
        except Exception as exc:
            return {'error': type(exc).__name__}
        m = re.findall(r'<choice>\s*\[?\s*(\d+)\s*\]?\s*</choice>', answer)
        choice = int(m[-1]) if m and int(m[-1]) < n else None
    rec = {'case_id': item['case_id'], 'step': item['step'], 'n_candidates': n, 'gt_index': item['gt_index'],
           'choice': choice, 'correct': choice is not None and choice == item['gt_index']}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1))
    return rec


def main():
    cases = R.pool_cases()
    items = [it for it in json.loads(R.ITEMS.read_text()) if len(it['candidates']) == P.N_CAND]
    built = [build(it, cases[it['case_id']]) for it in items]
    built = [b for b in built if b['candidates']]
    by_key = {(b['case_id'], b['step']): b for b in built}
    with concurrent.futures.ThreadPoolExecutor(110) as ex:
        recs = [f.result() for f in [ex.submit(job, cases[b['case_id']], b) for b in built]]
    errors = [r for r in recs if 'error' in r]
    recs = {(r['case_id'], r['step']): r for r in recs if 'error' not in r}
    rows, ids = [], []
    for it in items:
        key = (it['case_id'], it['step'])
        own = recs.get(key)
        b = by_key.get(key)
        s0 = json.loads((R.OUT / R.MODEL / 'sample' / f'{key[0]}_{key[1]}_0.json').read_text())['proposals']
        l0 = json.loads((R.OUT / R.MODEL / 'list' / f'{key[0]}_{key[1]}_0.json').read_text())['proposals']
        pick = own['choice'] if own else None
        rows.append({'select_own': float(pick is not None and pick in b['gt_indices']),
                     'one_sample': float(bool(s0) and s0[0]['class'] == 'next'),
                     'list_first': float(bool(l0) and l0[0]['class'] == 'next'),
                     'select_own_on_path': float(pick is not None and pick in b['gt_indices'] + b['later_indices']),
                     'one_sample_on_path': float(bool(s0) and s0[0]['class'] in ('next', 'later')),
                     'gt_in_pool': float(bool(b and b['gt_indices'])),
                     'n_candidates': own['n_candidates'] if own else 0})
        ids.append(key[0])
    report = {'steps': len(rows), 'errors': len(errors),
              'candidates_median': statistics.median(r['n_candidates'] for r in rows),
              **{k: round(statistics.mean(r[k] for r in rows), 4) for k in ('gt_in_pool', 'select_own', 'one_sample', 'list_first',
                                                                 'select_own_on_path', 'one_sample_on_path')}}
    inpool = [r for r in rows if r['gt_in_pool']]
    report['select_own_given_gt_in_pool'] = round(statistics.mean(r['select_own'] for r in inpool), 4)
    for base in ('one_sample', 'list_first'):
        d = [r['select_own'] - r[base] for r in rows]
        lo, hi = metrics.clustered_bootstrap_ci(d, ids, seed=20261101)
        report[f'select_own - {base}'] = {'delta': round(statistics.mean(d), 4), 'ci95': [round(lo, 4), round(hi, 4)],
                                          'better': sum(x > 0 for x in d), 'worse': sum(x < 0 for x in d)}
    (R.OUT / 'SELECT_OWN.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
