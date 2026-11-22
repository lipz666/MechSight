"""Stage 1a: FlowER vs LLM on the same FukuyamaBench cases and metrics.

FlowER: outputs/flower_compare_v2_20261031/{dev,heldout}/<case>.json (driver with the
        corrected scoring: whole beam graph kept, main-product reach, top-1 among
        finished paths). The v1 run (flower_compare_20261025) is superseded.
LLM:    held-out = confirmatory run (direct, direct@2; shortlist, shortlist@2),
        dev = P2/P3 probe (direct, direct@2..@4). Per case the LLM score is the
        mean over its samples. Same develop/fukuyama_metrics.py scoring.
Paired differences (LLM - FlowER) are bootstrapped over cases.
"""
from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from develop import fukuyama_metrics as fm
from develop import metrics

OUT = ROOT / 'outputs' / 'flower_compare_v2_20261031'
V1 = ROOT / 'outputs' / 'flower_compare_20261025'
HELD = ROOT / 'outputs' / 'heldout_fukuyama_20261024'
DEV = ROOT / 'outputs' / 'p23_planning_probe_20261024'
METRICS = ['sigma_L1', 'intermediates_in_order']


def llm_scores(case, folder, arms):
    vals = []
    for arm in arms:
        f = folder / arm / f"{case['case_id']}.json"
        if f.exists() and json.loads(f.read_text()).get('finished'):
            s = fm.score_case(case, json.loads(f.read_text())['actions_pgl'])
            vals.append(s)
    if not vals:
        return None
    return {m: statistics.mean(float(v[m]) for v in vals) for m in METRICS}


def ci(values, ids):
    lo, hi = metrics.clustered_bootstrap_ci(values, ids, seed=20261025)
    return [round(lo, 4), round(hi, 4)]


def main():
    reg = json.loads((HELD / 'REGISTRATION.json').read_text())
    held_cases = {c['case_id']: c for c in map(json.loads, (HELD / 'heldout.jsonl').read_text().splitlines())}
    clean = {c['case_id']: c for c in map(json.loads, (ROOT / 'outputs/fukuyama/clean.jsonl').read_text().splitlines())}
    dev_ids = json.loads((DEV / 'cases.json').read_text())
    sets = {'heldout_primary_123': ('heldout', reg['sets']['primary_T0_T1_T2'], held_cases, HELD / 'confirmatory'),
            'heldout_subsample60': ('heldout', json.loads((V1 / 'heldout_subsample60.json').read_text())['ids'], held_cases, HELD / 'confirmatory'),
            'heldout_T3_95': ('heldout', reg['sets']['sensitivity_T3'], held_cases, HELD / 'confirmatory'),
            'heldout_all_218': ('heldout', reg['sets']['all_218'], held_cases, HELD / 'confirmatory'),
            'dev_24': ('dev', dev_ids, clean, DEV)}
    report = {}
    for name, (folder, ids, pool, llm_dir) in sets.items():
        rows = []
        for cid in ids:
            f = OUT / folder / f'{cid}.json'
            if not f.exists():
                continue
            fl = json.loads(f.read_text())
            case = pool[cid]
            direct = llm_scores(case, llm_dir, ['direct', 'direct@2', 'direct@3', 'direct@4'] if folder == 'dev' else ['direct', 'direct@2'])
            short = llm_scores(case, llm_dir, ['shortlist', 'shortlist@2']) if folder == 'heldout' else None
            row = {'case_id': cid, 'flower_error': fl.get('error'), 'flower_paths': fl.get('n_paths', 0),
                   'flower_timed_out': fl.get('timed_out'),
                   'flower_reaches_target': bool(fl.get('any_reaches_target')),
                   'flower_reaches_main_product': bool(fl.get('any_reaches_main_product'))}
            for mode in ('top1', 'target'):
                sel = fl.get(mode) or {}
                for m in METRICS:
                    row[f'flower_{mode}_{m}'] = float(bool(sel.get(m)))
            row['flower_any_sigma_L1'] = float(bool(fl.get('any_sigma_L1')))
            row['flower_any_intermediates_in_order'] = float(bool(fl.get('any_in_order')))
            for m in METRICS:
                row[f'llm_direct_{m}'] = direct[m] if direct else None
                row[f'llm_shortlist_{m}'] = short[m] if short else None
            rows.append(row)
        block = {'cases_with_flower_result': len(rows), 'of': len(ids),
                 'flower_errors': sum(bool(r['flower_error']) for r in rows),
                 'flower_timeouts': sum(bool(r.get('flower_timed_out')) for r in rows),
                 'flower_no_path': sum(r['flower_paths'] == 0 for r in rows),
                 'flower_any_path_reaches_target': round(statistics.mean(r['flower_reaches_target'] for r in rows), 4) if rows else None,
                 'flower_any_path_reaches_main_product': round(statistics.mean(r['flower_reaches_main_product'] for r in rows), 4) if rows else None}
        for m in METRICS:
            for col in (f'flower_top1_{m}', f'flower_target_{m}', f'flower_any_{m}', f'llm_direct_{m}', f'llm_shortlist_{m}'):
                vals = [r[col] for r in rows if r.get(col) is not None]
                block[col] = round(statistics.mean(vals), 4) if vals else None
            for base in (f'flower_top1_{m}', f'flower_target_{m}', f'flower_any_{m}'):
                keys = [r for r in rows if r[f'llm_direct_{m}'] is not None]
                d = [r[f'llm_direct_{m}'] - r[base] for r in keys]
                if d:
                    block[f'llm_direct - {base}'] = {'delta': round(statistics.mean(d), 4), 'ci95': ci(d, [r['case_id'] for r in keys]),
                                                     'llm_better': sum(x > 0 for x in d), 'flower_better': sum(x < 0 for x in d)}
        report[name] = block
        (OUT / f'compare_rows_{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (OUT / 'COMPARE_1a.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
