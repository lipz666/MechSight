"""Rescore every historical FukuyamaBench run with develop/fukuyama_metrics.py.

Primary metrics: sigma_L1 and resonance-normalized intermediates_in_order.
The legacy state-difference L1/L2/L3 (develop/score_external_benchmark.py,
normalized reading) is reported next to them so that each historical number
can be traced. Only runs whose endpoints equal the current clean.jsonl are
included; see SOURCES for what was left out and why.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from develop import fukuyama_metrics as fm
from develop import metrics
from develop import score_external_benchmark as sx
from develop import standalone_chemistry as chem

CLEAN = ROOT / 'outputs' / 'fukuyama' / 'clean.jsonl'

# name -> (path, format, setting, max repeat). Excluded: fuku_g38 (pre-ledger
# conversion; 53/60 initial states differ from clean.jsonl), newjudge/
# fukuyama_r2 (byte-identical actions to bench r2), smoke runs and the partial
# full_snapshot, expA_paper_protocol (SMILES outputs, different task format).
SOURCES = {
    'agent_skills23_r2': ('outputs/bench_fukuyama_g38/full_r2.jsonl', 'rollout', 'target given; develop-2, 23 skills', 2),
    'agent_skills79_r2': ('outputs/bench_fukuyama_skills79/full.jsonl', 'rollout', 'target given; develop-2, 79 skills', 2),
    'agent_early_v2': ('outputs/fuku_g38high_v2/full.jsonl', 'rollout', 'target given; earlier develop-2 run', 1),
    'agent_forward': ('outputs/fuku_forward_g38high/full.jsonl', 'rollout', 'TARGET HIDDEN (forward); develop-2', 1),
    'p23_direct': ('outputs/p23_planning_probe_20261024/direct', 'p23', 'target given; 24-case subset', 1),
    'p23_interm': ('outputs/p23_planning_probe_20261024/interm', 'p23', 'target given; 24-case subset', 1),
    'p23_latent': ('outputs/p23_planning_probe_20261024/latent', 'p23', 'target given; 24-case subset', 1),
    'p23_oracle_lat': ('outputs/p23_planning_probe_20261024/oracle_lat', 'p23', 'DIAGNOSTIC oracle hint; 24-case subset', 1),
}

PRIMARY = ['sigma_L1', 'intermediates_in_order']
SECONDARY = ['sigma_L2', 'intermediates_in_order_strict', 'final_state_match', 'legacy_L1', 'legacy_L2', 'legacy_L3', 'legacy_bonds_only_L1']


def load(source):
    path, fmt, _, max_repeat = source
    path = ROOT / path
    if fmt == 'rollout':
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            repeat = int(r['reaction'].get('repeat_index') or 1)
            if repeat > max_repeat:
                continue
            yield (r['reaction']['reaction_id'].split('::')[-1], repeat,
                   [a['action_pgl'] for a in r.get('actions', []) if a.get('accepted_into_history')])
    else:
        for f in sorted(path.glob('*.json')):
            r = json.loads(f.read_text())
            if r.get('finished'):
                yield r['case_id'], 1, r['actions_pgl']


def legacy(case, actions):
    state = chem.parse_pgl_literal(case['initial_state_pgl'])
    predicted = []
    for text in actions:
        try:
            action, parsed = chem.extract_action_payload(f'<action>{text}</action>')
            before = chem.state_signature(state)
            state, _ = chem.apply_action(state, action)
        except Exception:
            break
        predicted.append(sx._step_from_signatures(before, chem.state_signature(state)))
    gold = [sx._step_from_clean(s) for s in case['gt_step_changes']]
    p = [sx.to_counter(s, normalized=True) for s in predicted]
    g = [sx.to_counter(s, normalized=True) for s in gold]
    cmp = metrics.compare_path(sx.as_actions(p), sx.as_actions(g))
    bonds = metrics.compare_path(sx.as_actions(p, bonds_only=True), sx.as_actions(g, bonds_only=True))
    return {'legacy_L1': cmp.transfer_set_match, 'legacy_L2': cmp.transfer_sequence_match,
            'legacy_L3': cmp.step_partition_match, 'legacy_bonds_only_L1': bonds.transfer_set_match}


def rate(rows, key):
    values = [int(bool(r[key])) for r in rows]
    lo, hi = metrics.clustered_bootstrap_ci(values, [r['case_id'] for r in rows], seed=20261024)
    return {'rate': sum(values) / len(values), 'count': sum(values), 'ci95': [lo, hi]}


def summarize(rows):
    out = {'rollouts': len(rows), 'cases': len({r['case_id'] for r in rows})}
    for key in PRIMARY + SECONDARY:
        out[key] = rate(rows, key)
    out['by_partition'] = {p: {k: statistics.mean(int(bool(r[k])) for r in rows if r['partition'] == p) for k in PRIMARY + ['legacy_L1']}
                           for p in sorted({r['partition'] for r in rows})}
    out['intermediate_recall'] = sum(r['intermediates_hit'] for r in rows) / sum(r['gold_steps'] for r in rows)
    m, p, g = (sum(r[k] for r in rows) for k in ('sigma_matched', 'sigma_predicted', 'sigma_gold'))
    out['sigma_event_f1'] = 2 * m / (p + g) if p + g else 0.0
    gt_t = sum(len(r['gold_transient']) for r in rows)
    hit_t = sum(len(set(map(tuple, r['gold_transient'])) & set(map(tuple, r['predicted_transient']))) for r in rows)
    out['transient_recall'] = hit_t / gt_t if gt_t else None
    # Where the legacy metric and the new ones disagree, and why.
    out['reclassified'] = {
        'in_order_but_legacy_L1_fail': sum(r['intermediates_in_order'] and not r['legacy_L1'] for r in rows),
        'of_which_finer_than_gt': sum(r['intermediates_in_order'] and not r['legacy_L1'] and r['executed_steps'] > r['gold_steps'] for r in rows),
        'in_order_only_with_resonance': sum(r['intermediates_in_order'] and not r['intermediates_in_order_strict'] for r in rows),
        'legacy_L1_but_not_in_order': sum(r['legacy_L1'] and not r['intermediates_in_order'] for r in rows),
        'sigma_L1_but_not_in_order': sum(r['sigma_L1'] and not r['intermediates_in_order'] for r in rows),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'fukuyama_rescore_20261024')
    args = ap.parse_args()
    cases = {c['case_id']: c for c in map(json.loads, CLEAN.read_text().splitlines())}
    args.output.mkdir(parents=True, exist_ok=True)
    report, all_rows = {}, []
    for name, source in SOURCES.items():
        rows = []
        for case_id, repeat, actions in load(source):
            case = cases.get(case_id)
            if case is None:
                continue
            row = {'source': name, 'case_id': case_id, 'repeat': repeat, 'partition': case['partition'],
                   **fm.score_case(case, actions), **legacy(case, actions)}
            rows.append(row)
        report[name] = {'setting': source[2], **summarize(rows)}
        all_rows += rows
        s = report[name]
        print(f"{name:20s} n={s['rollouts']:3d}  sigma_L1={s['sigma_L1']['rate']:.3f}  in_order={s['intermediates_in_order']['rate']:.3f}"
              f"  strict={s['intermediates_in_order_strict']['rate']:.3f}  legacy_L1={s['legacy_L1']['rate']:.3f}", flush=True)
    # The 24-case subset makes the P2/P3 arms comparable with the agents.
    subset = {r['case_id'] for r in all_rows if r['source'] == 'p23_direct'}
    report['agent_skills23_r2@p23_subset'] = {'setting': 'agent_skills23_r2 restricted to the 24 P2/P3 cases',
                                              **summarize([r for r in all_rows if r['source'] == 'agent_skills23_r2' and r['case_id'] in subset])}
    (args.output / 'rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in all_rows))
    (args.output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
