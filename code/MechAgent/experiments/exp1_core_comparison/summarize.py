"""Post-hoc scoring. This is the only experiment runner that reads gold.jsonl."""
from __future__ import annotations

import argparse
import collections
import copy
import dataclasses
import importlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from develop import metrics
from develop import standalone_chemistry as chem
from develop.score_intermediates import compare_sequences, same_intermediate


def translate(action, inverse):
    action = copy.deepcopy(action)
    def remap(v):
        if isinstance(v, (list, tuple)):
            return type(v)(remap(x) for x in v)
        return inverse.get(v, v) if isinstance(v, str) else v
    for transfer in action.electron_transfers:
        transfer.source = remap(transfer.source)
        transfer.target = remap(transfer.target)
    return action


def heavy_signature(state, inverse=None):
    inverse = inverse or {}
    signature = chem.state_signature(state)
    atoms = {inverse.get(k, k): {'element': a['element'], 'charge': a['charge'], 'Hcount': 0}
             for k, a in signature['atoms'].items() if a['element'] != 'H'}
    bonds = {}
    for key, b in signature['bonds'].items():
        a, c = key.split('|')
        a, c = inverse.get(a, a), inverse.get(c, c)
        if a in atoms and c in atoms:
            bonds['|'.join(sorted((a, c)))] = b
    return {'atoms': atoms, 'bonds': bonds}


def deduplicate(states):
    selected = []
    for state in states:
        if not selected or not same_intermediate(selected[-1], state):
            selected.append(state)
    return selected


def summarize(out):
    inputs = {r['case_id']: r for r in map(json.loads, (out / 'inputs.jsonl').read_text().splitlines())}
    gold = {r['case_id']: r for r in map(json.loads, (out / 'gold.jsonl').read_text().splitlines())}
    sys.path.insert(0, str(out / 'snapshot'))
    bridge_module = importlib.import_module('exp1_agent.chemistry')
    bridge = bridge_module.ChemistryBridge(str(out / 'snapshot'))
    local = sys.modules['standalone_chemistry']
    arms = ['direct', 'agent_no_skills', 'agent_skills79']
    summaries, all_rows = {}, {}
    for arm in arms:
        results, comparisons, rows = [], [], []
        for path in sorted((out / arm).glob('exp1_*.json')):
            try:
                r = json.loads(path.read_text())
            except FileNotFoundError:
                # Resume archives unfinished infrastructure attempts. A path
                # discovered by glob may disappear before this reader opens it.
                continue
            if not r.get('finished'):
                continue
            cid, repeat = r['case_id'], r['repeat_index']
            g = gold[cid]
            inv = {v: k for k, v in g['label_map'].items()}
            actions = [chem.parse_pgl_literal(text) for text in r['accepted_actions_pgl']]
            translated = [translate(a, inv) for a in actions]
            expected = [chem.parse_pgl_literal(text) for text in g['gt_actions_pgl']]
            cmp = metrics.compare_path(translated, expected, reaction_id=cid, repeat_index=repeat)
            comparisons.append(cmp)
            # Diagnostic: compare heavy-atom connectivity/charge, not literal H labels.
            state = local.parse_pgl_literal(inputs[cid]['initial_state_pgl'])
            predicted_states = []
            for text in r['accepted_actions_pgl']:
                action, parsed = bridge.parse_action(text)
                preview = bridge.apply_local_reaction_transformation(state, action)
                if not preview.valid:
                    break
                state = preview.state
                predicted_states.append(heavy_signature(state, inv))
            gt_states = [heavy_signature(chem.parse_pgl_literal(text)) for text in g['gt_states_pgl']]
            # Collapse hydrogen-only duplicate graphs; do not call this full molecular matching.
            gold_heavy = deduplicate(gt_states)[1:]
            predicted_heavy = deduplicate(predicted_states)
            intermediate = compare_sequences(predicted_heavy, gold_heavy) if gold_heavy else {}
            final = heavy_signature(chem.parse_pgl_literal(r['final_state_pgl']), inv)
            final_ok = same_intermediate(final, gt_states[-1])
            errors = r.get('errors', [])
            err_text = json.dumps(errors)
            infra = r.get('error') is not None or any(word in err_text for word in
                ['APIConnectionError', 'APITimeoutError', 'RateLimitError', 'InternalServerError', 'PermissionDeniedError', 'AuthenticationError'])
            row = {**cmp.to_dict(), 'case_id': cid, 'arm': arm, 'repeat_index': repeat,
                   'reaction_class': g['reaction_class'], 'heavy_final_match': final_ok,
                   'heavy_intermediate': intermediate, 'infrastructure_error': infra,
                   'generation_tokens': r['generation_tokens'], 'total_tokens': r['total_tokens'],
                   'prompt_tokens': r['prompt_tokens'], 'reasoning_tokens': r['reasoning_tokens'],
                   'calls': r['calls'], 'elapsed_seconds': r['elapsed_seconds'], 'stop_reason': r['stop_reason'],
                   'selected_skills': (r.get('planner_plan', {}).get('planner_skill_selection') or {}).get('selected_skills', [])}
            rows.append(row)
            results.append(r)
        all_rows[arm] = rows
        if not comparisons:
            summaries[arm] = {'num_paths': 0}
            continue
        summary = metrics.aggregate(comparisons, seed=20261018)
        summary['infrastructure_errors'] = sum(x['infrastructure_error'] for x in rows)
        summary['heavy_final_match_rate'] = statistics.mean(x['heavy_final_match'] for x in rows)
        summary['mean_generation_tokens'] = statistics.mean(x['generation_tokens'] for x in rows)
        summary['mean_total_tokens'] = statistics.mean(x['total_tokens'] for x in rows)
        summary['mean_calls'] = statistics.mean(x['calls'] for x in rows)
        summary['mean_elapsed_seconds'] = statistics.mean(x['elapsed_seconds'] for x in rows)
        summary['total_tokens'] = sum(x['total_tokens'] for x in rows)
        summary['skill_injected_paths'] = sum(bool(x['selected_skills']) for x in rows)
        summary['generation_budget_exceeded_paths'] = sum(r['generation_tokens'] > r['generation_budget'] for r in results)
        summary['heavy_intermediate_ordered_all_rate'] = statistics.mean(bool(x['heavy_intermediate'].get('ordered_all')) for x in rows)
        matched = sum(x['heavy_intermediate'].get('matched_in_order', 0) for x in rows)
        expected_states = sum(x['heavy_intermediate'].get('gold_intermediates', 0) for x in rows)
        summary['heavy_intermediate_ordered_recall'] = matched / expected_states if expected_states else 0
        summary['stop_reasons'] = dict(collections.Counter(x['stop_reason'] for x in rows))
        summary['returned_models'] = sorted({m for r in results for m in r.get('returned_models', [])})
        summaries[arm] = summary
    paired = {}
    for before, after in [('direct', 'agent_no_skills'), ('agent_no_skills', 'agent_skills79'), ('direct', 'agent_skills79')]:
        b = {(r['case_id'], r['repeat_index']): r for r in all_rows[before]}
        a = {(r['case_id'], r['repeat_index']): r for r in all_rows[after]}
        keys = sorted(set(a) & set(b))
        block = {'paired_rollouts': len(keys), 'paired_reactions': len({k[0] for k in keys}), 'deltas': {}}
        for field in ['transfer_set_match', 'transfer_sequence_match', 'step_partition_match', 'heavy_final_match']:
            differences = [int(a[k][field]) - int(b[k][field]) for k in keys]
            block['deltas'][field] = {'mean': statistics.mean(differences) if differences else None,
                                     'ci95': metrics.clustered_bootstrap_ci(differences, [k[0] for k in keys], seed=20261018),
                                     'ci95_by_reaction_class': metrics.clustered_bootstrap_ci(differences, [a[k]['reaction_class'] for k in keys], seed=20261018),
                                     'improved': sum(d > 0 for d in differences), 'regressed': sum(d < 0 for d in differences)}
        paired[before + '_to_' + after] = block
    service_status = json.loads((out / 'STATUS.json').read_text()) if (out / 'STATUS.json').exists() else {}
    report = {'expected_per_arm': 160, 'complete': all(len(all_rows[a]) == 160 for a in arms),
              'service_status': service_status,
              'arms': summaries, 'paired': paired,
              'metric_notes': ['Primary L1/L2/L3 score the executable accepted path using the existing normalized arrow metrics.',
                               'Heavy graph diagnostics omit hydrogen ownership and are not full chemical correctness.',
                               'All validly completed rollouts remain in the denominator, including model/protocol failures. Infrastructure interruptions stay pending on the same cases.',
                               'The gateway can exceed max_tokens with hidden reasoning; 98304 is an observed-usage stopping threshold, not a guaranteed hard provider ceiling. Overshoots are reported.',
                               'No LLM judge was used; expert validation is a separate unperformed experiment.']}
    (out / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (out / 'scored_rows.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for rows in all_rows.values() for row in rows))
    state_text = '480 条轨迹全部完成。' if report['complete'] else (
        '接口阻塞，运行队列已停止；有效完成 0/480，不能形成实验结论。' if service_status.get('status') == 'blocked_by_api'
        else '尚未完成，以下为阶段结果，不能据此提前下结论。')
    lines = ['# 实验 1：同模型三组对照', '', '**状态：' + state_text + '**', '',
             '| 方法 | 完成/160 | L1 | L2 | L3 | 重原子终态匹配 | 平均生成 token | 平均总 token | 平均调用 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm, s in summaries.items():
        if not s['num_paths']:
            lines.append(f'| {arm} | 0 | — | — | — | — | — | — | — |')
            continue
        values = [f"{s['levels'][key]['rate']:.1%}" for key in ['L1_transfer_set', 'L2_transfer_sequence', 'L3_step_partition']]
        lines.append(f"| {arm} | {s['num_paths']} | " + ' | '.join(values) + f" | {s['heavy_final_match_rate']:.1%} | {s['mean_generation_tokens']:.0f} | {s['mean_total_tokens']:.0f} | {s['mean_calls']:.1f} |")
    lines += ['', '## 配对差值（按反应重采样的 95% 区间）', '']
    for name, block in paired.items():
        lines.append(f"- {name}，{block['paired_reactions']} 个反应 / {block['paired_rollouts']} 对轨迹：")
        for key, val in block['deltas'].items():
            if val['mean'] is not None:
                lo, hi = val['ci95']
                lines.append(f"  - {key}: {val['mean'] * 100:+.1f} 个百分点，95% CI [{lo * 100:+.1f}, {hi * 100:+.1f}]。")
    lines += ['', '## 解释边界', '',
              '- 样本为已知反应家族的新记录，不能据此宣称未见类别泛化或排除预训练污染。',
              '- 原始来源是模板标注的 MechFinder；L1/L2/L3 衡量与标注箭头的一致性，不等同于专家认定的唯一正确机理。',
              '- 三组生成 token 上限相同，但实际 token 和调用量不同；不能省略成本直接声称等计算量优越。',
              '- 网关可能在一次请求中产生超过 max_tokens 的思考 token，因此 98304 是按已返回用量停止后续调用的阈值，并非供应商可保证的硬上限；超额轨迹数另行统计。',
              '- 重原子终态指标忽略氢归属，不能替代完整状态匹配。',
              '- 网关请求模型为 gemini-3.8-flash-high，并显式传入 reasoning_effort=high；实际响应模型名另存 summary.json。',
              '- 完整协议见 PROTOCOL.md，数据和代码哈希见 manifest.json。', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'complete': report['complete'], 'counts': {a: len(all_rows[a]) for a in arms},
                      'summary': str(out / 'summary.json')}, ensure_ascii=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    summarize(ap.parse_args().output.resolve())
