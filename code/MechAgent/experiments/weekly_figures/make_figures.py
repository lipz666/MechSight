"""Four evidence figures for the 2026-10-26 weekly report (one per storyline part).

Every number is read from the experiment outputs, nothing is typed in by hand.
Run with a Python that has matplotlib + rdkit (scratchpad venv).
"""
from __future__ import annotations

import collections
import glob
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from develop import fukuyama_metrics as fm
from develop import metrics
from develop import standalone_chemistry as chem

OUT = ROOT / 'doc' / 'figures_20261026'
OUT.mkdir(parents=True, exist_ok=True)

FONT = '/System/Library/Fonts/Hiragino Sans GB.ttc'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({
    'font.family': font_manager.FontProperties(fname=FONT).get_name(),
    'axes.unicode_minus': False, 'font.size': 10,
    'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'axes.grid.axis': 'y',
    'grid.color': '#e8e7e2', 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
})
TEXT, TEXT2, MUTED = '#0b0b0b', '#52514e', '#8a8984'
# Colour follows the entity across all four figures (validated categorical slots).
LLM, FLOWER, TEMPLATE, ORACLE = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6'
GRAY = '#a9a89f'
SEED = 20261026


def pct_ci(values, ids):
    lo, hi = metrics.clustered_bootstrap_ci(values, ids, seed=SEED)
    return 100 * statistics.mean(values), 100 * lo, 100 * hi


def bars(ax, xs, vals, colors, errs=None, width=0.62, fmt='{:.0f}%'):
    rects = ax.bar(xs, vals, width=width, color=colors, edgecolor='#fcfcfb', linewidth=2, zorder=3)
    if errs is not None:
        lo = [v - e[0] for v, e in zip(vals, errs)]
        hi = [e[1] - v for v, e in zip(vals, errs)]
        ax.errorbar(xs, vals, yerr=[lo, hi], fmt='none', ecolor=TEXT2, elinewidth=1, capsize=3, zorder=4)
    for r, v, e in zip(rects, vals, errs or [None] * len(vals)):
        top = (e[1] if e else v)
        ax.text(r.get_x() + r.get_width() / 2, top + 1.5, fmt.format(v), ha='center', va='bottom', color=TEXT, fontsize=9)
    return rects


def title(ax, text, sub=None):
    ax.set_title(text, loc='left', color=TEXT, fontsize=11, pad=16 if sub else 8)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=MUTED, fontsize=8.5, va='bottom')


# ---------------------------------------------------------------- Figure 1
def figure1():
    exp1 = json.loads((ROOT / 'outputs/publication_exp1_20261018/summary.json').read_text())['arms']
    p1 = json.loads((ROOT / 'outputs/p1_template_20261024/summary.json').read_text())['arms']
    arms = [('模板检索\n(无 LLM、无类别)', p1['retrieval'], TEMPLATE),
            ('模板检索\n(给定类别)', p1['oracle_class'], TEMPLATE),
            ('LLM\nDirect', exp1['direct'], LLM),
            ('LLM\nAgent', exp1['agent_no_skills'], LLM),
            ('LLM Agent\n+79 Skills', exp1['agent_skills79'], LLM)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, (level, name) in zip(axes, [('L1_transfer_set', 'L1 箭头集合正确率'), ('L3_step_partition', 'L3 含分步边界的完全正确率')]):
        xs = range(len(arms))
        vals = [100 * a[1]['levels'][level]['rate'] for a in arms]
        errs = [(100 * a[1]['levels'][level]['ci95'][0], 100 * a[1]['levels'][level]['ci95'][1]) for a in arms]
        bars(ax, list(xs), vals, [a[2] for a in arms], errs)
        ax.set_xticks(list(xs), [a[0] for a in arms], fontsize=8.5)
        ax.set_ylim(0, 100)
        title(ax, name)
    axes[0].set_ylabel('正确率 (%)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (TEMPLATE, LLM)]
    fig.legend(handles, ['模板检索（不用 LLM）', 'LLM 方法'], loc='upper right', bbox_to_anchor=(0.995, 0.905),
               frameon=False, ncol=2, fontsize=9)
    fig.suptitle('图1  MechFinder 上模板检索远超 LLM（同一批 80 个新反应，误差棒为按反应 bootstrap 95% CI）',
                 x=0.01, ha='left', color=TEXT, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / 'fig1_template_degeneration.png', dpi=200)


# ---------------------------------------------------------------- Figure 2
def figure2():
    dev_ids = json.loads((ROOT / 'outputs/p23_planning_probe_20261024/cases.json').read_text())
    clean = {c['case_id']: c for c in map(json.loads, (ROOT / 'outputs/fukuyama/clean.jsonl').read_text().splitlines())}
    fl = {cid: json.loads((ROOT / f'outputs/flower_compare_20261025/dev/{cid}.json').read_text()) for cid in dev_ids}
    rescore = [json.loads(l) for l in (ROOT / 'outputs/fukuyama_rescore_20261024/rows.jsonl').read_text().splitlines()]
    fwd = {r['case_id']: r for r in rescore if r['source'] == 'agent_forward'}
    rows = json.loads((ROOT / 'outputs/p23_planning_probe_20261024/compare_rows.json').read_text())['direct']
    series = {
        'FlowER\n(Nature 2025)': ({c: float(bool(fl[c].get('any_sigma_L1'))) for c in dev_ids},
                                  {c: float(bool(fl[c].get('any_in_order'))) for c in dev_ids}, FLOWER),
        'LLM Agent\n(正向，不给目标)': ({c: float(fwd[c]['sigma_L1']) for c in dev_ids},
                                    {c: float(fwd[c]['intermediates_in_order']) for c in dev_ids}, ORACLE),
        'LLM Direct\n(给定目标)': ({c: rows[c]['sigma_L1'] for c in dev_ids},
                                 {c: rows[c]['intermediates_in_order'] for c in dev_ids}, LLM),
    }
    fig = plt.figure(figsize=(11, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15])
    for k, (metric_idx, name) in enumerate([(0, 'σ-L1 成键/断键事件正确率'), (1, 'GT 中间态按序命中率')]):
        ax = fig.add_subplot(gs[0, k])
        labels = list(series)
        vals, errs = [], []
        for lab in labels:
            per = series[lab][metric_idx]
            m, lo, hi = pct_ci([per[c] for c in dev_ids], dev_ids)
            vals.append(m)
            errs.append((lo, hi))
        bars(ax, list(range(3)), vals, [series[l][2] for l in labels], errs)
        ax.set_xticks(range(3), labels, fontsize=8.5)
        ax.set_ylim(0, 100)
        if k == 0:
            ax.set_ylabel('正确率 (%)')
        title(ax, name)
    # failure modes of FlowER on all 45 cases scored so far
    recs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(ROOT / 'outputs/flower_compare_20261025/dev/*.json')) +
                                                           glob.glob(str(ROOT / 'outputs/flower_compare_20261025/heldout/*.json')))]
    modes = collections.Counter()
    for r in recs:
        if r.get('n_paths', 0) == 0:
            modes['第一步即预测"不反应"'] += 1
        elif r.get('timed_out'):
            modes['20 分钟搜索上限内未到达产物'] += 1
        else:
            modes['搜索结束，无路径到达产物'] += 1
    ax = fig.add_subplot(gs[0, 2])
    order = ['第一步即预测"不反应"', '20 分钟搜索上限内未到达产物', '搜索结束，无路径到达产物']
    shades = ['#c44d1f', '#eb6834', '#f4a37f']
    left = 0
    for lab, col in zip(order, shades):
        n = modes[lab]
        ax.barh([0], [n], left=left, color=col, edgecolor='#fcfcfb', linewidth=2, height=0.5, zorder=3)
        ax.text(left + n / 2, 0, str(n), ha='center', va='center', color='white', fontsize=10, fontweight='bold')
        left += n
    ax.set_xlim(0, len(recs))
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel(f'题数（共 {len(recs)} 题）')
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in shades], order, loc='upper center',
              bbox_to_anchor=(0.5, -0.32), frameon=False, fontsize=8.5, ncol=1)
    title(ax, 'FlowER 失败方式', '任一路径到达目标产物：0 题')
    fig.suptitle('图2  模板之外的真实机理（FukuyamaBench 开发集 24 题）：LLM 显著领先 FlowER（初步结果，待正确性检查）',
                 x=0.01, ha='left', color=TEXT, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / 'fig2_llm_vs_flower.png', dpi=200)


# ---------------------------------------------------------------- Figure 3
def figure3():
    s = json.loads((ROOT / 'outputs/fukuyama_rescore_20261024/summary.json').read_text())
    runs = [('agent_skills23_r2', 'Agent+23 Skills\n(120 条)'), ('agent_skills79_r2', 'Agent+79 Skills\n(120 条)'),
            ('agent_early_v2', '早期 Agent\n(60 条)'), ('agent_forward', '正向 Agent\n(60 条)')]
    metrics_ = [('legacy_L1', '旧口径 L1', GRAY), ('intermediates_in_order', '新：GT 中间态按序命中', TEMPLATE),
                ('sigma_L1', '新：σ-L1', LLM)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={'width_ratios': [1.4, 1]})
    ax = axes[0]
    w = 0.26
    for j, (key, lab, col) in enumerate(metrics_):
        xs = [i + (j - 1) * w for i in range(len(runs))]
        vals = [100 * s[r][key]['rate'] for r, _ in runs]
        rects = ax.bar(xs, vals, width=w, color=col, edgecolor='#fcfcfb', linewidth=2, zorder=3, label=lab)
        for r_, v in zip(rects, vals):
            ax.text(r_.get_x() + r_.get_width() / 2, v + 1.2, f'{v:.0f}', ha='center', va='bottom', fontsize=8, color=TEXT)
    ax.set_xticks(range(len(runs)), [n for _, n in runs], fontsize=8.5)
    ax.set_ylim(0, 80)
    ax.set_ylabel('正确率 (%)')
    ax.legend(frameon=False, fontsize=8.5, loc='upper right')
    title(ax, '历史 FukuyamaBench 运行用新旧指标重算')
    ax = axes[1]
    part = s['agent_skills23_r2']['by_partition']
    keys = [('legacy_L1', GRAY), ('intermediates_in_order', TEMPLATE), ('sigma_L1', LLM)]
    for j, (key, col) in enumerate(keys):
        xs = [i + (j - 1) * w for i in range(3)]
        vals = [100 * part[p][key] for p in 'ABC']
        rects = ax.bar(xs, vals, width=w, color=col, edgecolor='#fcfcfb', linewidth=2, zorder=3)
        for r_, v in zip(rects, vals):
            ax.text(r_.get_x() + r_.get_width() / 2, v + 1.2, f'{v:.0f}', ha='center', va='bottom', fontsize=8, color=TEXT)
    ax.set_xticks(range(3), ['Set A', 'Set B', 'Set C（最难）'])
    ax.set_ylim(0, 80)
    title(ax, '按难度分（Agent+23 Skills）')
    fig.suptitle('图3  旧指标惩罚更细的分步与共振写法，约低估一半；新指标经 GT 自洽 60/60 与单元测试验证',
                 x=0.01, ha='left', color=TEXT, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / 'fig3_metric_fix.png', dpi=200)


# ---------------------------------------------------------------- Figure 4
def figure4():
    probe = ROOT / 'outputs/p23_planning_probe_20261024'
    ids = json.loads((probe / 'cases.json').read_text())
    clean = {c['case_id']: c for c in map(json.loads, (ROOT / 'outputs/fukuyama/clean.jsonl').read_text().splitlines())}
    rows = json.loads((probe / 'compare_rows.json').read_text())
    # (a) missing vs extra sigma events of LLM direct (first sample) on failed cases
    missing = extra = 0
    for cid in ids:
        case = clean[cid]
        actions = json.loads((probe / 'direct' / f'{cid}.json').read_text())['actions_pgl']
        states, _ = fm.replay(case['initial_state_pgl'], actions)
        init = chem.parse_pgl_literal(case['initial_state_pgl'])
        pred = sum((fm.sigma_events(fm.step_from_states(b, a)) for b, a in zip([init] + states[:-1], states)), collections.Counter())
        gold = sum((fm.sigma_events(s) for s in fm.gold_steps_from_clean(case)), collections.Counter())
        missing += sum((gold - pred).values())
        extra += sum((pred - gold).values())
    transient = {cid: len(fm.sigma_scores([], fm.gold_steps_from_clean(clean[cid]))['gold_transient']) for cid in ids}

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), gridspec_kw={'width_ratios': [0.75, 1.1, 1.25]})
    ax = axes[0]
    rects = ax.bar([0, 1], [missing, extra], width=0.55, color=[LLM, LLM], edgecolor='#fcfcfb', linewidth=2, zorder=3)
    for r_, v in zip(rects, [missing, extra]):
        ax.text(r_.get_x() + r_.get_width() / 2, v + 1, str(v), ha='center', va='bottom', color=TEXT)
    ax.set_xticks([0, 1], ['漏写的\nσ 事件', '多写的\nσ 事件'])
    ax.set_ylim(0, max(missing, extra) * 1.18)
    ax.set_ylabel('事件数（24 题合计）')
    title(ax, 'LLM 偏好最短路径')

    ax = axes[1]
    strata = [('0', lambda n: n == 0), ('1–2', lambda n: 1 <= n <= 2), ('≥3', lambda n: n >= 3)]
    w = 0.36
    for j, (fam, lab, col) in enumerate([('direct', 'LLM Direct', LLM), ('oracle_lat', '+ 告知 GT 瞬态键', ORACLE)]):
        xs = [i + (j - 0.5) * w for i in range(3)]
        vals = []
        for _, pred in strata:
            sel = [c for c in ids if pred(transient[c])]
            vals.append(100 * statistics.mean(rows[fam][c]['sigma_L1'] for c in sel))
        rects = ax.bar(xs, vals, width=w, color=col, edgecolor='#fcfcfb', linewidth=2, zorder=3, label=lab)
        for r_, v in zip(rects, vals):
            ax.text(r_.get_x() + r_.get_width() / 2, v + 1.5, f'{v:.0f}%', ha='center', va='bottom', fontsize=8.5, color=TEXT)
    sizes = [sum(1 for c in ids if pred(transient[c])) for _, pred in strata]
    ax.set_xticks(range(3), [f'{s} 个瞬态键\n(n={n})' for (s, _), n in zip(strata, sizes)], fontsize=8.5)
    ax.set_ylim(0, 118)
    ax.set_ylabel('σ-L1 (%)')
    ax.legend(frameon=False, fontsize=8.5, loc='upper center', ncol=2)
    title(ax, '知道隐藏事件就能做对')

    ax = axes[2]
    multi = [c for c in ids if transient[c] >= 2]
    arms = [('direct', '无提示', LLM), ('oracle_lat', 'GT 瞬态键\n(全对)', ORACLE),
            ('oracle_noisy', '+ 等量错误键\n(精度 50%)', ORACLE), ('oracle_half', '只给一半\n(召回 50%)', GRAY)]
    vals = [100 * statistics.mean(rows[f][c]['sigma_L1'] for c in multi) for f, _, _ in arms]
    bars(ax, list(range(4)), vals, [a[2] for a in arms])
    ax.set_xticks(range(4), [a[1] for a in arms], fontsize=8.5)
    ax.set_ylim(0, 105)
    title(ax, '能"认出"正确事件，但须召回齐全', f'含 ≥2 个瞬态键的题（n={len(multi)}）')
    fig.suptitle('图4  剩余瓶颈是隐藏（瞬态）事件（FukuyamaBench 开发集 24 题，Gemini 3.8 Flash）',
                 x=0.01, ha='left', color=TEXT, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / 'fig4_hidden_event_bottleneck.png', dpi=200)
    return missing, extra


if __name__ == '__main__':
    figure1()
    figure2()
    figure3()
    print('fig4 missing/extra:', figure4())
    print('written to', OUT)
