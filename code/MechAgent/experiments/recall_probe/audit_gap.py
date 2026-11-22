"""Gap audit: how many of the model's 'wrong' next steps are bookkeeping/convention variants of the GT path?

Audited answers (115 GT-prefix steps, gemini-3.8-flash-high): the select-own pick and the single sample (r=0)
whenever they are not the GT next state. Each distinct (step, answer state) is classified on the sigma graph
(heavy-heavy bonds + H count per heavy atom), which is what fukuyama_metrics.same_state compares:
  merged     equals a later GT state (several GT steps in one)
  partial    its sigma events are a strict subset of the GT next step's (stepwise where GT is concerted)
  subpath    its sigma events are a subset of the net events from the current state to a later GT state
             (a later GT event done first, or part of a merged step)
  proton     same heavy-atom graph as the GT next state or a later one, only H counts differ (protonation
             state / site bookkeeping)
  on_atoms   only touches heavy atoms the remaining GT path touches, but makes/breaks other bonds
  off_path   touches heavy atoms the remaining GT path never touches
The first four are candidate convention variants; on_atoms/off_path are candidate chemical errors. The
classification is a screen, not a verdict: REVIEW.md lists every audited answer for a chemist.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recall as R
import select_own as S
from recall import C, P, chem, fm

OUT = R.OUT / 'gap_audit'
CONVENTION = ('merged', 'partial', 'subpath', 'proton')


def sigma(state):
    v = fm.state_view(state)
    bonds = {frozenset((a, b)) for a in v['adjacency'] for b in v['adjacency'][a]}
    return bonds, v['hcount']


def events(before, after):
    (b0, h0), (b1, h1) = sigma(before), sigma(after)
    dh = {n: h1.get(n, 0) - h0.get(n, 0) for n in set(h0) | set(h1) if h1.get(n, 0) != h0.get(n, 0)}
    return b1 - b0, b0 - b1, dh


def subset(e, f, strict=False):
    (fa, ba, ha), (fb, bb, hb) = e, f
    ok = fa <= fb and ba <= bb and all(n in hb and (d > 0) == (hb[n] > 0) and abs(d) <= abs(hb[n]) for n, d in ha.items())
    return ok and (not strict or e != f)


def atoms_of(e):
    fa, ba, ha = e
    return {x for p in fa | ba for x in p} | set(ha)


def molecules(state):
    return collections.Counter(C._molecule_smiles(state, c) for c in C._components(state))


def classify(cur, gt, pos, m):
    if not molecules(gt[pos + 1]) - molecules(m):
        return 'same_molecules'  # metric false negative: same species, different atom mapping (symmetry)
    if R.classify(m, gt, pos) == 'later':
        return 'merged'
    em = events(cur, m)
    if not any(em):
        return 'no_change'
    if subset(em, events(cur, gt[pos + 1]), strict=True):
        return 'partial'
    if any(subset(em, events(cur, g)) for g in gt[pos + 2:]):
        return 'subpath'
    bm = sigma(m)[0]
    if any(bm == sigma(g)[0] for g in gt[pos + 1:]):
        return 'proton'
    path_atoms = set().union(*(atoms_of(events(a, b)) for a, b in zip(gt[pos:-1], gt[pos + 1:])))
    return 'on_atoms' if atoms_of(em) <= path_atoms else 'off_path'


def main():
    cases = R.pool_cases()
    items = [it for it in json.loads(R.ITEMS.read_text()) if len(it['candidates']) == P.N_CAND]
    audited, rows = {}, []
    for it in items:
        case, pos = cases[it['case_id']], it['step']
        cur, gt = chem.parse_pgl_literal(it['current']), R.gt_chain(case)
        b = S.build(it, case)
        own = json.loads((S.OUT / f"{it['case_id']}_{pos}.json").read_text())
        answers = []
        if own['choice'] is not None and own['choice'] not in b['gt_indices']:
            answers.append(('select_own', chem.parse_pgl_literal(b['candidates'][own['choice']])))
        s0 = json.loads((R.OUT / R.MODEL / 'sample' / f"{it['case_id']}_{pos}_0.json").read_text())['proposals']
        if s0 and s0[0]['class'] != 'next':
            m = R.apply_changes(cur, s0[0]['text'])
            answers.append(('one_sample', m))
        for source, m in answers:
            key = (it['case_id'], pos, None if m is None else repr(sorted(map(str, sigma(m)[0]))) + repr(sorted(sigma(m)[1].items())))
            if key in audited:
                audited[key]['sources'].append(source)
                continue
            cat = 'invalid' if m is None else classify(cur, gt, pos, m)
            rec = {'case_id': it['case_id'], 'step': pos, 'n_gt_steps': len(gt) - 1, 'sources': [source], 'category': cat,
                   'gt_in_pool': bool(b['gt_indices']),
                   'model_step': None if m is None else C.step_text(cur, m)[0],
                   'model_species': None if m is None else C.step_text(cur, m)[1],
                   'gt_step': C.step_text(cur, gt[pos + 1])[0], 'gt_species': C.step_text(cur, gt[pos + 1])[1],
                   'gt_rest': [C.step_text(a, b2)[0] for a, b2 in zip(gt[pos + 1:-1], gt[pos + 2:])],
                   'history': it['history'], 'conditions': R.A.conditions_text(case),
                   'current_species': ', '.join(C._molecule_smiles(cur, c) for c in C._components(cur))}
            audited[key] = rec
            rows.append(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for src in ('select_own', 'one_sample', 'any'):
        sel = [r for r in rows if src == 'any' or src in r['sources']]
        cnt = collections.Counter(r['category'] for r in sel)
        summary[src] = {'answers': len(sel), 'cases': len({r['case_id'] for r in sel}), 'categories': dict(cnt.most_common()),
                        'convention_candidates': sum(cnt[c] for c in CONVENTION)}
    verdicts = OUT / 'verdicts_claude.json'
    if verdicts.exists():
        v = json.loads(verdicts.read_text())
        for r in rows:
            r['verdict'], r['verdict_type'], r['verdict_note'] = v[f"{r['case_id']}:{r['step']}:{'+'.join(r['sources'])}"]
        n = len(items)
        for src, base in (('select_own', 0.6783), ('one_sample', 0.7304)):
            sel = [r for r in rows if src in r['sources']]
            cnt = collections.Counter(r['verdict'] for r in sel)
            summary[src]['verdicts'] = dict(cnt)
            summary[src]['types'] = dict(collections.Counter(r['verdict_type'] for r in sel).most_common())
            summary[src]['accuracy_strict'] = base
            summary[src]['accuracy_acceptable'] = round(1 - (cnt['wrong'] + cnt['questionable']) / n, 4)
            summary[src]['accuracy_acceptable_or_questionable'] = round(1 - cnt['wrong'] / n, 4)
            summary[src]['accuracy_acceptable_excluding_merged'] = round(
                1 - sum(r['verdict'] != 'acceptable' or r['category'] == 'merged' for r in sel) / n, 4)
        render(rows)
    (OUT / 'audit.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (OUT / 'SUMMARY.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


def render(rows):
    lines = ['# 差距审计：模型"答错"的下一步逐条清单（供化学家复核）', '',
             '每条给出当前物种、GT 下一步、模型答案。自动类别基于 σ 图；初判由 Claude 给出（非化学家），'
             'acceptable = 约定/粒度差异或文献可辩护的替代，questionable = 无证据难以区分，wrong = 化学上不如 GT。', '']
    order = {'wrong': 0, 'questionable': 1, 'acceptable': 2}
    for r in sorted(rows, key=lambda r: (order[r['verdict']], r['case_id'], r['step'])):
        lines += [f"## {r['case_id']} 第 {r['step'] + 1}/{r['n_gt_steps']} 步 — 初判 **{r['verdict']}**（{r['verdict_type']}；自动类别 {r['category']}；来源 {', '.join(r['sources'])}）",
                  f"- 条件：{r['conditions']}", f"- 当前物种：`{r['current_species']}`",
                  f"- GT 下一步：`{r['gt_step']}` → `{r['gt_species']}`",
                  f"- 模型答案：`{r['model_step']}` → `{r['model_species']}`",
                  f"- GT 后续：{' ‖ '.join(f'`{s}`' for s in r['gt_rest'][:2]) or '（无）'}",
                  f"- 初判理由：{r['verdict_note']}", '- 化学家意见：', '']
    (OUT / 'REVIEW.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
