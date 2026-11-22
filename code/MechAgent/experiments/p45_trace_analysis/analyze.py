"""P4/P5 pre-analysis on the frozen exp1 trajectories; no new model calls.

P4: does the 79-skill arm lose because the wrong skill is retrieved, or
    because even a matching skill hurts?  A skill "matches" when it is the
    one trigger_expectations.json / the name table below assigns to the GT class.
P5: is the Planner review an error detector?  Every reviewed Executor
    proposal is labelled GT-consistent when all its transfers occur in the GT
    transfer multiset (H anonymized, labels un-blinded), then the review's
    APPROVE/REVISE decision is scored against that label.
"""
from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'exp1_core_comparison'))
from develop import metrics
from develop import standalone_chemistry as chem
from summarize import translate

EXP1 = ROOT / 'outputs' / 'publication_exp1_20261018'

# GT MechFinder class -> skills written for it (a class may have two).
CLASS_SKILLS = {
    '(hemi)acetal(aminal)_hydrolysis': {'plan-acetal-hydrolysis'},
    'Appel': {'plan-appel-halogenation'},
    'Boc_deprotection': {'plan-boc-deprotection'},
    'Cbz_deprotection': {'plan-cbz-acidic-cleavage'},
    'DCC_condensation': {'plan-dcc-amide-condensation'},
    'Fmoc_deprotection': {'plan-fmoc-deprotection'},
    'Friedel_Crafts_acylation': {'plan-friedel-crafts-acylation'},
    'Horner_Wadsworth_Emmons': {'plan-horner-wadsworth-emmons'},
    'Ing_Manske': {'plan-ing-manske-hydrazinolysis'},
    'Jones_oxidation': {'plan-jones-oxidation'},
    'Knorr_pyrazole_synthesis': {'plan-knorr-pyrazole-synthesis'},
    'Mannich': {'plan-mannich-reaction'},
    'Markovnikov_addition': {'plan-acidic-markovnikov-addition'},
    'Michael_addition': {'plan-michael-conjugate-addition'},
    'SN1': {'plan-sn1-displacement', 'plan-tosylate-assisted-sn1'},
    'SN2_alcohol(thiol)': {'plan-sn2-displacement', 'plan-sn2-tosylate'},
    'SNAr_alco(thi)ol(ortho)': {'plan-sn-ar-alkoxide-thiolate'},
    'Staudinger': {'plan-staudinger-azide-reduction'},
    'Weinreb_ketone_synthesis': {'plan-weinreb-ketone-substitution'},
    'Wittig': {'plan-wittig-olefination'},
    'Wittig_ver_2': {'plan-wittig-olefination'},
    'Wolf_Kishner_reduction': {'plan-wolff-kishner-reduction'},
    'acetal_formation': {'plan-acetal-formation'},
    'acetal_formation_from_enol_ether': {'plan-acidic-enol-ether-addition', 'plan-acetal-formation'},
    'alcohol_attack_to_carbonyl_or_sulfonyl': {'plan-carbonyl-addition-elimination', 'plan-methyl-ester-synthesis'},
    'aldol_addition': {'plan-aldol-addition'},
    'aldol_condensation': {'plan-aldol-condensation'},
    'alkynyl_attack_to_carbonyl': {'plan-alkynyl-carbonyl-addition'},
    'amine_oxidation': {'plan-amine-oxidation'},
    'base_cat_ester_hydrolysis': {'plan-ester-saponification-hydrolysis'},
    'carboxylic_acid_derivative_hydrolysis_or_formation': {'plan-carboxylic-derivative-hydrolysis', 'plan-carbonyl-addition-elimination'},
    'double_SN2': {'plan-double-sn2-cyclization'},
    'ester_reduction': {'plan-ester-reduction', 'plan-hydride-carbonyl-reduction'},
    'imine_formation': {'plan-imine-formation'},
    'imine_reduction': {'plan-imine-reduction'},
    'intramolecular_lactonization': {'plan-intramolecular-lactonization'},
    'lactone_reduction': {'plan-stepwise-lactone-reduction'},
    'nucleophilic_attack_to_(thio)carbonyl_or_sulfonyl': {'plan-carbonyl-addition-elimination'},
    'nucleophilic_attack_to_iso(thio)cyanate': {'plan-isocyanate-nucleophile-addition'},
    'reductive_amination': {'plan-reductive-amination'},
}


def rate(xs):
    xs = list(xs)
    return (round(statistics.mean(xs), 3), len(xs)) if xs else (None, 0)


def main():
    gold = {r['case_id']: r for r in map(json.loads, (EXP1 / 'gold.jsonl').read_text().splitlines())}
    scored = {(r['arm'], r['case_id'], r['repeat_index']): r
              for r in map(json.loads, (EXP1 / 'scored_rows.jsonl').read_text().splitlines())}
    report = {}

    # ---- P4: skill retrieval quality and its effect -------------------------
    groups = collections.defaultdict(list)
    for (arm, cid, rep), row in scored.items():
        if arm != 'agent_skills79':
            continue
        base = scored[('agent_no_skills', cid, rep)]
        chosen = set(row['selected_skills'] and [s['name'] if isinstance(s, dict) else s for s in row['selected_skills']] or [])
        expected = CLASS_SKILLS[row['reaction_class']]
        kind = 'none' if not chosen else ('matching' if chosen & expected else 'non_matching')
        groups[kind].append((row, base))
    p4 = {}
    for kind, pairs in groups.items():
        p4[kind] = {
            'rollouts': len(pairs),
            'L1_skills': rate(r['transfer_set_match'] for r, _ in pairs),
            'L1_no_skills_same_rollouts': rate(b['transfer_set_match'] for _, b in pairs),
            'heavy_final_skills': rate(r['heavy_final_match'] for r, _ in pairs),
            'heavy_final_no_skills': rate(b['heavy_final_match'] for _, b in pairs),
            'review_exhausted_skills': sum(r['stop_reason'] == 'planner_executor_review_exhausted' for r, _ in pairs),
            'mean_calls_skills': round(statistics.mean(r['calls'] for r, _ in pairs), 1),
        }
    report['P4_skill_groups'] = p4

    # ---- P5: review as an error detector ------------------------------------
    p5 = {}
    for arm in ['agent_no_skills', 'agent_skills79']:
        confusion = collections.Counter()
        rollouts_with_revise = 0
        fixed_after_revise = collections.Counter()
        for path in sorted((EXP1 / arm).glob('exp1_*.json')):
            r = json.loads(path.read_text())
            if not r.get('finished') or 'trajectory' not in r:
                continue
            g = gold[r['case_id']]
            inv = {v: k for k, v in g['label_map'].items()}
            gt = collections.Counter(k for a in g['gt_actions_pgl']
                                     for k in metrics.path_steps([chem.parse_pgl_literal(a)])[0])
            t = r['trajectory']
            proposals = {p['proposal_id']: p for p in t['executor_proposals']}
            reviews = t['planner_reviews']
            any_revise = False
            by_directive = collections.defaultdict(list)
            for rv in reviews:
                p = proposals.get(rv['proposal_id'])
                if not p or not p.get('action_pgl'):
                    continue
                try:
                    action = translate(chem.parse_pgl_literal(p['action_pgl']), inv)
                    keys = metrics.path_steps([action])[0]
                except Exception:
                    continue
                consistent = all(gt[k] > 0 for k in keys)
                decision = 'APPROVE' if rv['decision'] == 'APPROVE' else 'REVISE'
                any_revise |= decision == 'REVISE'
                confusion[(decision, 'gt_consistent' if consistent else 'gt_inconsistent')] += 1
                by_directive[p['directive_id']].append((decision, consistent))
            for seq in by_directive.values():
                # A directive whose first proposal was sent back: did the
                # finally approved one become GT-consistent?
                if seq[0][0] == 'REVISE':
                    final = next((c for d, c in reversed(seq) if d == 'APPROVE'), None)
                    fixed_after_revise[(('first_consistent' if seq[0][1] else 'first_inconsistent'),
                                        {True: 'approved_consistent', False: 'approved_inconsistent', None: 'never_approved'}[final])] += 1
            rollouts_with_revise += any_revise
        tp = confusion[('REVISE', 'gt_inconsistent')]
        fp = confusion[('REVISE', 'gt_consistent')]
        fn = confusion[('APPROVE', 'gt_inconsistent')]
        p5[arm] = {'confusion': {f'{a}|{b}': v for (a, b), v in confusion.items()},
                   'approve_rate': round(sum(v for (a, _), v in confusion.items() if a == 'APPROVE') / max(1, sum(confusion.values())), 3),
                   'error_detection_precision': round(tp / (tp + fp), 3) if tp + fp else None,
                   'error_detection_recall': round(tp / (tp + fn), 3) if tp + fn else None,
                   'rollouts_with_any_revise': rollouts_with_revise,
                   'revise_outcomes': {f'{a}->{b}': v for (a, b), v in fixed_after_revise.items()}}
    report['P5_review'] = p5
    out = ROOT / 'outputs' / 'p45_trace_analysis_20261024.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
