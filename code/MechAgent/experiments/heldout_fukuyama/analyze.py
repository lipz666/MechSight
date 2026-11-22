"""Pre-registered analysis of the confirmatory run (written before any result).

Sets (REGISTRATION.json): primary = T0+T1+T2 (123 cases); sensitivity = T3
(95 cases) and all 218.

Per case, a family's score is the mean over its samples:
  direct     direct, direct@2                (2 samples)
  shortlist  shortlist, shortlist@2          (2 samples; one shared rank call)
Primary contrast: shortlist - direct on sigma_L1 over the primary set,
case-clustered bootstrap 95% CI (2000 resamples, seed 20261024). The claim
"shortlist improves sigma_L1" is supported only if the CI excludes 0.

Compute-matched control: direct gets a third sample (direct@3), so its total
generation exceeds shortlist's (rank call + 2 paths). Its deployable use is
vote@3: the most frequent sigma-event multiset among the 3 direct samples
(ties -> earliest sample). Secondary contrast: shortlist - direct_vote3.

Secondary metrics: intermediates_in_order (same contrasts); sigma_L1 on the
GT-transient subset (diagnostic, not deployable); spurious-transient rate on
no-transient cases (fraction of samples that predict any transient bond);
ranker recall/precision against GT transient bonds; generation tokens.
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
from develop import standalone_chemistry as chem

BASE = ROOT / 'outputs' / 'heldout_fukuyama_20261024'
RUN = BASE / 'confirmatory'
FAMILIES = {'direct': ['direct', 'direct@2'], 'shortlist': ['shortlist', 'shortlist@2']}
METRICS = ['sigma_L1', 'intermediates_in_order']
SEED = 20261024


def load(arm, cid):
    f = RUN / arm / f'{cid}.json'
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    return r if r.get('finished') else None


def sigma_signature(case, actions):
    states, _ = fm.replay(case['initial_state_pgl'], actions)
    initial = chem.parse_pgl_literal(case['initial_state_pgl'])
    steps = [fm.step_from_states(b, a) for b, a in zip([initial] + states[:-1], states)]
    return tuple(sorted(sum((fm.sigma_events(s) for s in steps), collections.Counter()).items()))


def ci(values, ids):
    lo, hi = metrics.clustered_bootstrap_ci(values, ids, seed=SEED)
    return [round(lo, 4), round(hi, 4)]


def contrast(per, a, b, ids, m):
    keys = [c for c in ids if c in per[a] and c in per[b]]
    d = [per[b][c][m] - per[a][c][m] for c in keys]
    return {'cases': len(keys), 'delta': round(statistics.mean(d), 4) if d else None,
            'ci95': ci(d, keys) if d else None,
            'better_cases': sum(x > 0 for x in d), 'worse_cases': sum(x < 0 for x in d)}


def main():
    reg = json.loads((BASE / 'REGISTRATION.json').read_text())
    cases = {c['case_id']: c for c in map(json.loads, (BASE / 'heldout.jsonl').read_text().splitlines())}
    per = collections.defaultdict(dict)
    spurious = collections.defaultdict(list)
    transient = {cid: len(fm.sigma_scores([], fm.gold_steps_from_clean(c))['gold_transient']) for cid, c in cases.items()}
    tokens = collections.defaultdict(list)
    missing = collections.Counter()
    for cid, case in cases.items():
        for fam, arms in FAMILIES.items():
            scores = []
            for arm in arms:
                r = load(arm, cid)
                if r is None:
                    missing[arm] += 1
                    continue
                s = fm.score_case(case, r['actions_pgl'])
                scores.append(s)
                tokens[fam].append(r['generation_tokens'])
                if transient[cid] == 0:
                    spurious[fam].append(bool(s['predicted_transient']))
            if len(scores) == len(arms):
                per[fam][cid] = {m: statistics.mean(float(s[m]) for s in scores) for m in METRICS}
        # Compute-matched control: vote over the three direct samples.
        samples = [(arm, load(arm, cid)) for arm in ('direct', 'direct@2', 'direct@3')]
        if all(r is not None for _, r in samples):
            sigs = [sigma_signature(case, r['actions_pgl']) for _, r in samples]
            counts = collections.Counter(sigs)
            pick = next(i for i, s in enumerate(sigs) if counts[s] == max(counts.values()))
            s = fm.score_case(case, samples[pick][1]['actions_pgl'])
            per['direct_vote3'][cid] = {m: float(s[m]) for m in METRICS}
            tokens['direct_vote3'].append(sum(r['generation_tokens'] for _, r in samples))
        rank = RUN / 'rank' / f'{cid}.json'
        if rank.exists():
            tokens['rank'].append(json.loads(rank.read_text())['generation_tokens'])

    report = {'missing_results': dict(missing), 'sets': {}}
    for name, ids in reg['sets'].items():
        block = {'cases': len(ids)}
        for fam in ('direct', 'shortlist', 'direct_vote3'):
            have = [c for c in ids if c in per[fam]]
            block[fam] = {'cases': len(have), **{m: round(statistics.mean(per[fam][c][m] for c in have), 4) if have else None for m in METRICS}}
        for m in METRICS:
            block[f'{m}: shortlist - direct'] = contrast(per, 'direct', 'shortlist', ids, m)
            block[f'{m}: shortlist - direct_vote3'] = contrast(per, 'direct_vote3', 'shortlist', ids, m)
        trans_ids = [c for c in ids if transient[c] > 0]
        block['diagnostic_transient_subset'] = {'cases': len(trans_ids),
                                                'sigma_L1: shortlist - direct': contrast(per, 'direct', 'shortlist', trans_ids, 'sigma_L1')}
        report['sets'][name] = block
    report['spurious_transient_rate_no_transient_cases'] = {f: round(statistics.mean(v), 4) if v else None for f, v in spurious.items()}
    hit = gold = short = 0
    for cid, case in cases.items():
        f = RUN / 'rank' / f'{cid}.json'
        if not f.exists():
            continue
        sl = {tuple(k) for k in json.loads(f.read_text())['shortlist']}
        gt = {tuple(k) for k in fm.sigma_scores([], fm.gold_steps_from_clean(case))['gold_transient']}
        hit, gold, short = hit + len(sl & gt), gold + len(gt), short + len(sl)
    report['ranker'] = {'recall': round(hit / gold, 4) if gold else None, 'precision': round(hit / short, 4) if short else None,
                        'mean_shortlist': round(short / len(cases), 2)}
    report['mean_generation_tokens'] = {k: round(statistics.mean(v)) for k, v in tokens.items() if v}
    primary = report['sets']['primary_T0_T1_T2']['sigma_L1: shortlist - direct']
    report['preregistered_verdict'] = ('SUPPORTED: CI excludes 0' if primary['ci95'] and primary['ci95'][0] > 0
                                       else 'NOT SUPPORTED: CI includes 0' if primary['ci95'] else 'incomplete')
    (RUN / 'ANALYSIS.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
