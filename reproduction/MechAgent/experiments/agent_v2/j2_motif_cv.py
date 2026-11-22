"""J2: offline quality of the motif library as a hidden-event candidate generator.

5-fold cross-validation by case over all usable FukuyamaBench cases (clean +
held-out, 244 unique). For each held-out fold the library is mined from the
other folds only. Reported for several selection rules: recall of GT
transient bonds, precision, fraction of transient cases with every transient
bond selected, and how many hints cases without any transient bond receive.
Fold libraries are saved for J3 (each J3 case uses the library that never saw
it); a library mined from all cases is saved for J4.
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import motifs as M

ROOT = M.ROOT
OUT = ROOT / 'outputs' / 'agent_v2_J2'
RULES = [('top4', 4, 0.0), ('top6', 6, 0.0), ('top8', 8, 0.0), ('top12', 12, 0.0),
         ('top8_p>=0.05', 8, 0.05), ('top8_p>=0.10', 8, 0.10), ('top12_p>=0.05', 12, 0.05)]


def load_cases():
    pool = {}
    for src in [ROOT / 'outputs/fukuyama/clean.jsonl', ROOT / 'outputs/heldout_fukuyama_20261024/heldout.jsonl']:
        for line in src.read_text().splitlines():
            c = json.loads(line)
            pool[c['case_id']] = c
    return pool


def fold_of(cid, k=5):
    return int(hashlib.sha256(f'J2:{cid}'.encode()).hexdigest(), 16) % k


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pool = load_cases()
    folds = collections.defaultdict(list)
    for cid in pool:
        folds[fold_of(cid)].append(cid)
    stats = {name: collections.Counter() for name, _, _ in RULES}
    enum_recall = collections.Counter()
    per_case = {}
    for f in range(5):
        lib = M.MotifLibrary()
        for g in range(5):
            if g != f:
                for cid in folds[g]:
                    lib.add_case(pool[cid])
        (OUT / f'library_fold{f}.json').write_text(json.dumps(lib.to_json()))
        for cid in folds[f]:
            case = pool[cid]
            gold = M.gold_transients(case)
            ranked = lib.rank(case)
            cands = {key for _, key in ranked}
            enum_recall['gold'] += len(gold)
            enum_recall['in_candidates'] += len(gold & cands)
            per_case[cid] = {'fold': f, 'gold': sorted(map(list, gold)), 'top12': [[round(p, 4), list(k)] for p, k in ranked[:12]]}
            for name, k, t in RULES:
                sel = set(M.select(ranked, k, t))
                s = stats[name]
                s['gold'] += len(gold)
                s['hit'] += len(sel & gold)
                s['selected'] += len(sel)
                if gold:
                    s['transient_cases'] += 1
                    s['full_recall_cases'] += gold <= sel
                else:
                    s['no_transient_cases'] += 1
                    s['hints_on_no_transient'] += len(sel)
    report = {'cases': len(pool), 'enumerator_recall': round(enum_recall['in_candidates'] / enum_recall['gold'], 4),
              'reference_llm_ranker_dev24': {'recall': 0.60, 'precision': 0.27, 'note': 'shortlist pilot, 24 dev cases'}}
    for name, s in stats.items():
        report[name] = {'recall': round(s['hit'] / s['gold'], 4), 'precision': round(s['hit'] / max(1, s['selected']), 4),
                        'full_recall_case_rate': round(s['full_recall_cases'] / max(1, s['transient_cases']), 4),
                        'mean_hints_on_no_transient_case': round(s['hints_on_no_transient'] / max(1, s['no_transient_cases']), 2)}
    full = M.MotifLibrary()
    for cid in pool:
        full.add_case(pool[cid])
    (OUT / 'library_all.json').write_text(json.dumps(full.to_json()))
    (OUT / 'folds.json').write_text(json.dumps({cid: fold_of(cid) for cid in pool}))
    (OUT / 'per_case.json').write_text(json.dumps(per_case))
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
