"""Between-group differences for KS-1 with case-clustered bootstrap CIs (the registered schema-index test)."""
import json
import random
import statistics
from pathlib import Path

ROWS = Path(__file__).resolve().parents[2] / 'outputs' / 'inert_knowledge_20261102' / 'rows_ks.jsonl'


def diff_ci(rows, field, split, n=10000, seed=20261102):
    by_case = {}
    for r in rows:
        by_case.setdefault(r['case'], []).append(r)
    cases = list(by_case)

    def stat(sample):
        a = [float(r[field]) for c in sample for r in by_case[c] if split(r)]
        b = [float(r[field]) for c in sample for r in by_case[c] if not split(r)]
        return statistics.mean(a) - statistics.mean(b) if a and b else None

    rng = random.Random(seed)
    boots = sorted(s for s in (stat([rng.choice(cases) for _ in cases]) for _ in range(n)) if s is not None)
    return [round(stat(cases), 3), round(boots[int(0.025 * len(boots))], 3), round(boots[int(0.975 * len(boots)) - 1], 3)]


def main():
    rows = [json.loads(l) for l in ROWS.read_text().splitlines() if l]
    out = {}
    for name, rs in (('heavy', [r for r in rows if r['heavy']]), ('all', rows)):
        out[name] = {
            'K_named_minus_unnamed': diff_ci(rs, 'K', lambda r: r['named']),
            'E_named_minus_unnamed': diff_ci(rs, 'E', lambda r: r['named']),
            'K_seen_minus_unseen': diff_ci(rs, 'K', lambda r: r['seen_problem']),
            'K_A_minus_C': diff_ci([r for r in rs if r['partition'] in 'AC'], 'K', lambda r: r['partition'] == 'A'),
            'E_A_minus_C': diff_ci([r for r in rs if r['partition'] in 'AC'], 'E', lambda r: r['partition'] == 'A'),
            'unnamed_cases': len({r['case'] for r in rs if not r['named']}),
        }
    path = ROWS.parent / 'SUMMARY_ks_diff.json'
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
