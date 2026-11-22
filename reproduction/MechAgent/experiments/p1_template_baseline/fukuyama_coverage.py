"""P1b: how much of FukuyamaBench is a chain of MechFinder templates?

A Fukuyama case is a multi-reaction sequence, so one template cannot cover it.
Given the GT intermediate states (an oracle no deployed method has), every
window state_i -> state_j is tested against the MechFinder exemplar pool with
the P1 instantiation procedure; only target-verified instantiations count.
A dynamic program then asks whether the whole path splits into covered
windows. This is an upper bound on what a "segment + template" method could
reach, and it labels each case as template-decomposable or not.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as P
from develop import standalone_chemistry as chem
from prepare import normalize_endpoints

FUKUYAMA = P.ROOT / 'outputs' / 'fukuyama' / 'clean.jsonl'


def with_pending_reagents(state, cumulative):
    """Add reagents that the source introduces only at a later step.

    Labels are global after stitching, so an atom missing from an
    intermediate is taken, with its bonds to other missing atoms, from the
    cumulative initial state.
    """
    present = dict(chem.pgl.iter_atoms(state))
    extra_atoms = {n: a for n, a in chem.pgl.iter_atoms(cumulative) if n not in present}
    if not extra_atoms:
        return state
    bonds = [b for (x, y), b in chem.pgl.bond_map(cumulative).items() if x in extra_atoms and y in extra_atoms]
    species = list(state.species) + [chem.Species('reactant', 'stable', chem.Molecule('', [chem.Substructure('', extra_atoms, bonds)]))]
    return chem.ReactionState(species)


def window_test(initial, target, by_sig, limit):
    try:
        start, end, _ = normalize_endpoints(initial, target, 0)
    except Exception as exc:
        return None, 'untestable:' + str(exc)[:40]
    test = P.prepare_test({'initial_state_pgl': repr(start), 'target_state_pgl': repr(end)})
    for ex in by_sig.get(test['signature'], []):
        mapped, _, verified, _ = P.instantiate(ex, test, limit)
        if verified:
            return ex['reaction_class'], 'covered'
    return None, 'no_template'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=P.ROOT / 'outputs' / 'p1_template_20261024' / 'fukuyama_coverage.json')
    ap.add_argument('--per-class', type=int, default=40)
    ap.add_argument('--mapping-limit', type=int, default=200)
    ap.add_argument('--max-window', type=int, default=8)
    args = ap.parse_args()
    buckets = collections.defaultdict(list)
    with P.CORPUS.open() as f:
        for line in f:
            row = json.loads(line)
            rid = f"mechfinder::{row['reaction_class']}::{row['source_row_index']}"
            buckets[row['reaction_class']].append((hashlib.sha256(f'{P.SEED}:{rid}'.encode()).hexdigest(), row))
    by_sig = collections.defaultdict(list)
    for cls, rows in sorted(buckets.items()):
        taken = 0
        for _, row in sorted(rows, key=lambda x: x[0]):
            if taken >= args.per_class:
                break
            try:
                ex = P.load_exemplar(row)
            except Exception:
                continue
            by_sig[ex['signature']].append(ex)
            taken += 1
    cases = [json.loads(l) for l in FUKUYAMA.read_text().splitlines()]
    results = []
    for case in cases:
        t0 = time.time()
        cumulative = chem.parse_pgl_literal(case['initial_state_pgl'])
        states = [with_pending_reagents(chem.parse_pgl_literal(s), cumulative) for s in case['states_pgl']]
        n = len(states) - 1
        windows = {}
        for i in range(n):
            for j in range(i + 1, min(n, i + args.max_window) + 1):
                windows[(i, j)] = window_test(states[i], states[j], by_sig, args.mapping_limit)
        # best[j] = most GT steps inside covered windows for a split of 0..j
        best = [0] + [-1] * n
        full = [True] + [False] * n
        for j in range(1, n + 1):
            best[j] = best[j - 1]
            for i in range(j):
                if (i, j) in windows and windows[(i, j)][1] == 'covered':
                    best[j] = max(best[j], best[i] + (j - i))
                    full[j] = full[j] or full[i]
        status = collections.Counter(v[1] for v in windows.values())
        results.append({'case_id': case['case_id'], 'partition': case['partition'], 'gt_steps': n,
                        'single_template_whole_path': windows.get((0, n), (None, ''))[1] == 'covered',
                        'fully_decomposable': full[n], 'steps_in_covered_windows': best[n],
                        'covered_windows': sorted([[i, j, windows[(i, j)][0]] for (i, j) in windows if windows[(i, j)][1] == 'covered']),
                        'window_status': dict(status), 'seconds': round(time.time() - t0, 1)})
        print(json.dumps({k: results[-1][k] for k in ('case_id', 'gt_steps', 'fully_decomposable', 'steps_in_covered_windows', 'window_status')}), flush=True)
    summary = {}
    for part in ['A', 'B', 'C', 'all']:
        sel = [r for r in results if part == 'all' or r['partition'] == part]
        summary[part] = {'cases': len(sel),
                         'single_template_whole_path': sum(r['single_template_whole_path'] for r in sel) / len(sel),
                         'fully_decomposable_with_oracle_intermediates': sum(r['fully_decomposable'] for r in sel) / len(sel),
                         'step_coverage': sum(r['steps_in_covered_windows'] for r in sel) / sum(r['gt_steps'] for r in sel)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'config': {k: str(v) for k, v in vars(args).items()}, 'summary': summary,
                                       'cases': results}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
