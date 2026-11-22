"""Compare two runs of the same reaction set under the layered metrics.

Built for the step-count ablation: one run plans blind, the other is told how
many elementary steps the annotation used. Matching that count is the strongest
single predictor of scoring correct, and MechFinder fixes it per reaction class,
so the difference between the two runs separates guessing the annotation's
convention from doing the chemistry.

Reactions present in only one run are dropped, so a partially completed
ablation still compares like with like.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from . import metrics
from .evaluate_gt import PathRecord, iter_compact, score


def _by_reaction(records: Sequence[PathRecord]) -> Dict[str, List[PathRecord]]:
    grouped: Dict[str, List[PathRecord]] = defaultdict(list)
    for record in records:
        grouped[record.reaction_id].append(record)
    return grouped


def align(
    baseline: Sequence[PathRecord], variant: Sequence[PathRecord]
) -> Tuple[List[PathRecord], List[PathRecord], List[str]]:
    left = _by_reaction(baseline)
    right = _by_reaction(variant)
    shared = sorted(set(left) & set(right))
    dropped = sorted((set(left) | set(right)) - set(shared))
    return (
        [record for reaction in shared for record in left[reaction]],
        [record for reaction in shared for record in right[reaction]],
        dropped,
    )


def _comparisons(records: Sequence[PathRecord]) -> List[metrics.PathComparison]:
    return [
        metrics.compare_path(
            record.predicted_actions,
            record.gold_actions,
            reaction_id=record.reaction_id,
            repeat_index=record.repeat_index,
        )
        for record in records
    ]


def compare(
    baseline: Sequence[PathRecord],
    variant: Sequence[PathRecord],
    *,
    seed: int = 0,
) -> Dict[str, Any]:
    left, right, dropped = align(baseline, variant)
    base_report = score(left, seed=seed)
    var_report = score(right, seed=seed)

    base_comparisons = _comparisons(left)
    var_comparisons = _comparisons(right)

    # Paired at the rollout level where repeat indices line up, which is what
    # makes the delta meaningful on a 60-reaction design.
    base_by_key = {(c.reaction_id, c.repeat_index): c for c in base_comparisons}
    var_by_key = {(c.reaction_id, c.repeat_index): c for c in var_comparisons}
    paired_keys = sorted(set(base_by_key) & set(var_by_key))

    transitions: Dict[str, Counter] = {}
    for level in ("transfer_set_match", "transfer_sequence_match", "step_partition_match"):
        counter: Counter = Counter()
        for key in paired_keys:
            before = getattr(base_by_key[key], level)
            after = getattr(var_by_key[key], level)
            counter[f"{'pass' if before else 'fail'}->{'pass' if after else 'fail'}"] += 1
        transitions[level] = counter

    def step_match_rate(comparisons: Sequence[metrics.PathComparison]) -> float:
        if not comparisons:
            return 0.0
        return sum(1 for c in comparisons if c.predicted_steps == c.gold_steps) / len(comparisons)

    return {
        "num_shared_reactions": len(set(c.reaction_id for c in base_comparisons)),
        "dropped_reactions": dropped,
        "baseline": base_report,
        "variant": var_report,
        "paired_rollouts": len(paired_keys),
        "transitions": {level: dict(counter) for level, counter in transitions.items()},
        "step_count_hit_rate": {
            "baseline": step_match_rate(base_comparisons),
            "variant": step_match_rate(var_comparisons),
        },
    }


def render(report: Dict[str, Any], *, baseline_name: str, variant_name: str) -> str:
    lines = [
        f"shared reactions: {report['num_shared_reactions']}"
        f"   paired rollouts: {report['paired_rollouts']}",
    ]
    if report["dropped_reactions"]:
        lines.append(
            f"dropped (present in only one run): {len(report['dropped_reactions'])}"
        )
    lines += [
        "",
        f"{'level':22s} {baseline_name:>12s} {variant_name:>12s} {'delta':>9s}",
    ]
    for level in ("L1_transfer_set", "L2_transfer_sequence", "L3_step_partition"):
        before = report["baseline"]["levels"][level]["rate"]
        after = report["variant"]["levels"][level]["rate"]
        lines.append(
            f"{level:22s} {before:11.1%} {after:12.1%} {after - before:+9.1%}"
        )

    lines += ["", f"{'transfer F1':22s} {baseline_name:>12s} {variant_name:>12s} {'delta':>9s}"]
    for key, label in (
        ("set_f1", "boundary-agnostic"),
        ("step_aligned_f1", "step-position bound"),
    ):
        before = report["baseline"]["transfer_scores"][key]
        after = report["variant"]["transfer_scores"][key]
        lines.append(f"{label:22s} {before:11.1%} {after:12.1%} {after - before:+9.1%}")

    hit = report["step_count_hit_rate"]
    lines += [
        "",
        f"predicted step count == annotated: {hit['baseline']:.1%} -> {hit['variant']:.1%}",
        "",
        "Paired transitions (baseline -> variant):",
    ]
    for level, counter in report["transitions"].items():
        parts = ", ".join(f"{k} {v}" for k, v in sorted(counter.items()))
        lines.append(f"  {level:24s} {parts}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two compact analysis JSONL runs.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("variant", type=Path)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--variant-name", default="variant")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = compare(
        list(iter_compact(args.baseline)),
        list(iter_compact(args.variant)),
        seed=args.seed,
    )
    print(render(report, baseline_name=args.baseline_name, variant_name=args.variant_name))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
