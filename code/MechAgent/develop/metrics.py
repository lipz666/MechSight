"""Layered comparison between predicted and annotated mechanism paths.

The previous single metric compared predicted and gold electron transfers only
after aligning them by step position. That conflates four independent questions:

1. Did the model move the right electrons at all?
2. Did it move them in the right causal order?
3. Did it draw the same concerted/stepwise boundaries as the annotation?
4. Did it pick the same explicit hydrogen label as the annotation?

Question 4 is pure bookkeeping: protons are indistinguishable and the annotated
``H900003``-style labels are generated, not observed. Question 3 is an
annotation convention: MechFinder emits a fixed step count per reaction class,
so the same SN2 event is annotated as one step for a halide and as step 4 of 5
for a tosylate. Questions 1 and 2 are the actual chemistry.

This module reports the four levels separately instead of collapsing them into
one number. ``strict_step_alignment`` reproduces the legacy metric exactly, so
old numbers remain comparable.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

MARKERS = {
    "atom",
    "lone_pair",
    "single",
    "double",
    "triple",
    "aromatic",
    "single_bond",
    "double_bond",
    "triple_bond",
    "aromatic_bond",
    "pi_bond",
    "sigma_bond",
    "new_sigma_bond",
    "new_pi_bond",
    "hydride",
}

TransferKey = Tuple[Tuple[str, ...], Tuple[str, ...]]


def _is_hydrogen(label: str) -> bool:
    """True for an atom label naming a hydrogen, e.g. ``H302`` or ``H900003``."""

    return len(label) > 1 and label[0] == "H" and label[1:].isdigit()


def _endpoint_atoms(endpoint: Any, *, anonymize_h: bool) -> Tuple[str, ...]:
    atoms: List[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
        elif value is not None and str(value) not in MARKERS:
            atoms.append(str(value))

    collect(endpoint)
    if anonymize_h:
        atoms = ["H*" if _is_hydrogen(atom) else atom for atom in atoms]
    return tuple(sorted(atoms))


def transfer_key(transfer: Any, *, anonymize_h: bool = True) -> TransferKey:
    """Normalize one electron transfer into a comparable (source, target) key.

    Endpoint markers are dropped because the annotation and the Executor speak
    different dialects for the same thing: gold writes ``source='O205'`` where
    the Executor is required to write ``source=('O205', 'lone_pair')``. Atoms
    already named by the source are removed from the target so that
    ``target=('O1', 'H302', 'new_sigma_bond')`` matches gold's ``target='H302'``.
    """

    source = _get(transfer, "source")
    target = _get(transfer, "target")
    source_atoms = _endpoint_atoms(source, anonymize_h=anonymize_h)
    target_atoms = _endpoint_atoms(target, anonymize_h=anonymize_h)
    if len(target_atoms) > 1:
        target_without_source = tuple(atom for atom in target_atoms if atom not in source_atoms)
        if target_without_source:
            target_atoms = tuple(sorted(target_without_source))
    return source_atoms, target_atoms


def _get(transfer: Any, name: str) -> Any:
    if isinstance(transfer, dict):
        return transfer.get(name)
    return getattr(transfer, name, None)


def _transfers_of(action: Any) -> List[Any]:
    if isinstance(action, dict):
        return list(action.get("electron_transfers") or [])
    return list(getattr(action, "electron_transfers", None) or [])


def path_steps(actions: Iterable[Any], *, anonymize_h: bool = True) -> List[List[TransferKey]]:
    """Normalize a sequence of ReactionTransformations into per-step key lists."""

    return [
        [transfer_key(transfer, anonymize_h=anonymize_h) for transfer in _transfers_of(action)]
        for action in actions
    ]


@dataclass
class PathComparison:
    """One predicted path scored against one annotated path."""

    reaction_id: str
    repeat_index: int = 0
    predicted_steps: int = 0
    gold_steps: int = 0

    # Level 1: are the electron movements themselves right, ignoring how they
    # were grouped into steps and in what order they were emitted?
    transfer_set_match: bool = False
    # Level 2: level 1 plus causal order of the flattened transfer sequence.
    transfer_sequence_match: bool = False
    # Level 3: level 2 plus identical step boundaries. Equals the legacy
    # ``full_path_exact``.
    step_partition_match: bool = False

    set_matched: int = 0
    aligned_matched: int = 0
    gold_transfers: int = 0
    predicted_transfers: int = 0
    exact_steps: int = 0

    missing_transfers: List[TransferKey] = field(default_factory=list)
    extra_transfers: List[TransferKey] = field(default_factory=list)
    boundary_diagnosis: str = "content_mismatch"
    h_label_only: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "reaction_id": self.reaction_id,
            "repeat_index": self.repeat_index,
            "predicted_steps": self.predicted_steps,
            "gold_steps": self.gold_steps,
            "transfer_set_match": self.transfer_set_match,
            "transfer_sequence_match": self.transfer_sequence_match,
            "step_partition_match": self.step_partition_match,
            "set_matched": self.set_matched,
            "aligned_matched": self.aligned_matched,
            "gold_transfers": self.gold_transfers,
            "predicted_transfers": self.predicted_transfers,
            "exact_steps": self.exact_steps,
            "boundary_diagnosis": self.boundary_diagnosis,
            "h_label_only": self.h_label_only,
            "missing_transfers": [list(map(list, key)) for key in self.missing_transfers],
            "extra_transfers": [list(map(list, key)) for key in self.extra_transfers],
        }
        return payload


def compare_path(
    predicted_actions: Sequence[Any],
    gold_actions: Sequence[Any],
    *,
    reaction_id: str = "",
    repeat_index: int = 0,
    anonymize_h: bool = True,
) -> PathComparison:
    predicted = path_steps(predicted_actions, anonymize_h=anonymize_h)
    gold = path_steps(gold_actions, anonymize_h=anonymize_h)

    flat_predicted = [key for step in predicted for key in step]
    flat_gold = [key for step in gold for key in step]
    predicted_counts = Counter(flat_predicted)
    gold_counts = Counter(flat_gold)

    set_matched = sum(min(predicted_counts[key], gold_counts[key]) for key in gold_counts)
    transfer_set_match = predicted_counts == gold_counts
    # Transfers inside one ReactionTransformation are concerted, so their
    # emission order carries no chemical meaning. Sorting within each step keeps
    # L3 (same partition) a strict refinement of L2 (same causal order).
    transfer_sequence_match = [
        key for step in predicted for key in sorted(step)
    ] == [key for step in gold for key in sorted(step)]

    exact_steps = 0
    aligned_matched = 0
    for index in range(max(len(predicted), len(gold))):
        step_predicted = Counter(predicted[index]) if index < len(predicted) else Counter()
        step_gold = Counter(gold[index]) if index < len(gold) else Counter()
        aligned_matched += sum(min(step_predicted[key], step_gold[key]) for key in step_gold)
        if index < len(gold) and step_predicted == step_gold:
            exact_steps += 1
    step_partition_match = (
        bool(gold) and len(predicted) == len(gold) and exact_steps == len(gold)
    )

    missing = list((gold_counts - predicted_counts).elements())
    extra = list((predicted_counts - gold_counts).elements())

    if transfer_set_match and step_partition_match:
        diagnosis = "match"
    elif transfer_set_match and len(predicted) < len(gold):
        diagnosis = "merged_steps"
    elif transfer_set_match and len(predicted) > len(gold):
        diagnosis = "split_steps"
    elif transfer_set_match:
        diagnosis = "reordered"
    elif not extra and missing:
        diagnosis = "missing_events"
    elif extra and not missing:
        diagnosis = "extra_events"
    else:
        diagnosis = "content_mismatch"

    h_label_only = False
    if anonymize_h and transfer_set_match:
        strict = compare_path(
            predicted_actions,
            gold_actions,
            reaction_id=reaction_id,
            repeat_index=repeat_index,
            anonymize_h=False,
        )
        h_label_only = not strict.transfer_set_match

    return PathComparison(
        reaction_id=reaction_id,
        repeat_index=repeat_index,
        predicted_steps=len(predicted),
        gold_steps=len(gold),
        transfer_set_match=transfer_set_match,
        transfer_sequence_match=transfer_sequence_match,
        step_partition_match=step_partition_match,
        set_matched=set_matched,
        aligned_matched=aligned_matched,
        gold_transfers=len(flat_gold),
        predicted_transfers=len(flat_predicted),
        exact_steps=exact_steps,
        missing_transfers=missing,
        extra_transfers=extra,
        boundary_diagnosis=diagnosis,
        h_label_only=h_label_only,
    )


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall)


def clustered_bootstrap_ci(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    iterations: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile CI resampling whole reactions, not individual rollouts.

    Repeats of one reaction are not independent samples: 48 of 60 tasks in the
    Gemini run produced the same judgment all three times. Resampling rollouts
    would report a confidence interval for n=180 when the design only has n=60.
    """

    if not values:
        return (0.0, 0.0)
    grouped: Dict[str, List[float]] = {}
    for value, cluster in zip(values, clusters):
        grouped.setdefault(cluster, []).append(value)
    keys = list(grouped)
    if len(keys) < 2:
        mean = sum(values) / len(values)
        return (mean, mean)
    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(iterations):
        sampled: List[float] = []
        for _ in range(len(keys)):
            sampled.extend(grouped[keys[rng.randrange(len(keys))]])
        means.append(sum(sampled) / len(sampled))
    means.sort()
    low = means[int((alpha / 2) * len(means))]
    high = means[min(len(means) - 1, int((1 - alpha / 2) * len(means)))]
    return (low, high)


def aggregate(comparisons: Sequence[PathComparison], *, seed: int = 0) -> Dict[str, Any]:
    """Aggregate per-path comparisons into the layered report."""

    n = len(comparisons)
    clusters = [comparison.reaction_id for comparison in comparisons]

    def rate(predicate) -> Dict[str, Any]:
        values = [1.0 if predicate(comparison) else 0.0 for comparison in comparisons]
        count = int(sum(values))
        low, high = clustered_bootstrap_ci(values, clusters, seed=seed)
        return {
            "count": count,
            "rate": _safe_div(count, n),
            "ci95": [low, high],
        }

    gold_transfers = sum(comparison.gold_transfers for comparison in comparisons)
    predicted_transfers = sum(comparison.predicted_transfers for comparison in comparisons)
    set_matched = sum(comparison.set_matched for comparison in comparisons)
    aligned_matched = sum(comparison.aligned_matched for comparison in comparisons)
    gold_steps = sum(comparison.gold_steps for comparison in comparisons)
    exact_steps = sum(comparison.exact_steps for comparison in comparisons)

    set_precision = _safe_div(set_matched, predicted_transfers)
    set_recall = _safe_div(set_matched, gold_transfers)
    aligned_precision = _safe_div(aligned_matched, predicted_transfers)
    aligned_recall = _safe_div(aligned_matched, gold_transfers)

    return {
        "num_paths": n,
        "num_reactions": len(set(clusters)),
        "levels": {
            "L1_transfer_set": rate(lambda c: c.transfer_set_match),
            "L2_transfer_sequence": rate(lambda c: c.transfer_sequence_match),
            "L3_step_partition": rate(lambda c: c.step_partition_match),
        },
        "transfer_scores": {
            "set_precision": set_precision,
            "set_recall": set_recall,
            "set_f1": _f1(set_precision, set_recall),
            "step_aligned_precision": aligned_precision,
            "step_aligned_recall": aligned_recall,
            "step_aligned_f1": _f1(aligned_precision, aligned_recall),
        },
        "step_scores": {
            "exact_steps": exact_steps,
            "gold_steps": gold_steps,
            "exact_step_rate": _safe_div(exact_steps, gold_steps),
        },
        "diagnosis": dict(Counter(c.boundary_diagnosis for c in comparisons)),
        "h_label_only_paths": sum(1 for c in comparisons if c.h_label_only),
        "step_count_relation": dict(
            Counter(
                "same"
                if c.predicted_steps == c.gold_steps
                else ("more" if c.predicted_steps > c.gold_steps else "fewer")
                for c in comparisons
            )
        ),
    }


def render_report(report: Dict[str, Any]) -> str:
    """Human-readable summary emphasising what each level actually measures."""

    levels = report["levels"]
    scores = report["transfer_scores"]
    lines = [
        f"paths={report['num_paths']}  distinct reactions={report['num_reactions']}",
        "",
        "Layered path accuracy (95% CI resampled over reactions, not rollouts):",
        f"  L1 transfer set      {levels['L1_transfer_set']['rate']:7.1%}"
        f"  [{levels['L1_transfer_set']['ci95'][0]:.1%}, {levels['L1_transfer_set']['ci95'][1]:.1%}]"
        "   right electrons moved",
        f"  L2 transfer sequence {levels['L2_transfer_sequence']['rate']:7.1%}"
        f"  [{levels['L2_transfer_sequence']['ci95'][0]:.1%}, {levels['L2_transfer_sequence']['ci95'][1]:.1%}]"
        "   + right causal order",
        f"  L3 step partition    {levels['L3_step_partition']['rate']:7.1%}"
        f"  [{levels['L3_step_partition']['ci95'][0]:.1%}, {levels['L3_step_partition']['ci95'][1]:.1%}]"
        "   + annotation's step boundaries (legacy metric)",
        "",
        "Transfer-level scores:",
        f"  boundary-agnostic   P={scores['set_precision']:.1%} R={scores['set_recall']:.1%}"
        f" F1={scores['set_f1']:.1%}",
        f"  step-position bound P={scores['step_aligned_precision']:.1%}"
        f" R={scores['step_aligned_recall']:.1%} F1={scores['step_aligned_f1']:.1%}  (legacy)",
        "",
        f"Paths matching only after H-label anonymization: {report['h_label_only_paths']}",
        f"Predicted vs gold step count: {report['step_count_relation']}",
        "Failure diagnosis: " + ", ".join(f"{k}={v}" for k, v in sorted(report["diagnosis"].items())),
    ]
    return "\n".join(lines)
