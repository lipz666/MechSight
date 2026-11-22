"""Score a FukuyamaBench rollout against its per-step bond-change ground truth.

The dataset records states, not arrows, so a path is compared on what each
step *changes* rather than on how the electrons were drawn. Both sides are
derived the same way: apply the step's edits to the running state and diff the
bonds, formal charges and recorded hydrogen counts against the state before it.

Unlike ``state_diff.diff_signatures``, which compares only atoms shared by two
states because it gates completion against a target that may omit byproducts,
this compares every bond. Between consecutive steps a reagent genuinely enters
and a byproduct genuinely leaves, and both are part of the step.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .standalone_chemistry import (
    apply_action,
    extract_action_payload,
    parse_pgl_literal,
    state_signature,
)

Change = Tuple[str, ...]


def _anon(label: str) -> str:
    """Collapse hydrogen labels: protons are indistinguishable.

    With several mobile hydrogens on one heavy atom, which of them the agent
    moves is arbitrary, and the ground truth's choice is equally arbitrary.
    """

    return "H*" if len(label) > 1 and label[0] == "H" and label[1:].isdigit() else label


def _bond_key(kind: str, first: str, second: str) -> Change:
    return (kind,) + tuple(sorted((_anon(first), _anon(second))))


def _changes_between(before: Dict[str, Any], after: Dict[str, Any]) -> Counter:
    """Every bond, charge and Hcount difference, as comparable keys."""

    before_bonds = before.get("bonds", {})
    after_bonds = after.get("bonds", {})
    changes: Counter = Counter()
    for key in set(after_bonds) - set(before_bonds):
        changes[_bond_key("form", *key.split("|"))] += 1
    for key in set(before_bonds) - set(after_bonds):
        changes[_bond_key("break", *key.split("|"))] += 1
    for key in set(before_bonds) & set(after_bonds):
        if before_bonds[key].get("order") != after_bonds[key].get("order"):
            changes[_bond_key("order", *key.split("|"))] += 1
    before_atoms = before.get("atoms", {})
    after_atoms = after.get("atoms", {})
    for name in set(before_atoms) & set(after_atoms):
        if before_atoms[name].get("charge") != after_atoms[name].get("charge"):
            changes[("charge", _anon(name))] += 1
        if before_atoms[name].get("Hcount") != after_atoms[name].get("Hcount"):
            changes[("hcount", _anon(name))] += 1
    return changes


def gold_steps(record: Dict[str, Any]) -> List[Counter]:
    """The annotated per-step changes, in the same key shape."""

    steps: List[Counter] = []
    for step in record["gt_step_changes"]:
        changes: Counter = Counter()
        for first, second, _kind in step["bonds_formed"]:
            changes[_bond_key("form", first, second)] += 1
        for first, second, _kind in step["bonds_broken"]:
            changes[_bond_key("break", first, second)] += 1
        for first, second, _before, _after in step["bond_order_changes"]:
            changes[_bond_key("order", first, second)] += 1
        for name, _before, _after in step["charge_changes"]:
            changes[("charge", _anon(name))] += 1
        steps.append(changes)
    return steps


def predicted_steps(rollout: Dict[str, Any]) -> List[Counter]:
    """Replay the accepted actions, taking the change each one makes."""

    state = parse_pgl_literal(rollout["reaction"]["initial_state_pgl"])
    steps: List[Counter] = []
    for action_record in rollout.get("actions", []):
        if not action_record.get("accepted_into_history"):
            continue
        action, _ = extract_action_payload(f"<action>{action_record['action_pgl']}</action>")
        before = state_signature(state)
        state, _ = apply_action(state, action)
        steps.append(_changes_between(before, state_signature(state)))
    return steps


def compare(gold: Sequence[Counter], predicted: Sequence[Counter]) -> Dict[str, Any]:
    flat_gold: Counter = Counter()
    for step in gold:
        flat_gold.update(step)
    flat_predicted: Counter = Counter()
    for step in predicted:
        flat_predicted.update(step)

    matched = sum(min(flat_predicted[key], flat_gold[key]) for key in flat_gold)
    aligned = 0
    exact_steps = 0
    for index in range(max(len(gold), len(predicted))):
        gold_step = gold[index] if index < len(gold) else Counter()
        predicted_step = predicted[index] if index < len(predicted) else Counter()
        aligned += sum(min(predicted_step[key], gold_step[key]) for key in gold_step)
        if index < len(gold) and gold_step == predicted_step:
            exact_steps += 1
    return {
        "gold_steps": len(gold),
        "predicted_steps": len(predicted),
        "gold_changes": sum(flat_gold.values()),
        "predicted_changes": sum(flat_predicted.values()),
        # L1: the right changes happened, however they were grouped.
        "change_set_match": flat_gold == flat_predicted,
        # L3: the annotation's step boundaries were reproduced as well.
        "step_partition_match": len(gold) == len(predicted) and exact_steps == len(gold),
        "matched_changes": matched,
        "aligned_matched_changes": aligned,
        "exact_steps": exact_steps,
    }


def _f1(matched: int, predicted: int, gold: int) -> float:
    precision = matched / predicted if predicted else 0.0
    recall = matched / gold if gold else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def score(dataset: Path, rollouts: Path) -> Dict[str, Any]:
    cases = {
        json.loads(line)["case_id"]: json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    rows: List[Dict[str, Any]] = []
    for line in rollouts.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rollout = json.loads(line)
        case_id = str(rollout["reaction"]["reaction_id"]).split("::")[-1]
        record = cases.get(case_id)
        if record is None:
            continue
        result = compare(gold_steps(record), predicted_steps(rollout))
        result["case_id"] = case_id
        result["stop_reason"] = rollout["reaction"].get("stop_reason")
        rows.append(result)

    n = len(rows)
    gold_total = sum(r["gold_changes"] for r in rows)
    predicted_total = sum(r["predicted_changes"] for r in rows)
    return {
        "cases": n,
        "L1_change_set": sum(r["change_set_match"] for r in rows) / n if n else 0.0,
        "L3_step_partition": sum(r["step_partition_match"] for r in rows) / n if n else 0.0,
        "change_f1_boundary_agnostic": _f1(
            sum(r["matched_changes"] for r in rows), predicted_total, gold_total
        ),
        "change_f1_step_aligned": _f1(
            sum(r["aligned_matched_changes"] for r in rows), predicted_total, gold_total
        ),
        "step_count_match": sum(
            r["gold_steps"] == r["predicted_steps"] for r in rows
        ) / n if n else 0.0,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="fukuyama clean.jsonl")
    parser.add_argument("rollouts", type=Path, help="merged analysis JSONL")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = score(args.dataset, args.rollouts)
    print(f"cases {report['cases']}")
    print(f"  L1 change set          {report['L1_change_set']:.1%}")
    print(f"  L3 step partition      {report['L3_step_partition']:.1%}")
    print(f"  change F1 (no bounds)  {report['change_f1_boundary_agnostic']:.1%}")
    print(f"  change F1 (step bound) {report['change_f1_step_aligned']:.1%}")
    print(f"  step count == gold     {report['step_count_match']:.1%}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
