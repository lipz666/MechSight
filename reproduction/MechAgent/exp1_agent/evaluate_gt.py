from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .chemistry import DEFAULT_SKILLRL_ROOT, ChemistryBridge


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
    "pi_bond",
    "sigma_bond",
    "new_sigma_bond",
    "new_pi_bond",
    "hydride",
}


def _endpoint_atoms(endpoint: Any) -> Tuple[str, ...]:
    atoms: List[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
        elif value is not None and str(value) not in MARKERS:
            atoms.append(str(value))

    collect(endpoint)
    return tuple(sorted(atoms))


def _transfer_key(transfer: Dict[str, Any]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    source_atoms = _endpoint_atoms(transfer.get("source"))
    target_atoms = _endpoint_atoms(transfer.get("target"))
    if len(target_atoms) > 1:
        target_without_source = tuple(atom for atom in target_atoms if atom not in source_atoms)
        if target_without_source:
            target_atoms = tuple(sorted(target_without_source))
    return source_atoms, target_atoms


def _literal_endpoint(endpoint: Any) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    values: List[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
        elif value is not None:
            values.append(str(value))

    collect(endpoint)
    atoms = tuple(sorted(value for value in values if value not in MARKERS))
    markers = tuple(sorted(value for value in values if value in MARKERS))
    return atoms, markers


def _literal_transfer_key(transfer: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        _literal_endpoint(transfer.get("source")),
        _literal_endpoint(transfer.get("target")),
        int(transfer.get("electrons", 2)),
        str(transfer.get("kind", "unknown")),
    )


def _literal_exact(predicted: Iterable[Dict[str, Any]], gold: Iterable[Dict[str, Any]]) -> bool:
    return Counter(_literal_transfer_key(item) for item in predicted) == Counter(
        _literal_transfer_key(item) for item in gold
    )


def _match_counts(
    predicted: Iterable[Dict[str, Any]],
    gold: Iterable[Dict[str, Any]],
) -> Tuple[int, int, int, bool]:
    predicted_keys = [_transfer_key(item) for item in predicted]
    gold_keys = [_transfer_key(item) for item in gold]
    remaining = Counter(predicted_keys)
    matched = 0
    for key in gold_keys:
        if remaining[key] > 0:
            remaining[key] -= 1
            matched += 1
    exact = Counter(predicted_keys) == Counter(gold_keys)
    return matched, len(gold_keys), len(predicted_keys), exact


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _load_predicted_actions(bridge: ChemistryBridge, record: Dict[str, Any]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    for action_pgl in record["state"]["environment"].get("history_actions_pgl", []):
        action, parse_result = bridge.parse_action(str(action_pgl))
        if not parse_result.get("valid"):
            raise ValueError(
                f"cannot parse stored action for {record['state']['task']['reaction_id']}: {parse_result}"
            )
        actions.append(action)
    return actions


def evaluate(
    result_path: Path,
    bridge: ChemistryBridge,
    *,
    core_subset: str = "auto",
    split: str = "all",
    core_dir: str = "data/chem/sft2_mechfindermerge_alignschema/core",
    mechfinder_num_classes: int | None = None,
    mechfinder_per_class: int = 20,
) -> Dict[str, Any]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    records = payload.get("results") or []
    reaction_ids = [str(record["state"]["task"]["reaction_id"]) for record in records]
    if core_subset == "auto":
        is_mechfinder = [reaction_id.startswith("mechfinder::") for reaction_id in reaction_ids]
        if is_mechfinder and all(is_mechfinder):
            core_subset = "mechfinder"
        elif not any(is_mechfinder):
            core_subset = "named_val"
        else:
            raise ValueError("cannot auto-detect one GT subset for mixed result sources")
    tasks = bridge.load_tasks(
        data_source="core",
        split=split,
        extracted_dir="data/chem/extracted_paths",
        core_dir=core_dir,
        core_subset=core_subset,
        max_tasks=None,
        reaction_id=None,
        mechfinder_num_classes=mechfinder_num_classes,
        mechfinder_per_class=mechfinder_per_class,
    )
    tasks_by_id = {str(task.mechanism_id): task for task in tasks}

    reaction_rows: List[Dict[str, Any]] = []
    total_exact_steps = 0
    total_literal_exact_steps = 0
    total_aligned_gold_steps = 0
    total_matched = 0
    total_gold_transfers = 0
    total_predicted_transfers = 0

    for record in records:
        reaction_id = str(record["state"]["task"]["reaction_id"])
        task = tasks_by_id.get(reaction_id)
        if task is None:
            raise ValueError(f"result reaction is absent from {core_subset} GT: {reaction_id}")
        predicted_actions = _load_predicted_actions(bridge, record)
        gold_actions: List[Dict[str, Any]] = []
        for item in task.path.sequence:
            if isinstance(item, bridge.ReactionState) or not getattr(item, "electron_transfers", None):
                continue
            gold_action, parse_result = bridge.parse_action(str(item))
            if not parse_result.get("valid"):
                raise ValueError(f"cannot parse GT action for {reaction_id}: {parse_result}")
            gold_actions.append(gold_action)

        step_rows: List[Dict[str, Any]] = []
        exact_steps = 0
        literal_exact_steps = 0
        matched_reaction = 0
        gold_reaction = 0
        predicted_reaction = 0
        aligned_steps = max(len(predicted_actions), len(gold_actions))
        for index in range(aligned_steps):
            predicted = predicted_actions[index] if index < len(predicted_actions) else {}
            gold = gold_actions[index] if index < len(gold_actions) else {}
            matched, gold_count, predicted_count, exact = _match_counts(
                predicted.get("electron_transfers", []),
                gold.get("electron_transfers", []),
            )
            literal_exact = _literal_exact(
                predicted.get("electron_transfers", []),
                gold.get("electron_transfers", []),
            )
            if index < len(gold_actions):
                total_aligned_gold_steps += 1
                if exact:
                    exact_steps += 1
                    total_exact_steps += 1
                if literal_exact:
                    literal_exact_steps += 1
                    total_literal_exact_steps += 1
            matched_reaction += matched
            gold_reaction += gold_count
            predicted_reaction += predicted_count
            step_rows.append(
                {
                    "step": index,
                    "matched_transfers": matched,
                    "gold_transfers": gold_count,
                    "predicted_transfers": predicted_count,
                    "gold_covered": gold_count > 0 and matched == gold_count,
                    "exact_transfer_set": exact,
                    "literal_exact_transfer_set": literal_exact,
                }
            )

        full_path_exact = (
            bool(gold_actions)
            and len(predicted_actions) == len(gold_actions)
            and exact_steps == len(gold_actions)
        )
        full_path_literal_exact = (
            bool(gold_actions)
            and len(predicted_actions) == len(gold_actions)
            and literal_exact_steps == len(gold_actions)
        )
        total_matched += matched_reaction
        total_gold_transfers += gold_reaction
        total_predicted_transfers += predicted_reaction
        reaction_rows.append(
            {
                "reaction_id": reaction_id,
                "success": bool(record["state"]["control"]["success"]),
                "stop_reason": record["state"]["control"]["stop_reason"],
                "predicted_steps": len(predicted_actions),
                "gold_steps": len(gold_actions),
                "exact_steps": exact_steps,
                "literal_exact_steps": literal_exact_steps,
                "matched_transfers": matched_reaction,
                "gold_transfers": gold_reaction,
                "predicted_transfers": predicted_reaction,
                "transfer_recall": _safe_div(matched_reaction, gold_reaction),
                "transfer_precision": _safe_div(matched_reaction, predicted_reaction),
                "full_path_exact": full_path_exact,
                "full_path_literal_exact": full_path_literal_exact,
                "steps": step_rows,
            }
        )

    exact_paths = [row for row in reaction_rows if row["full_path_exact"]]
    literal_exact_paths = [row for row in reaction_rows if row["full_path_literal_exact"]]
    successful = [row for row in reaction_rows if row["success"]]
    successful_exact = [row for row in exact_paths if row["success"]]
    n = len(reaction_rows)
    precision = _safe_div(total_matched, total_predicted_transfers)
    recall = _safe_div(total_matched, total_gold_transfers)
    return {
        "source_result": str(result_path),
        "num_result_records": n,
        "gt_subset": core_subset,
        "num_gt_tasks_loaded": len(tasks),
        "definition": {
            "transfer_match": "same normalized source atom set and target atom set; marker/kind wording ignored",
            "full_path_exact": "same non-empty step count and order, with identical transfer multisets at every step",
            "full_path_literal_exact": "full_path_exact plus normalized endpoint markers, electron count, and kind",
        },
        "aggregate": {
            "success_count": len(successful),
            "success_rate": _safe_div(len(successful), n),
            "full_path_exact_count": len(exact_paths),
            "full_path_exact_rate": _safe_div(len(exact_paths), n),
            "full_path_literal_exact_count": len(literal_exact_paths),
            "full_path_literal_exact_rate": _safe_div(len(literal_exact_paths), n),
            "success_and_full_path_exact_count": len(successful_exact),
            "exact_step_count": total_exact_steps,
            "gold_step_count": total_aligned_gold_steps,
            "exact_step_rate": _safe_div(total_exact_steps, total_aligned_gold_steps),
            "literal_exact_step_count": total_literal_exact_steps,
            "literal_exact_step_rate": _safe_div(
                total_literal_exact_steps, total_aligned_gold_steps
            ),
            "matched_transfers": total_matched,
            "gold_transfers": total_gold_transfers,
            "predicted_transfers": total_predicted_transfers,
            "transfer_recall": recall,
            "transfer_precision": precision,
            "transfer_f1": _safe_div(2 * precision * recall, precision + recall),
        },
        "full_path_exact_reaction_ids": [row["reaction_id"] for row in exact_paths],
        "full_path_literal_exact_reaction_ids": [
            row["reaction_id"] for row in literal_exact_paths
        ],
        "success_reaction_ids": [row["reaction_id"] for row in successful],
        "reactions": reaction_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Planner–Executor actions with offline GT transfers.")
    parser.add_argument("result", type=Path)
    parser.add_argument("--skillrl-root", default=DEFAULT_SKILLRL_ROOT)
    parser.add_argument(
        "--core-subset",
        choices=["auto", "named_train", "named_val", "mechfinder", "all"],
        default="auto",
    )
    parser.add_argument("--split", default="all")
    parser.add_argument("--core-dir", default="data/chem/sft2_mechfindermerge_alignschema/core")
    parser.add_argument("--mechfinder-num-classes", type=int, default=None)
    parser.add_argument("--mechfinder-per-class", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = evaluate(
        args.result,
        ChemistryBridge(args.skillrl_root),
        core_subset=args.core_subset,
        split=args.split,
        core_dir=args.core_dir,
        mechfinder_num_classes=args.mechfinder_num_classes,
        mechfinder_per_class=args.mechfinder_per_class,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
