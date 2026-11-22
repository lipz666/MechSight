"""Score predicted mechanism paths against the annotated ones.

Two input formats are accepted:

- ``compact``: the ``--analysis-output`` JSONL, one rollout per line. Gold
  actions are embedded in the record, so no chemistry data has to be reloaded.
- ``legacy``: the full result JSON, which stores only predicted actions and
  needs the annotated path reloaded through :class:`ChemistryBridge`.

Scoring itself lives in :mod:`develop.metrics`, which reports the annotation's
step boundaries as a separate level rather than folding them into one number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from . import metrics
from .chemistry import DEFAULT_SKILLRL_ROOT, ChemistryBridge

# Re-exported for callers that still import the old normalization helpers.
MARKERS = metrics.MARKERS


class PathRecord:
    """One rollout: the predicted actions and the annotated ones."""

    def __init__(
        self,
        reaction_id: str,
        predicted_actions: Sequence[Any],
        gold_actions: Sequence[Any],
        *,
        repeat_index: int = 0,
        success: bool = False,
        stop_reason: str = "",
    ):
        self.reaction_id = reaction_id
        self.predicted_actions = list(predicted_actions)
        self.gold_actions = list(gold_actions)
        self.repeat_index = repeat_index
        self.success = success
        self.stop_reason = stop_reason


def _parse_action_pgl(text: str) -> Any:
    """Parse one ``ReactionTransformation(...)`` literal.

    Gold actions use bare endpoints (``source='O205'``) while Executor actions
    use structural tuples (``source=('O205', 'lone_pair')``). Both parse into
    the same object shape; :func:`metrics.transfer_key` reconciles the dialects.
    """

    from .standalone_chemistry import parse_pgl_literal

    return parse_pgl_literal(text)


def iter_compact(path: Path) -> Iterator[PathRecord]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            reaction = row["reaction"]
            predicted = [
                _parse_action_pgl(action["action_pgl"])
                for action in row.get("actions", [])
                if action.get("accepted_into_history") and action.get("action_pgl")
            ]
            gold = [_parse_action_pgl(text) for text in row["gt"]["actions_pgl"]]
            yield PathRecord(
                str(reaction["reaction_id"]),
                predicted,
                gold,
                repeat_index=int(reaction.get("repeat_index", 0)),
                success=bool(reaction.get("success", False)),
                stop_reason=str(reaction.get("stop_reason", "")),
            )


def iter_legacy(
    path: Path,
    bridge: ChemistryBridge,
    *,
    core_subset: str = "auto",
    split: str = "all",
    core_dir: str = "data/chem/sft2_mechfindermerge_alignschema/core",
    mechfinder_num_classes: Optional[int] = None,
    mechfinder_per_class: int = 20,
) -> Iterator[PathRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
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

    for record in records:
        reaction_id = str(record["state"]["task"]["reaction_id"])
        task = tasks_by_id.get(reaction_id)
        if task is None:
            raise ValueError(f"result reaction is absent from {core_subset} GT: {reaction_id}")
        predicted = []
        for action_pgl in record["state"]["environment"].get("history_actions_pgl", []):
            action, parse_result = bridge.parse_action(str(action_pgl))
            if not parse_result.get("valid"):
                raise ValueError(f"cannot parse stored action for {reaction_id}: {parse_result}")
            predicted.append(action)
        gold = []
        for item in task.path.sequence:
            if isinstance(item, bridge.ReactionState) or not getattr(item, "electron_transfers", None):
                continue
            gold_action, parse_result = bridge.parse_action(str(item))
            if not parse_result.get("valid"):
                raise ValueError(f"cannot parse GT action for {reaction_id}: {parse_result}")
            gold.append(gold_action)
        yield PathRecord(
            reaction_id,
            predicted,
            gold,
            success=bool(record["state"]["control"]["success"]),
            stop_reason=str(record["state"]["control"]["stop_reason"]),
        )


def score(
    records: Sequence[PathRecord],
    *,
    anonymize_h: bool = True,
    seed: int = 0,
) -> Dict[str, Any]:
    comparisons = [
        metrics.compare_path(
            record.predicted_actions,
            record.gold_actions,
            reaction_id=record.reaction_id,
            repeat_index=record.repeat_index,
            anonymize_h=anonymize_h,
        )
        for record in records
    ]
    report = metrics.aggregate(comparisons, seed=seed)
    report["anonymize_h"] = anonymize_h
    report["definition"] = {
        "L1_transfer_set": "identical transfer multiset over the whole path; step boundaries and emission order ignored",
        "L2_transfer_sequence": "L1 plus identical flattened transfer order",
        "L3_step_partition": "L2 plus identical step count and per-step grouping; equals the legacy full_path_exact",
        "transfer_match": "same normalized source atom set and target atom set; endpoint markers, electron count and kind wording ignored",
        "anonymize_h": "hydrogen atom labels collapsed to H*, because protons are indistinguishable and annotated H labels are generated",
    }
    report["paths"] = [comparison.to_dict() for comparison in comparisons]
    return report


def evaluate(
    result_path: Path,
    bridge: Optional[ChemistryBridge] = None,
    *,
    input_format: str = "auto",
    anonymize_h: bool = True,
    seed: int = 0,
    **legacy_kwargs: Any,
) -> Dict[str, Any]:
    if input_format == "auto":
        input_format = "compact" if result_path.suffix == ".jsonl" else "legacy"
    if input_format == "compact":
        records = list(iter_compact(result_path))
    else:
        if bridge is None:
            bridge = ChemistryBridge(DEFAULT_SKILLRL_ROOT)
        records = list(iter_legacy(result_path, bridge, **legacy_kwargs))
    report = score(records, anonymize_h=anonymize_h, seed=seed)
    report["source_result"] = str(result_path)
    report["input_format"] = input_format
    return report


def compare_h_sensitivity(records: Sequence[PathRecord], *, seed: int = 0) -> Dict[str, Any]:
    """Report each level with and without hydrogen-label anonymization."""

    anonymized = score(records, anonymize_h=True, seed=seed)
    literal = score(records, anonymize_h=False, seed=seed)
    return {
        level: {
            "anonymized": anonymized["levels"][level]["rate"],
            "literal": literal["levels"][level]["rate"],
            "delta": anonymized["levels"][level]["rate"] - literal["levels"][level]["rate"],
        }
        for level in anonymized["levels"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score Planner-Executor paths against annotated mechanisms."
    )
    parser.add_argument("result", type=Path)
    parser.add_argument("--format", choices=["auto", "compact", "legacy"], default="auto")
    parser.add_argument(
        "--literal-h",
        action="store_true",
        help="compare explicit hydrogen labels literally instead of anonymizing them",
    )
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument("--json", action="store_true", help="print the full JSON report")
    args = parser.parse_args()

    input_format = args.format
    if input_format == "auto":
        input_format = "compact" if args.result.suffix == ".jsonl" else "legacy"
    if input_format == "compact":
        records = list(iter_compact(args.result))
    else:
        records = list(
            iter_legacy(
                args.result,
                ChemistryBridge(args.skillrl_root),
                core_subset=args.core_subset,
                split=args.split,
                core_dir=args.core_dir,
                mechfinder_num_classes=args.mechfinder_num_classes,
                mechfinder_per_class=args.mechfinder_per_class,
            )
        )

    report = score(records, anonymize_h=not args.literal_h, seed=args.seed)
    report["source_result"] = str(args.result)
    report["input_format"] = input_format
    report["h_sensitivity"] = compare_h_sensitivity(records, seed=args.seed)

    print(metrics.render_report(report))
    print("\nH-label sensitivity (anonymized - literal):")
    for level, values in report["h_sensitivity"].items():
        print(
            f"  {level:22s} {values['anonymized']:7.1%} vs {values['literal']:7.1%}"
            f"   ({values['delta']:+.1%})"
        )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
