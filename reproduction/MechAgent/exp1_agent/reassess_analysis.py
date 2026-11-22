from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agents import PosthocGroundTruthEvaluator
from .client import OpenAIChatConfig, OpenAIChatModel
from .schemas import GlobalPlan, MechanismTaskContext


def _accepted_actions(record: Dict[str, Any]) -> List[str]:
    return [
        str(action["action_pgl"])
        for action in record.get("actions", [])
        if action.get("accepted_into_history") and action.get("action_pgl")
    ]


def render_gt_changes(case: Dict[str, Any]) -> List[str]:
    """Render a converted case's per-step state changes for the evaluator.

    The RMR datasets record states, not arrows, so this is the most the
    annotation says about each elementary step.
    """

    def bonds(items: List[List[Any]]) -> str:
        return ", ".join(f"{a}-{b} ({kind})" for a, b, kind in items) or "none"

    rendered: List[str] = []
    for index, step in enumerate(case["gt_step_changes"], start=1):
        orders = ", ".join(
            f"{a}-{b} {before} -> {after}" for a, b, before, after in step["bond_order_changes"]
        ) or "none"
        charges = ", ".join(
            f"{name} {int(before):+d} -> {int(after):+d}"
            for name, before, after in step["charge_changes"]
        ) or "none"
        rendered.append(
            f"GT elementary step {index} (net changes between annotated states; no arrows recorded):\n"
            f"  bonds formed: {bonds(step['bonds_formed'])}\n"
            f"  bonds broken: {bonds(step['bonds_broken'])}\n"
            f"  bond-order changes: {orders}\n"
            f"  formal-charge changes: {charges}"
        )
    return rendered


def _key(record: Dict[str, Any]) -> str:
    reaction = record["reaction"]
    return f"{reaction['reaction_id']}#{reaction.get('repeat_index')}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run only the post-hoc LLM evaluator for compact analysis JSONL."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="gemma4-31b")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--omit-thinking-extra-body", action="store_true")
    parser.add_argument(
        "--gt-changes",
        type=Path,
        action="append",
        default=None,
        help=(
            "Converted clean JSONL whose per-step state changes replace the annotated "
            "arrows as ground truth. Repeatable; records without a matching case are skipped."
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep records already assessed in OUTPUT and append only the rest.",
    )
    args = parser.parse_args()

    cases: Optional[Dict[str, Dict[str, Any]]] = None
    if args.gt_changes:
        cases = {}
        for path in args.gt_changes:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    case = json.loads(line)
                    cases[str(case["case_id"])] = case

    def build_evaluator() -> PosthocGroundTruthEvaluator:
        return PosthocGroundTruthEvaluator(
            OpenAIChatModel(
                OpenAIChatConfig(
                    model=args.model,
                    base_url=args.base_url,
                    api_key=args.api_key or os.environ.get("GT_EVALUATOR_API_KEY"),
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                    enable_thinking=args.enable_thinking,
                    send_thinking_kwargs=not args.omit_thinking_extra_body,
                    timeout=300.0,
                    max_retries=2,
                    trace_path=None,
                )
            )
        )

    records = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    done = set()
    if args.resume and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                previous = json.loads(line)
                if "error" not in ((previous.get("gt") or {}).get("evaluator_assessment") or {"error": 1}):
                    done.add(_key(previous))
        # Rewrite without failed attempts so they are retried exactly once.
        kept = [
            line
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip() and _key(json.loads(line)) in done
        ]
        args.output.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")

    todo = []
    for record in records:
        if _key(record) in done:
            continue
        if cases is not None:
            case_id = str(record["reaction"]["reaction_id"]).split("::")[-1]
            if case_id not in cases:
                continue
        todo.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    progress = {"count": 0}

    def assess(record: Dict[str, Any]) -> None:
        reaction = record["reaction"]
        plan_data = record["planner_plan"]
        task = MechanismTaskContext(
            reaction_id=str(reaction["reaction_id"]),
            task_description=str(reaction["task_description"]),
            initial_state_pgl=str(reaction["initial_state_pgl"]),
            target_state_pgl=str(reaction["target_state_pgl"]),
            conditions=reaction.get("conditions"),
            reaction_metadata={"input_mode": reaction.get("task_input_mode")},
        )
        if cases is not None:
            gold = render_gt_changes(cases[str(reaction["reaction_id"]).split("::")[-1]])
            gt_format = "bond_changes"
        else:
            gold = [str(item) for item in record["gt"].get("actions_pgl", [])]
            gt_format = "arrows"
        try:
            assessment = build_evaluator().assess(
                task=task,
                initial_plan=GlobalPlan.from_dict(
                    plan_data,
                    plan_version=int(plan_data.get("plan_version", 1)),
                ),
                predicted_actions_pgl=_accepted_actions(record),
                gold_actions_pgl=gold,
                rollout_outcome={
                    "success": bool(reaction.get("success")),
                    "stop_reason": reaction.get("stop_reason"),
                    "executed_step_count": len(_accepted_actions(record)),
                },
                gt_format=gt_format,
            ).to_dict()
        except Exception as exc:  # stored so a failure is visible, then retried by --resume
            assessment = {"error": repr(exc)}
        record["gt"]["evaluator_assessment"] = assessment
        record["gt"]["evaluator_gt_format"] = gt_format
        with lock:
            with args.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            progress["count"] += 1
            status = "error" if "error" in assessment else "ok"
            print(
                f"[assessment {progress['count']}/{len(todo)}] {_key(record)} {status}",
                flush=True,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(assess, todo))


if __name__ == "__main__":
    main()
