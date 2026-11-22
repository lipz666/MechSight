from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from .agents import ExecutorAgent, PlannerAgent, PosthocGroundTruthEvaluator
from .chemistry import DEFAULT_SKILLRL_ROOT, ChemistryBridge
from .client import OpenAIChatConfig, OpenAIChatModel
from .orchestrator import PlannerExecutorAgent, PlannerExecutorConfig
from .schemas import GlobalPlan, PlannerExecutorResult


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_SERVED_MODEL = "gemma4-31b"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Planner–Executor PGL mechanism agent.")
    parser.add_argument("--skillrl-root", default=DEFAULT_SKILLRL_ROOT)
    parser.add_argument("--reaction-id", default=None, help="Exact id or substring; comma-separated values are accepted.")
    parser.add_argument("--data-source", choices=["extracted_paths", "core"], default="core")
    parser.add_argument("--split", default="val")
    parser.add_argument("--extracted-dir", default="data/chem/extracted_paths")
    parser.add_argument("--core-dir", default="data/chem/sft2_mechfindermerge_alignschema/core")
    parser.add_argument("--core-subset", choices=["named_train", "named_val", "mechfinder", "all"], default="named_val")
    parser.add_argument("--mechfinder-num-classes", type=int, default=None)
    parser.add_argument("--mechfinder-per-class", type=int, default=20)
    parser.add_argument(
        "--task-input-mode",
        choices=["metadata", "structures_only"],
        default="structures_only",
        help=(
            "structures_only (default) hides reaction name/class/conditions and asks "
            "the Planner to infer the family from initial/target PGL; metadata exposes "
            "the annotated reaction information for ablation only."
        ),
    )
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument(
        "--repeats-per-task",
        type=int,
        default=1,
        help="Independent rollout repetitions for each selected task.",
    )

    parser.add_argument("--planner-model", default=DEFAULT_SERVED_MODEL)
    parser.add_argument("--executor-model", default=DEFAULT_SERVED_MODEL)
    parser.add_argument("--planner-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--executor-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--planner-api-key", default=None)
    parser.add_argument("--executor-api-key", default=None)
    parser.add_argument("--planner-temperature", type=float, default=0.2)
    parser.add_argument("--executor-temperature", type=float, default=0.2)
    parser.add_argument("--planner-max-tokens", type=int, default=8192)
    parser.add_argument("--executor-max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--api-retries", type=int, default=2)
    parser.add_argument("--disable-planner-thinking", action="store_true")
    parser.add_argument("--disable-executor-thinking", action="store_true")
    parser.add_argument(
        "--omit-thinking-extra-body",
        action="store_true",
        help="Do not send vLLM chat_template_kwargs; useful for closed providers that reject unknown fields.",
    )
    parser.add_argument("--planner-trace", default=None)
    parser.add_argument("--executor-trace", default=None)
    parser.add_argument("--gt-evaluator-model", default=None)
    parser.add_argument("--gt-evaluator-base-url", default=None)
    parser.add_argument("--gt-evaluator-api-key", default=None)
    parser.add_argument("--gt-evaluator-temperature", type=float, default=0.0)
    parser.add_argument("--gt-evaluator-max-tokens", type=int, default=4096)
    parser.add_argument("--disable-gt-evaluator-thinking", action="store_true")
    parser.add_argument("--gt-evaluator-trace", default=None)
    parser.add_argument(
        "--disable-gt-assessment",
        action="store_true",
        help="Skip the one-call post-hoc comparison with annotated GT actions.",
    )

    parser.add_argument(
        "--oracle-step-count",
        action="store_true",
        help=(
            "Ablation only. Tell the Planner how many elementary steps the annotated "
            "path uses, to separate guessing the step-count convention from doing "
            "chemistry. Never enable this for a reported capability run."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-env-calls", type=int, default=12)
    parser.add_argument("--max-executor-format-repairs", type=int, default=2)
    parser.add_argument("--max-planner-review-rounds", type=int, default=3)
    parser.add_argument("--max-replans-per-env-step", type=int, default=1)
    parser.add_argument("--max-total-replans", type=int, default=3)
    parser.add_argument("--allow-repeated-state", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--analysis-output",
        default=None,
        help=(
            "Append one compact JSONL record per completed rollout containing only "
            "the reaction boundary, GT actions, initial Planner plan, and Executor "
            "actions. The GT field also contains the optional post-hoc evaluator "
            "assessment."
        ),
    )
    return parser


def _initial_plan_data(result: PlannerExecutorResult) -> Dict[str, Any]:
    return dict(
        result.plan_versions[0]
        if result.plan_versions
        else result.state.plan.to_dict()
    )


def _analysis_actions(result: PlannerExecutorResult) -> List[Dict[str, Any]]:
    reviews_by_proposal = {
        str(review["proposal_id"]): review
        for review in result.planner_reviews
        if review.get("proposal_id")
    }
    observations_by_review = {
        str(observation["approved_review_id"]): observation
        for observation in result.execution_observations
        if observation.get("approved_review_id")
    }
    actions: List[Dict[str, Any]] = []
    for proposal in result.executor_proposals:
        review = reviews_by_proposal.get(str(proposal.get("proposal_id")))
        observation = (
            observations_by_review.get(str(review.get("review_id")))
            if review is not None
            else None
        )
        accepted = bool(
            observation
            and observation.get("edit_valid")
            and not observation.get("repeated_state")
        )
        actions.append(
            {
                "proposal_id": proposal.get("proposal_id"),
                "directive_id": proposal.get("directive_id"),
                "review_round": proposal.get("review_round"),
                "event_summary": proposal.get("event_summary"),
                "plan_objection": proposal.get("plan_objection"),
                "action_pgl": proposal.get("action_pgl"),
                "validation_result": proposal.get("validation_result"),
                "planner_decision": review.get("decision") if review else None,
                "planner_assessment": (
                    review.get("chemical_assessment") if review else None
                ),
                "executed_in_environment": observation is not None,
                "edit_valid": (
                    observation.get("edit_valid") if observation else None
                ),
                "accepted_into_history": accepted,
            }
        )
    return actions


def build_analysis_record(
    *,
    result: PlannerExecutorResult,
    source_reaction_id: str,
    task_index: int,
    repeat_index: int,
    repeats_per_task: int,
    task_input_mode: str,
    gold_actions_pgl: List[str],
    evaluator_assessment: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Build the compact, trajectory-level record used for mechanism analysis."""

    task = result.state.task
    return {
        "reaction": {
            "reaction_id": source_reaction_id,
            "task_index": task_index,
            "repeat_index": repeat_index,
            "repeats_per_task": repeats_per_task,
            "task_input_mode": task_input_mode,
            "task_description": task.task_description,
            "conditions": task.conditions,
            "initial_state_pgl": task.initial_state_pgl,
            "target_state_pgl": task.target_state_pgl,
            "success": result.state.control.success,
            "stop_reason": result.state.control.stop_reason,
        },
        "gt": {
            "actions_pgl": list(gold_actions_pgl),
            "evaluator_assessment": evaluator_assessment,
        },
        "planner_plan": _initial_plan_data(result),
        "actions": _analysis_actions(result),
    }


def _append_jsonl(path_text: str, record: Dict[str, Any]) -> None:
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _model(
    *,
    model: str,
    base_url: str,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
    send_thinking_kwargs: bool,
    timeout: float,
    max_retries: int,
    trace_path: str | None,
) -> OpenAIChatModel:
    return OpenAIChatModel(
        OpenAIChatConfig(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            send_thinking_kwargs=send_thinking_kwargs,
            timeout=timeout,
            max_retries=max_retries,
            trace_path=trace_path,
        )
    )


def main() -> None:
    args = build_parser().parse_args()
    bridge = ChemistryBridge(args.skillrl_root)
    tasks = bridge.load_tasks(
        data_source=args.data_source,
        split=args.split,
        extracted_dir=args.extracted_dir,
        core_dir=args.core_dir,
        core_subset=args.core_subset,
        max_tasks=args.max_tasks,
        reaction_id=args.reaction_id,
        mechfinder_num_classes=args.mechfinder_num_classes,
        mechfinder_per_class=args.mechfinder_per_class,
    )
    if not tasks:
        raise ValueError("no matching mechanism tasks found")

    planner_model = _model(
        model=args.planner_model,
        base_url=args.planner_base_url,
        api_key=args.planner_api_key or os.environ.get("PLANNER_API_KEY"),
        temperature=args.planner_temperature,
        max_tokens=args.planner_max_tokens,
        enable_thinking=not args.disable_planner_thinking,
        send_thinking_kwargs=not args.omit_thinking_extra_body,
        timeout=args.timeout,
        max_retries=args.api_retries,
        trace_path=args.planner_trace,
    )
    executor_model = _model(
        model=args.executor_model,
        base_url=args.executor_base_url,
        api_key=args.executor_api_key or os.environ.get("EXECUTOR_API_KEY"),
        temperature=args.executor_temperature,
        max_tokens=args.executor_max_tokens,
        enable_thinking=not args.disable_executor_thinking,
        send_thinking_kwargs=not args.omit_thinking_extra_body,
        timeout=args.timeout,
        max_retries=args.api_retries,
        trace_path=args.executor_trace,
    )
    planner = PlannerAgent(planner_model)
    executor = ExecutorAgent(executor_model)
    gt_evaluator = None
    if not args.disable_gt_assessment:
        gt_evaluator_model = _model(
            model=args.gt_evaluator_model or args.planner_model,
            base_url=args.gt_evaluator_base_url or args.planner_base_url,
            api_key=(
                args.gt_evaluator_api_key
                or os.environ.get("GT_EVALUATOR_API_KEY")
                or args.planner_api_key
                or os.environ.get("PLANNER_API_KEY")
            ),
            temperature=args.gt_evaluator_temperature,
            max_tokens=args.gt_evaluator_max_tokens,
            enable_thinking=not args.disable_gt_evaluator_thinking,
            send_thinking_kwargs=not args.omit_thinking_extra_body,
            timeout=args.timeout,
            max_retries=args.api_retries,
            trace_path=args.gt_evaluator_trace,
        )
        gt_evaluator = PosthocGroundTruthEvaluator(gt_evaluator_model)
    config = PlannerExecutorConfig(
        max_steps=args.max_steps,
        max_env_calls=args.max_env_calls,
        max_executor_format_repairs=args.max_executor_format_repairs,
        max_planner_review_rounds=args.max_planner_review_rounds,
        max_replans_per_env_step=args.max_replans_per_env_step,
        max_total_replans=args.max_total_replans,
        stop_on_repeated_state=not args.allow_repeated_state,
    )

    if args.repeats_per_task < 1:
        raise ValueError("--repeats-per-task must be at least 1")
    selected_tasks = tasks[: args.num_runs]
    total_rollouts = len(selected_tasks) * args.repeats_per_task
    results: List[Dict[str, Any]] = []
    rollout_index = 0
    completed_rollouts = 0
    success_count = 0
    gt_assessment_count = 0
    for task_index, task in enumerate(selected_tasks, start=1):
        for repeat_index in range(1, args.repeats_per_task + 1):
            rollout_index += 1
            print(
                f"[mechanism {rollout_index}/{total_rollouts}] {task.mechanism_id} "
                f"(repeat {repeat_index}/{args.repeats_per_task})",
                flush=True,
            )
            env = bridge.make_env(task, max_env_calls=args.max_env_calls)
            runtime = PlannerExecutorAgent(
                env=env,
                planner=planner,
                executor=executor,
                state_to_pgl=bridge.state_to_pgl,
                parse_action=bridge.parse_action,
                validate_action=bridge.validate_action,
                config=config,
                state_signature=bridge.strict_state_signature,
            )
            result = runtime.run(
                bridge.task_context(
                    task,
                    input_mode=args.task_input_mode,
                    oracle_step_count=args.oracle_step_count,
                )
            )
            record = result.to_dict()
            record["run_metadata"] = {
                "task_index": task_index,
                "repeat_index": repeat_index,
                "repeats_per_task": args.repeats_per_task,
                "source_reaction_id": str(task.mechanism_id),
                "task_input_mode": args.task_input_mode,
            }
            gold_actions_pgl = (
                bridge.gold_actions_pgl(task)
                if gt_evaluator is not None or args.analysis_output
                else []
            )
            initial_plan_data = _initial_plan_data(result)
            if gt_evaluator is not None:
                try:
                    initial_plan = GlobalPlan.from_dict(
                        initial_plan_data,
                        plan_version=int(initial_plan_data.get("plan_version", 1)),
                    )
                    assessment = gt_evaluator.assess(
                        task=result.state.task,
                        initial_plan=initial_plan,
                        predicted_actions_pgl=list(
                            result.state.environment.history_actions_pgl
                        ),
                        gold_actions_pgl=gold_actions_pgl,
                        rollout_outcome={
                            "success": result.state.control.success,
                            "stop_reason": result.state.control.stop_reason,
                            "executed_step_count": len(
                                result.state.environment.history_actions_pgl
                            ),
                        },
                    )
                    record["gt_assessment"] = assessment.to_dict()
                    gt_assessment_count += 1
                except Exception as exc:
                    record["gt_assessment"] = {"error": repr(exc)}
            if args.analysis_output:
                _append_jsonl(
                    args.analysis_output,
                    build_analysis_record(
                        result=result,
                        source_reaction_id=str(task.mechanism_id),
                        task_index=task_index,
                        repeat_index=repeat_index,
                        repeats_per_task=args.repeats_per_task,
                        task_input_mode=args.task_input_mode,
                        gold_actions_pgl=gold_actions_pgl,
                        evaluator_assessment=record.get("gt_assessment"),
                    ),
                )
            completed_rollouts += 1
            success_count += int(result.state.control.success)
            if args.output or not args.analysis_output:
                results.append(record)

    payload = {"num_runs": completed_rollouts, "results": results}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        print(
            json.dumps(
                {
                    "num_runs": completed_rollouts,
                    "output": str(path),
                    "analysis_output": args.analysis_output,
                    "success_count": success_count,
                    "gt_assessment_count": gt_assessment_count,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.analysis_output:
        print(
            json.dumps(
                {
                    "num_runs": completed_rollouts,
                    "analysis_output": args.analysis_output,
                    "success_count": success_count,
                    "gt_assessment_count": gt_assessment_count,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(rendered)


if __name__ == "__main__":
    main()
