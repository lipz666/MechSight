from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar

from .parsers import parse_executor_output, parse_json_tag
from .prompts import (
    completion_check_messages,
    create_plan_messages,
    directive_messages,
    executor_messages,
    posthoc_gt_assessment_messages,
    replan_messages,
    review_messages,
)
from .schemas import (
    CompletionReview,
    ExecutorProposal,
    GlobalPlan,
    GroundTruthAssessment,
    MechanismTaskContext,
    PlannerReview,
    StepDirective,
)


class ChatCompleter(Protocol):
    events: List[Dict[str, Any]]

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        phase: str,
        enable_thinking: Optional[bool] = None,
    ) -> Any:
        ...


ParsedT = TypeVar("ParsedT")


def _retry_messages(messages: List[Dict[str, str]], requirement: str) -> List[Dict[str, str]]:
    repaired = [dict(message) for message in messages]
    repaired[-1]["content"] += (
        "\n\n<critical_output_requirement>\n"
        "Return the required structured output immediately. Do not include analysis, "
        "a preamble, markdown, or repeated structure comparison. "
        f"{requirement}\n"
        "</critical_output_requirement>"
    )
    return repaired


def _complete_parsed(
    model: ChatCompleter,
    messages: List[Dict[str, str]],
    *,
    phase: str,
    requirement: str,
    parser: Callable[[str], ParsedT],
) -> ParsedT:
    response = model.complete(messages, phase=phase)
    try:
        return parser(response.answer)
    except (TypeError, ValueError) as exc:
        if model.events:
            model.events[-1]["protocol_error"] = repr(exc)
        retry = model.complete(
            _retry_messages(messages, requirement),
            phase=f"{phase}.retry_without_thinking",
            enable_thinking=False,
        )
        return parser(retry.answer)


class PlannerAgent:
    def __init__(self, model: ChatCompleter):
        self.model = model

    def create_plan(
        self,
        task: MechanismTaskContext,
        *,
        required_changes: Optional[str] = None,
    ) -> GlobalPlan:
        return _complete_parsed(
            self.model,
            create_plan_messages(task, required_changes),
            phase="planner.create_plan",
            requirement="Output exactly one <plan>valid JSON</plan> block.",
            parser=lambda answer: GlobalPlan.from_dict(
                parse_json_tag(answer, "plan"), plan_version=1
            ),
        )

    def replan(
        self,
        *,
        task: MechanismTaskContext,
        plan: GlobalPlan,
        current_state_pgl: str,
        history_actions_pgl: List[str],
        reason: Any,
        remaining_changes: Optional[str] = None,
    ) -> GlobalPlan:
        new_version = plan.plan_version + 1

        def parse_replan(answer: str) -> GlobalPlan:
            data = parse_json_tag(answer, "plan")
            data["replanned_from_version"] = plan.plan_version
            data["replan_reason"] = str(reason)
            return GlobalPlan.from_dict(data, plan_version=new_version)

        return _complete_parsed(
            self.model,
            replan_messages(
                task, plan, current_state_pgl, history_actions_pgl, reason, remaining_changes
            ),
            phase="planner.replan",
            requirement="Output exactly one <plan>valid JSON</plan> block.",
            parser=parse_replan,
        )

    def make_directive(
        self,
        *,
        task: MechanismTaskContext,
        plan: GlobalPlan,
        active_elementary_step: Dict[str, str],
        mechanism_step_idx: int,
        current_state_pgl: str,
        history_actions_pgl: List[str],
        last_env_feedback: Optional[Dict[str, Any]],
        directive_id: str,
        remaining_changes: Optional[str] = None,
    ) -> StepDirective:
        messages = directive_messages(
            task,
            plan,
            active_elementary_step,
            mechanism_step_idx,
            current_state_pgl,
            history_actions_pgl,
            last_env_feedback,
            remaining_changes,
        )
        return _complete_parsed(
            self.model,
            messages,
            phase="planner.make_directive",
            requirement="Output exactly one <directive>valid JSON</directive> block.",
            parser=lambda answer: StepDirective.from_dict(
                parse_json_tag(answer, "directive"),
                plan=plan,
                mechanism_step_idx=mechanism_step_idx,
                directive_id=directive_id,
                expected_elementary_step=active_elementary_step,
            ),
        )

    def review(
        self,
        *,
        task: MechanismTaskContext,
        plan: GlobalPlan,
        directive: StepDirective,
        current_state_pgl: str,
        history_actions_pgl: List[str],
        proposal: ExecutorProposal,
        deterministic_validation: Dict[str, Any],
        previous_reviews: List[Dict[str, Any]],
        review_id: str,
        remaining_changes: Optional[str] = None,
        action_evidence: Optional[Dict[str, Any]] = None,
    ) -> PlannerReview:
        messages = review_messages(
            task,
            plan,
            directive,
            current_state_pgl,
            history_actions_pgl,
            proposal,
            deterministic_validation,
            previous_reviews,
            remaining_changes,
            action_evidence,
        )
        return _complete_parsed(
            self.model,
            messages,
            phase="planner.review",
            requirement="Output exactly one <review>valid JSON</review> block.",
            parser=lambda answer: PlannerReview.from_dict(
                parse_json_tag(answer, "review"),
                proposal_id=proposal.proposal_id,
                review_id=review_id,
            ),
        )

    def check_completion(
        self,
        *,
        task: MechanismTaskContext,
        plan: GlobalPlan,
        mechanism_step_idx: int,
        current_state_pgl: str,
        history_actions_pgl: List[str],
        check_id: str,
        strict_product_match: bool,
        plan_exhausted: bool,
        remaining_changes: Optional[str] = None,
        boundary_gate: Optional[Dict[str, Any]] = None,
    ) -> CompletionReview:
        messages = completion_check_messages(
            task,
            plan,
            mechanism_step_idx,
            current_state_pgl,
            history_actions_pgl,
            strict_product_match=strict_product_match,
            plan_exhausted=plan_exhausted,
            remaining_changes=remaining_changes,
            boundary_gate=boundary_gate,
        )
        return _complete_parsed(
            self.model,
            messages,
            phase="planner.check_completion",
            requirement=(
                "Output exactly one <completion_review>valid JSON</completion_review> block."
            ),
            parser=lambda answer: CompletionReview.from_dict(
                parse_json_tag(answer, "completion_review"),
                check_id=check_id,
                mechanism_step_idx=mechanism_step_idx,
            ),
        )


class ExecutorAgent:
    def __init__(self, model: ChatCompleter):
        self.model = model

    def propose(
        self,
        *,
        task: MechanismTaskContext,
        plan: GlobalPlan,
        directive: StepDirective,
        current_state_pgl: str,
        history_actions_pgl: List[str],
        legal_action_shapes: List[Dict[str, Any]],
        planner_feedback: Optional[Dict[str, Any]],
        proposal_id: str,
        review_round: int,
    ) -> ExecutorProposal:
        messages = executor_messages(
            task,
            plan,
            directive,
            current_state_pgl,
            history_actions_pgl,
            legal_action_shapes,
            planner_feedback,
        )
        return _complete_parsed(
            self.model,
            messages,
            phase="executor.propose",
            requirement=(
                "Output exactly one <event_summary>, one <plan_objection>, and one "
                "<action> block in that order."
            ),
            parser=lambda answer: parse_executor_output(
                answer,
                proposal_id=proposal_id,
                directive_id=directive.directive_id,
                review_round=review_round,
            ),
        )


class PosthocGroundTruthEvaluator:
    """Read-only evaluator called after the Planner-Executor rollout ends."""

    def __init__(self, model: ChatCompleter):
        self.model = model

    def assess(
        self,
        *,
        task: MechanismTaskContext,
        initial_plan: GlobalPlan,
        predicted_actions_pgl: List[str],
        gold_actions_pgl: List[str],
        rollout_outcome: Dict[str, Any],
    ) -> GroundTruthAssessment:
        messages = posthoc_gt_assessment_messages(
            task,
            initial_plan,
            predicted_actions_pgl,
            gold_actions_pgl,
            rollout_outcome,
        )
        return _complete_parsed(
            self.model,
            messages,
            phase="posthoc.gt_assessment",
            requirement=(
                "Output exactly one <gt_assessment>valid JSON</gt_assessment> block."
            ),
            parser=lambda answer: GroundTruthAssessment.from_dict(
                parse_json_tag(answer, "gt_assessment")
            ),
        )
