from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar

from .parsers import parse_executor_output, parse_json_tag
from .prompts import (
    completion_check_messages,
    create_plan_messages,
    directive_messages,
    executor_messages,
    posthoc_gt_assessment_messages,
    planner_reaction_feature_messages,
    planner_skill_evidence_repair_messages,
    planner_skill_resolution_messages,
    planner_skill_selection_messages,
    planner_skill_verification_messages,
    replan_messages,
    review_messages,
)
from .planner_skills import (
    PlannerSkillLibrary,
    audit_skill_verification_evidence,
    normalize_reaction_features,
    normalize_skill_resolution,
    normalize_skill_selection,
    normalize_skill_verification,
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
    def __init__(
        self,
        model: ChatCompleter,
        *,
        skill_library: Optional[PlannerSkillLibrary] = None,
        max_planner_skills: int = 2,
        skill_catalog_batch_size: int = 6,
    ):
        self.model = model
        self.skill_library = skill_library
        self.max_planner_skills = max_planner_skills
        self.skill_catalog_batch_size = max(1, skill_catalog_batch_size)

    def _select_skills(self, task: MechanismTaskContext) -> Dict[str, Any]:
        if self.skill_library is None:
            return {}
        reaction_features = _complete_parsed(
            self.model,
            planner_reaction_feature_messages(task),
            phase="planner.extract_reaction_features",
            requirement=(
                "Output exactly one <reaction_features>valid JSON"
                "</reaction_features> block."
            ),
            parser=lambda answer: normalize_reaction_features(
                parse_json_tag(answer, "reaction_features")
            ),
        )
        if hasattr(self.skill_library, "filter_candidates_l0"):
            catalog = self.skill_library.filter_candidates_l0(
                task.initial_state_pgl,
                task.target_state_pgl,
            )
        else:
            catalog = self.skill_library.catalog()
        catalog_batches = [
            catalog[index : index + self.skill_catalog_batch_size]
            for index in range(0, len(catalog), self.skill_catalog_batch_size)
        ]
        batch_results: List[Dict[str, Any]] = []
        candidates: List[Dict[str, str]] = []
        candidate_names_seen = set()
        for batch_index, catalog_batch in enumerate(catalog_batches, start=1):
            allowed_names = [item["name"] for item in catalog_batch]
            phase = (
                "planner.select_skills"
                if len(catalog_batches) == 1
                else f"planner.select_skills.batch_{batch_index}"
            )
            batch_selection = _complete_parsed(
                self.model,
                planner_skill_selection_messages(
                    task,
                    catalog_batch,
                    self.max_planner_skills,
                    None,
                ),
                phase=phase,
                requirement=(
                    "Output exactly one <skill_selection>valid JSON"
                    "</skill_selection> block."
                ),
                parser=lambda answer, names=allowed_names: normalize_skill_selection(
                    parse_json_tag(answer, "skill_selection"),
                    allowed_names=names,
                    max_skills=self.max_planner_skills,
                ),
            )
            batch_results.append(batch_selection)
            for item in batch_selection.get("selected_skills", []):
                if item["name"] not in candidate_names_seen:
                    candidates.append(item)
                    candidate_names_seen.add(item["name"])

        selection = {
            "reaction_features": reaction_features,
            "inferred_reaction_pattern": reaction_features[
                "reaction_center_summary"
            ],
            "structural_evidence": list(
                reaction_features.get("distinguishing_evidence", [])
            ),
            "selected_skills": list(candidates),
            "no_match_reason": "",
            "retrieval_batches": batch_results,
        }
        if not candidates:
            selection["candidate_skills"] = []
            selection["trigger_verification"] = []
            selection["no_match_reason"] = (
                "No catalog batch returned a structure-supported candidate."
            )
            return selection

        candidate_names = [item["name"] for item in candidates]
        catalog_by_name = {
            item["name"]: item for item in self.skill_library.catalog()
        }
        candidate_catalog = [catalog_by_name[name] for name in candidate_names]
        verification_messages = planner_skill_verification_messages(
            task,
            candidate_catalog,
            selection,
        )
        raw_checks = _complete_parsed(
            self.model,
            verification_messages,
            phase="planner.verify_skill_triggers",
            requirement=(
                "Output exactly one <skill_verification>valid JSON"
                "</skill_verification> block."
            ),
            parser=lambda answer: normalize_skill_verification(
                parse_json_tag(answer, "skill_verification"),
                candidate_names=candidate_names,
            ),
        )
        checks = audit_skill_verification_evidence(
            raw_checks,
            initial_pgl=task.initial_state_pgl,
            target_pgl=task.target_state_pgl,
        )
        invalidated_positive = any(
            before["trigger_satisfied"] and not after["trigger_satisfied"]
            for before, after in zip(raw_checks, checks)
        )
        if invalidated_positive:
            repaired_checks = _complete_parsed(
                self.model,
                planner_skill_evidence_repair_messages(
                    verification_messages,
                    checks,
                ),
                phase="planner.repair_skill_evidence",
                requirement=(
                    "Output exactly one <skill_verification>valid JSON"
                    "</skill_verification> block."
                ),
                parser=lambda answer: normalize_skill_verification(
                    parse_json_tag(answer, "skill_verification"),
                    candidate_names=candidate_names,
                ),
            )
            checks = audit_skill_verification_evidence(
                repaired_checks,
                initial_pgl=task.initial_state_pgl,
                target_pgl=task.target_state_pgl,
            )
        accepted = {
            item["name"] for item in checks if item["trigger_satisfied"]
        }
        selection["candidate_skills"] = candidates
        accepted_items = [
            item for item in candidates if item["name"] in accepted
        ]
        selection["skill_resolution"] = None
        if len(accepted_items) > 1:
            accepted_catalog = [
                catalog_by_name[item["name"]] for item in accepted_items
            ]
            resolution = _complete_parsed(
                self.model,
                planner_skill_resolution_messages(
                    reaction_features,
                    accepted_catalog,
                    [item for item in checks if item["trigger_satisfied"]],
                    self.max_planner_skills,
                ),
                phase="planner.resolve_skill_redundancy",
                requirement=(
                    "Output exactly one <skill_resolution>valid JSON"
                    "</skill_resolution> block."
                ),
                parser=lambda answer: normalize_skill_resolution(
                    parse_json_tag(answer, "skill_resolution"),
                    accepted_names=[item["name"] for item in accepted_items],
                    max_skills=self.max_planner_skills,
                ),
            )
            retained = set(resolution["selected_skill_names"])
            accepted_items = [
                item for item in accepted_items if item["name"] in retained
            ]
            selection["skill_resolution"] = resolution
        selection["selected_skills"] = accepted_items[: self.max_planner_skills]
        selection["trigger_verification"] = checks
        if not selection["selected_skills"]:
            selection["no_match_reason"] = (
                "All retrieved candidates failed center-aligned trigger verification."
            )
        return selection

    def _skill_content(self, selection: Dict[str, Any]) -> List[Dict[str, str]]:
        if self.skill_library is None:
            return []
        names = [item["name"] for item in selection.get("selected_skills", [])]
        return self.skill_library.selected_content(names)

    def create_plan(self, task: MechanismTaskContext) -> GlobalPlan:
        selection = self._select_skills(task)
        plan = _complete_parsed(
            self.model,
            create_plan_messages(task, self._skill_content(selection)),
            phase="planner.create_plan",
            requirement="Output exactly one <plan>valid JSON</plan> block.",
            parser=lambda answer: GlobalPlan.from_dict(
                parse_json_tag(answer, "plan"), plan_version=1
            ),
        )
        plan.planner_skill_selection = selection
        return plan

    def replan(
        self,
        *,
        task: MechanismTaskContext,
        plan: GlobalPlan,
        current_state_pgl: str,
        history_actions_pgl: List[str],
        reason: Any,
    ) -> GlobalPlan:
        new_version = plan.plan_version + 1

        def parse_replan(answer: str) -> GlobalPlan:
            data = parse_json_tag(answer, "plan")
            data["replanned_from_version"] = plan.plan_version
            data["replan_reason"] = str(reason)
            return GlobalPlan.from_dict(data, plan_version=new_version)

        selection = dict(plan.planner_skill_selection)
        new_plan = _complete_parsed(
            self.model,
            replan_messages(
                task,
                plan,
                current_state_pgl,
                history_actions_pgl,
                reason,
                self._skill_content(selection),
            ),
            phase="planner.replan",
            requirement="Output exactly one <plan>valid JSON</plan> block.",
            parser=parse_replan,
        )
        new_plan.planner_skill_selection = selection
        return new_plan

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
    ) -> StepDirective:
        messages = directive_messages(
            task,
            plan,
            active_elementary_step,
            mechanism_step_idx,
            current_state_pgl,
            history_actions_pgl,
            last_env_feedback,
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
    ) -> CompletionReview:
        messages = completion_check_messages(
            task,
            plan,
            mechanism_step_idx,
            current_state_pgl,
            history_actions_pgl,
            strict_product_match=strict_product_match,
            plan_exhausted=plan_exhausted,
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
        gt_format: str = "arrows",
    ) -> GroundTruthAssessment:
        messages = posthoc_gt_assessment_messages(
            task,
            initial_plan,
            predicted_actions_pgl,
            gold_actions_pgl,
            rollout_outcome,
            gt_format=gt_format,
        )
        assessment = _complete_parsed(
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
        if {
            assessment.planner_elementary_steps_judgment,
            assessment.predicted_events_judgment,
        } == {"correct"}:
            return assessment

        audit_messages = list(messages) + [
            {
                "role": "assistant",
                "content": (
                    "<gt_assessment>"
                    + json.dumps(assessment.to_dict(), ensure_ascii=False)
                    + "</gt_assessment>"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Perform a decision-consistency audit of your assessment. Apply the "
                    "mandatory normalization rules literally. If a partial or incorrect "
                    "judgment cites only concerted-versus-stepwise decomposition, a reasonable "
                    "short-lived intermediate, adjacent action splitting/merging, resonance "
                    "restoration moved between adjacent blocks, alternative Kekule/pi-bond "
                    "localization within the same delocalized system while the actual reacting "
                    "atom is unchanged, H3O+ versus H2O + H+, an "
                    "explicit proton acceptor omitted by GT, or an interchangeable transient-H "
                    "label, that judgment must become correct. Compare cumulative electron-flow "
                    "and state changes before treating an explicit intermediate as an extra event. "
                    "This is especially mandatory when the "
                    "Planner steps with the same event decomposition were judged correct. Keep "
                    "partially_correct or incorrect only if you can name at least one real "
                    "missing, extra, duplicated, reordered net event, non-equivalent chemical "
                    "site, persistent divergent intermediate, or unmatched catalyst/proton "
                    "balance after normalization. Return one complete replacement "
                    "<gt_assessment> block."
                ),
            },
        ]
        return _complete_parsed(
            self.model,
            audit_messages,
            phase="posthoc.gt_assessment.normalization_audit",
            requirement=(
                "Output exactly one replacement <gt_assessment>valid JSON</gt_assessment> block."
            ),
            parser=lambda answer: GroundTruthAssessment.from_dict(
                parse_json_tag(answer, "gt_assessment")
            ),
        )
