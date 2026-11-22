from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .state_diff import (
    StructuralDiff,
    boundary_completion_allowed,
    diff_signatures,
    proposal_relevance,
    render_diff,
)
from .schemas import (
    ControlState,
    EnvironmentState,
    ExecutionObservation,
    GlobalPlan,
    MechanismTaskContext,
    PlannerExecutorResult,
    PlannerExecutorState,
    TurnState,
)


@dataclass
class PlannerExecutorConfig:
    max_steps: int = 12
    max_env_calls: int = 12
    max_executor_format_repairs: int = 2
    max_planner_review_rounds: int = 3
    max_replans_per_env_step: int = 1
    max_total_replans: int = 3
    stop_on_repeated_state: bool = True


class PlannerExecutorAgent:
    """Global-plan/local-action loop with pre-execution Planner review."""

    def __init__(
        self,
        *,
        env: Any,
        planner: Any,
        executor: Any,
        state_to_pgl: Callable[[Any], str],
        parse_action: Callable[[str], Tuple[Dict[str, Any], Dict[str, Any]]],
        validate_action: Callable[[Dict[str, Any], Any, List[Dict[str, Any]]], Dict[str, Any]],
        config: Optional[PlannerExecutorConfig] = None,
        state_signature: Optional[Callable[[Any], Dict[str, Any]]] = None,
    ):
        self.env = env
        self.planner = planner
        self.executor = executor
        self.state_to_pgl = state_to_pgl
        self.parse_action = parse_action
        self.validate_action = validate_action
        self.config = config or PlannerExecutorConfig()
        # Optional because the orchestrator tests drive a stub environment.
        # When supplied, the runtime can diff the current and target graphs and
        # stop relying on the Planner to notice missing events by itself.
        self.state_signature = state_signature

    def run(self, task: MechanismTaskContext) -> PlannerExecutorResult:
        self.env.reset()
        planner_event_start = len(
            getattr(getattr(self.planner, "model", None), "events", [])
        )
        executor_event_start = len(
            getattr(getattr(self.executor, "model", None), "events", [])
        )
        current_state_pgl = self.state_to_pgl(self.env.current_state)
        errors: List[Dict[str, Any]] = []
        plan_versions: List[Dict[str, Any]] = []
        directives: List[Dict[str, Any]] = []
        proposals: List[Dict[str, Any]] = []
        reviews: List[Dict[str, Any]] = []
        completion_reviews: List[Dict[str, Any]] = []
        observations: List[Dict[str, Any]] = []

        try:
            plan = self.planner.create_plan(
                task,
                required_changes=self._rendered_diff(
                    self._current_diff(), heading="Required structural changes"
                ),
            )
        except Exception as exc:
            plan = self._failed_plan()
            state = self._initial_state(task, plan, current_state_pgl)
            self._stop(state, "planner_create_plan_error", errors, exc)
            return self._result(
                state,
                plan_versions,
                directives,
                proposals,
                reviews,
                completion_reviews,
                observations,
                errors,
                planner_event_start,
                executor_event_start,
            )

        plan_versions.append(plan.to_dict())
        state = self._initial_state(task, plan, current_state_pgl)
        active_elementary_step = plan.elementary_step_at(state.control.plan_step_idx)
        if active_elementary_step is None:
            raise ValueError("initial elementary step index is outside the plan")
        visited = {self._signature_key(self.env.state_signature())}
        completion_checked_signatures: set[str] = set()

        while True:
            strict_product_match = bool(self.env.product_reached())
            state.environment.strict_product_match = strict_product_match
            current_signature = self._signature_key(self.env.state_signature())
            if strict_product_match and current_signature not in completion_checked_signatures:
                completion_checked_signatures.add(current_signature)
                confirmed = self._check_completion(
                    state=state,
                    task=task,
                    plan=plan,
                    completion_reviews=completion_reviews,
                    errors=errors,
                    strict_product_match=True,
                    plan_exhausted=False,
                )
                if confirmed is None:
                    break
                if confirmed:
                    self._finish_success(state)
                    break
            if state.control.mechanism_step_idx >= self.config.max_steps:
                self._stop(state, "max_steps", errors)
                break
            if state.control.env_call_count >= self.config.max_env_calls:
                self._stop(state, "max_env_calls", errors)
                break

            active_elementary_step = plan.elementary_step_at(state.control.plan_step_idx)
            if active_elementary_step is None:
                boundary_complete = self._check_completion(
                    state=state,
                    task=task,
                    plan=plan,
                    completion_reviews=completion_reviews,
                    errors=errors,
                    strict_product_match=False,
                    plan_exhausted=True,
                )
                if boundary_complete is None:
                    break
                if boundary_complete:
                    self._finish_success(
                        state,
                        stop_reason="mechanism_complete_with_boundary_difference",
                    )
                    break
                replanned = self._try_replan(
                    state=state,
                    task=task,
                    plan=plan,
                    reason={
                        "source": "elementary_plan_exhausted",
                        "detail": "All planned elementary steps were executed but the strict target was not reached.",
                    },
                    per_step_replans=0,
                    errors=errors,
                )
                if replanned is None:
                    break
                plan = replanned
                state.plan = plan
                plan_versions.append(plan.to_dict())
                self._reset_plan_position(state, plan)
                active_elementary_step = plan.elementary_step_at(state.control.plan_step_idx)
                if active_elementary_step is None:
                    self._stop(state, "empty_replan", errors)
                    break

            state.turn = TurnState()
            state.active_elementary_step_id = active_elementary_step["step_id"]
            directive_id = self._directive_id(plan, state.control.mechanism_step_idx, len(directives) + 1)
            try:
                directive = self.planner.make_directive(
                    task=task,
                    plan=plan,
                    active_elementary_step=active_elementary_step,
                    mechanism_step_idx=state.control.mechanism_step_idx,
                    current_state_pgl=state.environment.current_state_pgl,
                    history_actions_pgl=state.environment.history_actions_pgl,
                    last_env_feedback=state.environment.last_env_feedback,
                    directive_id=directive_id,
                    remaining_changes=self._rendered_diff(
                        self._current_diff(), heading="Remaining structural changes"
                    ),
                )
            except Exception as exc:
                self._stop(state, "planner_directive_error", errors, exc)
                break
            directives.append(directive.to_dict())
            state.turn.current_directive = directive.to_dict()
            per_step_replans = 0

            if directive.directive_decision == "REPLAN":
                replanned = self._try_replan(
                    state=state,
                    task=task,
                    plan=plan,
                    reason={
                        "source": "directive_decision",
                        "active_elementary_step": active_elementary_step,
                        "current_position": directive.current_position,
                        "uncertainty": directive.uncertainty,
                    },
                    per_step_replans=per_step_replans,
                    errors=errors,
                )
                if replanned is None:
                    break
                per_step_replans += 1
                plan = replanned
                state.plan = plan
                plan_versions.append(plan.to_dict())
                self._reset_plan_position(state, plan)
                active_elementary_step = plan.elementary_step_at(state.control.plan_step_idx)
                if active_elementary_step is None:
                    self._stop(state, "empty_replan", errors)
                    break
                directive_id = self._directive_id(plan, state.control.mechanism_step_idx, len(directives) + 1)
                try:
                    directive = self.planner.make_directive(
                        task=task,
                        plan=plan,
                        active_elementary_step=active_elementary_step,
                        mechanism_step_idx=state.control.mechanism_step_idx,
                        current_state_pgl=state.environment.current_state_pgl,
                        history_actions_pgl=state.environment.history_actions_pgl,
                        last_env_feedback=state.environment.last_env_feedback,
                        directive_id=directive_id,
                        remaining_changes=self._rendered_diff(
                            self._current_diff(), heading="Remaining structural changes"
                        ),
                    )
                except Exception as exc:
                    self._stop(state, "planner_directive_after_replan_error", errors, exc)
                    break
                directives.append(directive.to_dict())
                state.turn.current_directive = directive.to_dict()
                state.active_elementary_step_id = active_elementary_step["step_id"]

            approved_action: Optional[Dict[str, Any]] = None
            approved_proposal: Any = None
            approved_review: Any = None
            planner_feedback: Optional[Dict[str, Any]] = None
            format_failures = 0
            review_round = 0

            while review_round < self.config.max_planner_review_rounds:
                proposal_id = self._proposal_id(directive.directive_id, len(proposals) + 1)
                try:
                    proposal = self.executor.propose(
                        task=task,
                        plan=plan,
                        directive=directive,
                        current_state_pgl=state.environment.current_state_pgl,
                        history_actions_pgl=state.environment.history_actions_pgl,
                        legal_action_shapes=self.env.legal_local_actions(),
                        planner_feedback=planner_feedback,
                        proposal_id=proposal_id,
                        review_round=review_round + 1,
                    )
                except Exception as exc:
                    format_failures += 1
                    planner_feedback = self._feedback("executor_output_error", repr(exc))
                    state.turn.review_history.append(planner_feedback)
                    errors.append({"phase": "executor.propose", "error": repr(exc)})
                    if format_failures > self.config.max_executor_format_repairs:
                        break
                    continue

                proposals.append(proposal.to_dict())
                state.turn.current_proposal = proposal.to_dict()

                validation: Dict[str, Any]
                action: Optional[Dict[str, Any]] = None
                if not proposal.action_pgl:
                    validation = {
                        "valid": False,
                        "errors": [{"error": "empty_action", "plan_objection": proposal.plan_objection}],
                    }
                else:
                    try:
                        action, parse_result = self.parse_action(proposal.action_pgl)
                        validation = (
                            parse_result
                            if not parse_result.get("valid")
                            else self.validate_action(action, self.env.current_state, self.env.legal_local_actions())
                        )
                    except Exception as exc:
                        validation = {"valid": False, "errors": [{"error": "action_parse_error", "detail": repr(exc)}]}

                proposal.parsed_action = action
                proposal.validation_result = validation
                proposals[-1] = proposal.to_dict()
                state.turn.current_proposal = proposal.to_dict()

                # Pure syntax/schema/local-endpoint failures return directly to
                # the Executor. A genuine plan objection still goes to Planner.
                if not validation.get("valid") and not proposal.plan_objection:
                    format_failures += 1
                    planner_feedback = self._feedback("deterministic_validation", validation)
                    state.turn.review_history.append(planner_feedback)
                    if format_failures > self.config.max_executor_format_repairs:
                        break
                    continue

                review_round += 1
                state.turn.review_round = review_round
                review_id = self._review_id(proposal.proposal_id, len(reviews) + 1)
                review_diff = self._current_diff()
                try:
                    review = self.planner.review(
                        task=task,
                        plan=plan,
                        directive=directive,
                        current_state_pgl=state.environment.current_state_pgl,
                        history_actions_pgl=state.environment.history_actions_pgl,
                        proposal=proposal,
                        deterministic_validation=validation,
                        previous_reviews=state.turn.review_history,
                        review_id=review_id,
                        remaining_changes=self._rendered_diff(
                            review_diff, heading="Remaining structural changes"
                        ),
                        action_evidence=(
                            proposal_relevance(action, review_diff)
                            if review_diff is not None and action is not None
                            else None
                        ),
                    )
                except Exception as exc:
                    errors.append({"phase": "planner.review", "error": repr(exc)})
                    planner_feedback = self._feedback("planner_review_error", repr(exc))
                    state.turn.review_history.append(planner_feedback)
                    continue

                review_dict = review.to_dict()
                reviews.append(review_dict)
                state.turn.review_history.append(review_dict)

                if review.decision == "APPROVE":
                    if not validation.get("valid") or action is None:
                        planner_feedback = self._feedback(
                            "invalid_approval",
                            "Planner approved a missing or deterministically invalid action",
                        )
                        state.turn.review_history.append(planner_feedback)
                        errors.append({"phase": "planner.review", **planner_feedback})
                        continue
                    approved_action = action
                    approved_proposal = proposal
                    approved_review = review
                    break

                if review.decision == "REPLAN":
                    replanned = self._try_replan(
                        state=state,
                        task=task,
                        plan=plan,
                        reason=review_dict,
                        per_step_replans=per_step_replans,
                        errors=errors,
                    )
                    if replanned is None:
                        break
                    per_step_replans += 1
                    plan = replanned
                    state.plan = plan
                    plan_versions.append(plan.to_dict())
                    self._reset_plan_position(state, plan)
                    active_elementary_step = plan.elementary_step_at(state.control.plan_step_idx)
                    if active_elementary_step is None:
                        self._stop(state, "empty_replan", errors)
                        break
                    directive_id = self._directive_id(plan, state.control.mechanism_step_idx, len(directives) + 1)
                    try:
                        directive = self.planner.make_directive(
                            task=task,
                            plan=plan,
                            active_elementary_step=active_elementary_step,
                            mechanism_step_idx=state.control.mechanism_step_idx,
                            current_state_pgl=state.environment.current_state_pgl,
                            history_actions_pgl=state.environment.history_actions_pgl,
                            last_env_feedback=state.environment.last_env_feedback,
                            directive_id=directive_id,
                        )
                    except Exception as exc:
                        self._stop(state, "planner_directive_after_replan_error", errors, exc)
                        break
                    directives.append(directive.to_dict())
                    state.turn.current_directive = directive.to_dict()
                    state.active_elementary_step_id = active_elementary_step["step_id"]
                    planner_feedback = review_dict
                    continue

                planner_feedback = review_dict

            if state.control.done:
                break
            if approved_action is None or approved_proposal is None or approved_review is None:
                self._stop(state, "planner_executor_review_exhausted", errors)
                break

            previous_state_pgl = state.environment.current_state_pgl
            _, reward, env_done, info = self.env.step(approved_action)
            state.control.env_call_count += 1
            info = dict(info or {})
            current_state_pgl = self.state_to_pgl(self.env.current_state)
            edit_valid = bool(info.get("edit_valid", True))
            strict_product_match = bool(self.env.product_reached())
            signature_key = self._signature_key(self.env.state_signature())
            repeated = edit_valid and signature_key in visited and not strict_product_match
            if edit_valid:
                visited.add(signature_key)

            observation = ExecutionObservation(
                mechanism_step_idx=state.control.mechanism_step_idx,
                elementary_step_id=active_elementary_step["step_id"],
                approved_review_id=approved_review.review_id,
                action_pgl=approved_proposal.action_pgl,
                previous_state_pgl=previous_state_pgl,
                current_state_pgl=current_state_pgl,
                edit_valid=edit_valid,
                edit_operations=[str(x) for x in info.get("edit_operations", [])],
                reward=float(reward),
                strict_product_match=strict_product_match,
                product_reached=False,
                repeated_state=repeated,
                env_done=bool(env_done),
                env_error=info.get("edit_error"),
            )
            observation_dict = observation.to_dict()
            observations.append(observation_dict)
            state.environment.current_state_pgl = current_state_pgl
            state.environment.last_env_feedback = self._planner_observation(observation_dict)
            state.environment.strict_product_match = strict_product_match
            state.environment.product_reached = False
            state.environment.repeated_state = repeated

            step_completed = edit_valid and not repeated
            if step_completed:
                state.environment.history_actions_pgl.append(approved_proposal.action_pgl)
                state.environment.history_states_pgl.append(current_state_pgl)
                state.control.mechanism_step_idx += 1
                state.control.plan_step_idx += 1
                next_step = plan.elementary_step_at(state.control.plan_step_idx)
                state.active_elementary_step_id = (
                    next_step["step_id"] if next_step is not None else None
                )

            if repeated and self.config.stop_on_repeated_state:
                self._stop(state, "repeated_state_loop", errors)
                break
            # A strict product match must receive the mandatory LLM audit at
            # the top of the next loop even if the wrapped env reports done.
            if env_done and not strict_product_match:
                self._stop(state, "env_done", errors)
                break
        if not state.control.done:
            self._stop(state, "max_steps", errors)
        return self._result(
            state,
            plan_versions,
            directives,
            proposals,
            reviews,
            completion_reviews,
            observations,
            errors,
            planner_event_start,
            executor_event_start,
        )

    def _check_completion(
        self,
        *,
        state: PlannerExecutorState,
        task: MechanismTaskContext,
        plan: GlobalPlan,
        completion_reviews: List[Dict[str, Any]],
        errors: List[Dict[str, Any]],
        strict_product_match: bool,
        plan_exhausted: bool,
    ) -> Optional[bool]:
        check_id = (
            f"plan{plan.plan_version}-step{state.control.mechanism_step_idx}"
            f"-completion{len(completion_reviews) + 1}"
        )
        diff = self._current_diff()
        gate: Optional[Dict[str, Any]] = None
        if diff is not None:
            allowed, reason = boundary_completion_allowed(diff)
            gate = {
                "allowed": allowed,
                "reason": reason,
                "remaining": diff.to_dict(),
            }
        try:
            review = self.planner.check_completion(
                task=task,
                plan=plan,
                mechanism_step_idx=state.control.mechanism_step_idx,
                current_state_pgl=state.environment.current_state_pgl,
                history_actions_pgl=state.environment.history_actions_pgl,
                check_id=check_id,
                strict_product_match=strict_product_match,
                plan_exhausted=plan_exhausted,
                remaining_changes=self._rendered_diff(
                    diff, heading="Remaining structural changes"
                ),
                boundary_gate=gate,
            )
        except Exception as exc:
            self._stop(state, "planner_completion_check_error", errors, exc)
            return None

        review_dict = review.to_dict()
        confirmed = review.decision in {
            "CONFIRM_COMPLETE",
            "CONFIRM_COMPLETE_WITH_BOUNDARY_DIFFERENCE",
        }
        # A boundary difference is an omitted species, never an unfinished
        # principal product. The Gemini run ended 156 of 180 rollouts on this
        # verdict against 24 strict matches, with nothing checking it.
        if (
            confirmed
            and not strict_product_match
            and gate is not None
            and not gate["allowed"]
        ):
            review_dict = dict(review_dict)
            review_dict["runtime_override"] = {
                "original_decision": review.decision,
                "enforced_decision": "CONTINUE",
                "reason": gate["reason"],
            }
            confirmed = False
        completion_reviews.append(review_dict)
        state.environment.product_reached = confirmed
        if not confirmed:
            feedback = dict(state.environment.last_env_feedback or {})
            feedback["strict_product_match"] = strict_product_match
            feedback["llm_completion_review"] = review_dict
            if gate is not None:
                feedback["boundary_completion_gate"] = gate
            state.environment.last_env_feedback = feedback
        return confirmed

    def _try_replan(
        self,
        *,
        state: PlannerExecutorState,
        task: MechanismTaskContext,
        plan: GlobalPlan,
        reason: Any,
        per_step_replans: int,
        errors: List[Dict[str, Any]],
    ) -> Optional[GlobalPlan]:
        if per_step_replans >= self.config.max_replans_per_env_step:
            self._stop(state, "max_replans_per_env_step", errors)
            return None
        if state.control.replan_count >= self.config.max_total_replans:
            self._stop(state, "max_total_replans", errors)
            return None
        try:
            new_plan = self.planner.replan(
                task=task,
                plan=plan,
                current_state_pgl=state.environment.current_state_pgl,
                history_actions_pgl=state.environment.history_actions_pgl,
                reason=reason,
                remaining_changes=self._rendered_diff(
                    self._current_diff(), heading="Remaining structural changes"
                ),
            )
        except Exception as exc:
            # A replan that returns no steps is the Planner asserting, on a
            # second independent look with the diff in hand, that nothing
            # remains. That is a disagreement with the completion gate, not a
            # runtime error, and it must not be recorded as a crash.
            if self._asserts_nothing_remains(exc):
                self._stop(state, "replan_asserts_complete", errors)
                return None
            self._stop(state, "planner_replan_error", errors, exc)
            return None
        state.control.replan_count += 1
        return new_plan

    @staticmethod
    def _initial_state(
        task: MechanismTaskContext,
        plan: GlobalPlan,
        current_state_pgl: str,
    ) -> PlannerExecutorState:
        return PlannerExecutorState(
            task=task,
            plan=plan,
            active_elementary_step_id=plan.initial_elementary_step_id,
            environment=EnvironmentState(
                current_state_pgl=current_state_pgl,
                history_states_pgl=[current_state_pgl],
            ),
            control=ControlState(
                plan_step_idx=plan.elementary_step_index(plan.initial_elementary_step_id)
            ),
        )

    @staticmethod
    def _failed_plan() -> GlobalPlan:
        return GlobalPlan(
            plan_version=0,
            reaction_family="unavailable",
            reaction_summary="Planner failed before producing a plan.",
            global_mechanism="Unavailable.",
            elementary_steps=[{"step_id": "E0", "description": "Unavailable."}],
            initial_elementary_step_id="E0",
        )

    @staticmethod
    def _reset_plan_position(state: PlannerExecutorState, plan: GlobalPlan) -> None:
        state.control.plan_step_idx = plan.elementary_step_index(
            plan.initial_elementary_step_id
        )
        state.active_elementary_step_id = plan.initial_elementary_step_id

    @staticmethod
    def _asserts_nothing_remains(exc: Exception) -> bool:
        """True when a replan failed because the Planner returned no steps."""

        return isinstance(exc, ValueError) and "elementary_steps must be a non-empty list" in str(exc)

    def _target_state(self) -> Any:
        for name in ("final_state", "target_state"):
            state = getattr(self.env, name, None)
            if state is not None:
                return state
        return None

    def _current_diff(self) -> Optional[StructuralDiff]:
        """Structural changes still separating the current state from the target."""

        if self.state_signature is None:
            return None
        target = self._target_state()
        if target is None:
            return None
        try:
            return diff_signatures(
                self.state_signature(self.env.current_state),
                self.state_signature(target),
            )
        except Exception:
            # A diff is an aid, never a precondition. A runtime that cannot
            # produce one falls back to the previous prompt-only behaviour.
            return None

    def _rendered_diff(self, diff: Optional[StructuralDiff], *, heading: str) -> Optional[str]:
        return render_diff(diff, heading=heading) if diff is not None else None

    @staticmethod
    def _feedback(kind: str, detail: Any) -> Dict[str, Any]:
        return {"feedback_type": kind, "detail": detail}

    @staticmethod
    def _planner_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
        """Expose only observable, non-GT-derived environment feedback."""

        allowed = {
            "mechanism_step_idx",
            "elementary_step_id",
            "approved_review_id",
            "action_pgl",
            "previous_state_pgl",
            "current_state_pgl",
            "edit_valid",
            "edit_operations",
            "strict_product_match",
            "product_reached",
            "repeated_state",
            "env_done",
            "env_error",
        }
        return {key: value for key, value in observation.items() if key in allowed}

    @staticmethod
    def _signature_key(signature: Any) -> str:
        return json.dumps(signature, sort_keys=True, ensure_ascii=True, default=str)

    @staticmethod
    def _directive_id(plan: GlobalPlan, step_idx: int, sequence: int) -> str:
        return f"plan{plan.plan_version}-step{step_idx}-directive{sequence}"

    @staticmethod
    def _proposal_id(directive_id: str, sequence: int) -> str:
        return f"{directive_id}-proposal{sequence}"

    @staticmethod
    def _review_id(proposal_id: str, sequence: int) -> str:
        return f"{proposal_id}-review{sequence}"

    @staticmethod
    def _stop(
        state: PlannerExecutorState,
        reason: str,
        errors: List[Dict[str, Any]],
        exc: Optional[Exception] = None,
    ) -> None:
        state.control.done = True
        state.control.success = False
        state.control.stop_reason = reason
        if exc is not None:
            errors.append({"phase": reason, "error": repr(exc)})

    @staticmethod
    def _finish_success(
        state: PlannerExecutorState,
        *,
        stop_reason: str = "product_reached",
    ) -> None:
        state.control.done = True
        state.control.success = True
        state.control.stop_reason = stop_reason
        state.environment.product_reached = True

    def _result(
        self,
        state: PlannerExecutorState,
        plan_versions: List[Dict[str, Any]],
        directives: List[Dict[str, Any]],
        proposals: List[Dict[str, Any]],
        reviews: List[Dict[str, Any]],
        completion_reviews: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
        errors: List[Dict[str, Any]],
        planner_event_start: int,
        executor_event_start: int,
    ) -> PlannerExecutorResult:
        planner_events = list(
            getattr(getattr(self.planner, "model", None), "events", [])[
                planner_event_start:
            ]
        )
        executor_events = list(
            getattr(getattr(self.executor, "model", None), "events", [])[
                executor_event_start:
            ]
        )
        return PlannerExecutorResult(
            state=state,
            plan_versions=plan_versions,
            directives=directives,
            executor_proposals=proposals,
            planner_reviews=reviews,
            completion_reviews=completion_reviews,
            execution_observations=observations,
            errors=errors,
            model_events={"planner": planner_events, "executor": executor_events},
        )
