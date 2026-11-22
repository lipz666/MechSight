from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def _required_text(data: Dict[str, Any], key: str) -> str:
    value = str(data.get(key) or "").strip()
    if not value:
        raise ValueError(f"missing non-empty field: {key}")
    return value


def _required_bool(data: Dict[str, Any], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"field must be a JSON boolean: {key}")
    return value


@dataclass
class MechanismTaskContext:
    reaction_id: str
    task_description: str
    initial_state_pgl: str
    target_state_pgl: str
    conditions: Any = None
    reaction_metadata: Dict[str, Any] = field(default_factory=dict)
    # Ablation-only. Set from the annotated path to measure how much of the
    # scored gap is guessing MechFinder's step-count convention rather than
    # doing chemistry. It reaches the Planner alone and must stay None in any
    # run whose numbers are reported as a capability result.
    oracle_step_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GlobalPlan:
    plan_version: int
    reaction_family: str
    reaction_summary: str
    global_mechanism: str
    elementary_steps: List[Dict[str, str]]
    initial_elementary_step_id: str
    alternative_mechanisms: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    replanned_from_version: Optional[int] = None
    replan_reason: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], plan_version: int) -> "GlobalPlan":
        # Old plan files remain readable, but every newly serialized plan uses
        # the elementary-step protocol.
        raw_steps = data.get("elementary_steps")
        if raw_steps is None:
            raw_steps = data.get("macro_steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("elementary_steps must be a non-empty list")
        steps: List[Dict[str, str]] = []
        for index, item in enumerate(raw_steps, start=1):
            if isinstance(item, str):
                description = item.strip()
                step_id = f"E{index}"
            elif isinstance(item, dict):
                description = str(item.get("description") or "").strip()
                step_id = str(item.get("step_id") or f"E{index}").strip()
            else:
                raise ValueError(f"elementary_steps[{index - 1}] must be an object or string")
            if not description:
                raise ValueError(f"elementary_steps[{index - 1}] has no description")
            steps.append({"step_id": step_id, "description": description})
        if len({item["step_id"] for item in steps}) != len(steps):
            raise ValueError("elementary step ids must be unique")

        initial = str(
            data.get("initial_elementary_step_id")
            or data.get("initial_macro_step_id")
            or steps[0]["step_id"]
        ).strip()
        if initial not in {item["step_id"] for item in steps}:
            raise ValueError(
                f"initial_elementary_step_id is not in elementary_steps: {initial}"
            )
        return cls(
            plan_version=plan_version,
            reaction_family=_required_text(data, "reaction_family"),
            reaction_summary=_required_text(data, "reaction_summary"),
            global_mechanism=_required_text(data, "global_mechanism"),
            elementary_steps=steps,
            initial_elementary_step_id=initial,
            alternative_mechanisms=[str(x) for x in data.get("alternative_mechanisms", [])],
            uncertainties=[str(x) for x in data.get("uncertainties", [])],
            replanned_from_version=data.get("replanned_from_version"),
            replan_reason=data.get("replan_reason"),
        )

    def global_context(self) -> Dict[str, Any]:
        return {
            "reaction_family": self.reaction_family,
            "reaction_summary": self.reaction_summary,
            "global_mechanism": self.global_mechanism,
            "elementary_steps": self.elementary_steps,
        }

    def elementary_step(self, step_id: Optional[str]) -> Dict[str, str]:
        selected = step_id or self.initial_elementary_step_id
        for item in self.elementary_steps:
            if item["step_id"] == selected:
                return dict(item)
        raise ValueError(f"unknown elementary step id: {selected}")

    def elementary_step_index(self, step_id: Optional[str]) -> int:
        selected = step_id or self.initial_elementary_step_id
        for index, item in enumerate(self.elementary_steps):
            if item["step_id"] == selected:
                return index
        raise ValueError(f"unknown elementary step id: {selected}")

    def elementary_step_at(self, index: int) -> Optional[Dict[str, str]]:
        if 0 <= index < len(self.elementary_steps):
            return dict(self.elementary_steps[index])
        return None

    # Read-only aliases ease loading integrations written against v2. New
    # results never serialize these names.
    @property
    def macro_steps(self) -> List[Dict[str, str]]:
        return self.elementary_steps

    @property
    def initial_macro_step_id(self) -> str:
        return self.initial_elementary_step_id

    def macro_step(self, step_id: Optional[str]) -> Dict[str, str]:
        return self.elementary_step(step_id)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepDirective:
    directive_id: str
    plan_version: int
    mechanism_step_idx: int
    global_context: Dict[str, Any]
    directive_decision: str
    active_elementary_step: Dict[str, str]
    current_position: str
    expected_electron_transfer_event: str
    must_preserve: List[str] = field(default_factory=list)
    avoid: List[str] = field(default_factory=list)
    uncertainty: str = ""

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        plan: GlobalPlan,
        mechanism_step_idx: int,
        directive_id: str,
        expected_elementary_step: Optional[Dict[str, str]] = None,
    ) -> "StepDirective":
        active = data.get("active_elementary_step")
        if not isinstance(active, dict):
            active = data.get("active_macro_step")
        if not isinstance(active, dict):
            requested_id = data.get("active_elementary_step_id") or data.get("active_macro_step_id")
            active = plan.elementary_step(requested_id)
        step_id = str(active.get("step_id") or plan.initial_elementary_step_id)
        known_step_ids = {item["step_id"] for item in plan.elementary_steps}
        if step_id not in known_step_ids:
            raise ValueError(f"active elementary step is not in the current plan: {step_id}")
        if expected_elementary_step is not None:
            expected_id = str(expected_elementary_step["step_id"])
            if step_id != expected_id:
                raise ValueError(
                    f"Planner cannot change the active elementary step: expected {expected_id}, got {step_id}"
                )
            active = dict(expected_elementary_step)
        else:
            active = plan.elementary_step(step_id)

        raw_decision = str(data.get("directive_decision") or "").strip().upper()
        if raw_decision.startswith("EXECUTE"):
            decision = "EXECUTE"
        elif raw_decision.startswith("REPLAN"):
            decision = "REPLAN"
        else:
            decision = raw_decision
        if not decision:
            # Backward-compatible interpretation of v2 progress output.
            progress = data.get("progress_from_previous_action")
            progress_decision = (
                str(progress.get("decision") or "") if isinstance(progress, dict) else str(progress or "")
            ).strip().upper()
            decision = "REPLAN" if progress_decision == "REPLAN" else "EXECUTE"
        if decision not in {"EXECUTE", "REPLAN"}:
            raise ValueError(f"invalid directive_decision: {decision}")
        return cls(
            directive_id=directive_id,
            plan_version=plan.plan_version,
            mechanism_step_idx=mechanism_step_idx,
            global_context=plan.global_context(),
            directive_decision=decision,
            active_elementary_step=active,
            current_position=_required_text(data, "current_position"),
            expected_electron_transfer_event=_required_text(data, "expected_electron_transfer_event"),
            must_preserve=[str(x) for x in data.get("must_preserve", [])],
            avoid=[str(x) for x in data.get("avoid", [])],
            uncertainty=str(data.get("uncertainty") or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def active_macro_step(self) -> Dict[str, str]:
        return self.active_elementary_step


@dataclass
class ExecutorProposal:
    proposal_id: str
    directive_id: str
    review_round: int
    event_summary: str
    plan_objection: str
    action_pgl: str
    parsed_action: Optional[Dict[str, Any]] = None
    validation_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlannerReview:
    review_id: str
    proposal_id: str
    decision: str
    global_consistency: bool
    elementary_step_consistency: bool
    electron_event_consistency: bool
    chemical_assessment: str
    required_correction: str = ""
    preserve_from_proposal: List[str] = field(default_factory=list)
    forbid_in_next_attempt: List[str] = field(default_factory=list)
    plan_status: str = "KEEP"

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, proposal_id: str, review_id: str) -> "PlannerReview":
        decision = str(data.get("decision") or "").upper()
        if decision not in {"APPROVE", "REVISE", "REPLAN"}:
            raise ValueError(f"invalid Planner decision: {decision!r}")
        plan_status = str(data.get("plan_status") or ("REVISE_REMAINDER" if decision == "REPLAN" else "KEEP")).upper()
        if plan_status not in {"KEEP", "REVISE_REMAINDER"}:
            raise ValueError(f"invalid plan_status: {plan_status!r}")
        elementary_consistency = data.get("elementary_step_consistency")
        if elementary_consistency is None:
            elementary_consistency = data.get("macro_step_consistency")
        if not isinstance(elementary_consistency, bool):
            raise ValueError("field must be a JSON boolean: elementary_step_consistency")
        consistency_fields = (
            _required_bool(data, "global_consistency"),
            elementary_consistency,
            _required_bool(data, "electron_event_consistency"),
        )
        if decision == "APPROVE" and not all(consistency_fields):
            raise ValueError("APPROVE requires all three consistency checks to be true")
        return cls(
            review_id=review_id,
            proposal_id=proposal_id,
            decision=decision,
            global_consistency=consistency_fields[0],
            elementary_step_consistency=consistency_fields[1],
            electron_event_consistency=consistency_fields[2],
            chemical_assessment=_required_text(data, "chemical_assessment"),
            required_correction=str(data.get("required_correction") or ""),
            preserve_from_proposal=[str(x) for x in data.get("preserve_from_proposal", [])],
            forbid_in_next_attempt=[str(x) for x in data.get("forbid_in_next_attempt", [])],
            plan_status=plan_status,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def macro_step_consistency(self) -> bool:
        return self.elementary_step_consistency


@dataclass
class CompletionReview:
    check_id: str
    mechanism_step_idx: int
    decision: str
    atom_connectivity_match: bool
    bond_order_match: bool
    formal_charge_match: bool
    explicit_h_match: bool
    chemical_assessment: str
    required_next_event: str = ""
    ignored_boundary_differences: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        check_id: str,
        mechanism_step_idx: int,
    ) -> "CompletionReview":
        decision = str(data.get("decision") or "").upper()
        if decision not in {
            "CONFIRM_COMPLETE",
            "CONFIRM_COMPLETE_WITH_BOUNDARY_DIFFERENCE",
            "CONTINUE",
        }:
            raise ValueError(f"invalid completion decision: {decision!r}")
        checks = (
            _required_bool(data, "atom_connectivity_match"),
            _required_bool(data, "bond_order_match"),
            _required_bool(data, "formal_charge_match"),
            _required_bool(data, "explicit_h_match"),
        )
        if decision == "CONFIRM_COMPLETE" and not all(checks):
            raise ValueError("CONFIRM_COMPLETE requires all structural checks to be true")
        boundary_differences = [
            str(x) for x in data.get("ignored_boundary_differences", [])
        ]
        if (
            decision == "CONFIRM_COMPLETE_WITH_BOUNDARY_DIFFERENCE"
            and not boundary_differences
        ):
            raise ValueError(
                "boundary completion requires ignored_boundary_differences"
            )
        return cls(
            check_id=check_id,
            mechanism_step_idx=mechanism_step_idx,
            decision=decision,
            atom_connectivity_match=checks[0],
            bond_order_match=checks[1],
            formal_charge_match=checks[2],
            explicit_h_match=checks[3],
            chemical_assessment=_required_text(data, "chemical_assessment"),
            required_next_event=str(data.get("required_next_event") or ""),
            ignored_boundary_differences=boundary_differences,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GroundTruthAssessment:
    planner_global_mechanism_judgment: str
    planner_elementary_steps_judgment: str
    predicted_events_judgment: str
    gt_mechanism_summary: str
    planner_global_assessment: str
    planner_elementary_steps_assessment: str
    predicted_events_assessment: str
    concise_conclusion: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundTruthAssessment":
        allowed = {"correct", "partially_correct", "incorrect"}
        global_judgment = str(
            data.get("planner_global_mechanism_judgment") or ""
        ).lower()
        elementary_judgment = str(
            data.get("planner_elementary_steps_judgment") or ""
        ).lower()
        event_judgment = str(data.get("predicted_events_judgment") or "").lower()
        if global_judgment not in allowed:
            raise ValueError(
                "invalid planner_global_mechanism_judgment: "
                f"{global_judgment!r}"
            )
        if elementary_judgment not in allowed:
            raise ValueError(
                "invalid planner_elementary_steps_judgment: "
                f"{elementary_judgment!r}"
            )
        if event_judgment not in allowed:
            raise ValueError(f"invalid predicted_events_judgment: {event_judgment!r}")
        return cls(
            planner_global_mechanism_judgment=global_judgment,
            planner_elementary_steps_judgment=elementary_judgment,
            predicted_events_judgment=event_judgment,
            gt_mechanism_summary=_required_text(data, "gt_mechanism_summary"),
            planner_global_assessment=_required_text(
                data, "planner_global_assessment"
            ),
            planner_elementary_steps_assessment=_required_text(
                data, "planner_elementary_steps_assessment"
            ),
            predicted_events_assessment=_required_text(
                data, "predicted_events_assessment"
            ),
            concise_conclusion=_required_text(data, "concise_conclusion"),
        )

    @property
    def planner_mechanism_judgment(self) -> str:
        """Compatibility alias for code reading the earlier global judgment."""

        return self.planner_global_mechanism_judgment

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionObservation:
    mechanism_step_idx: int
    elementary_step_id: str
    approved_review_id: str
    action_pgl: str
    previous_state_pgl: str
    current_state_pgl: str
    edit_valid: bool
    edit_operations: List[str]
    reward: float
    strict_product_match: bool
    product_reached: bool
    repeated_state: bool
    env_done: bool
    env_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnvironmentState:
    current_state_pgl: str
    history_states_pgl: List[str]
    history_actions_pgl: List[str] = field(default_factory=list)
    last_env_feedback: Optional[Dict[str, Any]] = None
    strict_product_match: bool = False
    product_reached: bool = False
    repeated_state: bool = False


@dataclass
class TurnState:
    current_directive: Optional[Dict[str, Any]] = None
    current_proposal: Optional[Dict[str, Any]] = None
    review_history: List[Dict[str, Any]] = field(default_factory=list)
    review_round: int = 0


@dataclass
class ControlState:
    mechanism_step_idx: int = 0
    plan_step_idx: int = 0
    env_call_count: int = 0
    replan_count: int = 0
    done: bool = False
    success: bool = False
    stop_reason: Optional[str] = None


@dataclass
class PlannerExecutorState:
    task: MechanismTaskContext
    plan: GlobalPlan
    active_elementary_step_id: Optional[str]
    environment: EnvironmentState
    turn: TurnState = field(default_factory=TurnState)
    control: ControlState = field(default_factory=ControlState)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def active_macro_step_id(self) -> Optional[str]:
        return self.active_elementary_step_id


@dataclass
class PlannerExecutorResult:
    state: PlannerExecutorState
    plan_versions: List[Dict[str, Any]] = field(default_factory=list)
    directives: List[Dict[str, Any]] = field(default_factory=list)
    executor_proposals: List[Dict[str, Any]] = field(default_factory=list)
    planner_reviews: List[Dict[str, Any]] = field(default_factory=list)
    completion_reviews: List[Dict[str, Any]] = field(default_factory=list)
    execution_observations: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    model_events: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.state.control.success

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
