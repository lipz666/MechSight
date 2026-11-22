from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .schemas import ExecutorProposal, GlobalPlan, MechanismTaskContext, StepDirective


COMMON_MECHANISM_TASK = """The task is to infer a stepwise electron-transfer mechanism connecting a complete initial ReactionState to a target ReactionState under the provided reaction conditions.

A mechanism path alternates:
ReactionState_0 -> ReactionTransformation_0 -> ReactionState_1 -> ... -> ReactionState_n.

Each action must describe only the next elementary event from the current ReactionState. Do not jump directly to the target product and do not use unobserved future intermediate states.

Keep internal reasoning concise and convergent. Compare each structure once, avoid repeatedly rereading the same bonds, and reserve enough output capacity for the required final protocol block."""


PGL_STATE_GUIDE = """PGL ReactionState reference:

1. ReactionState represents the complete chemical system at one point in the mechanism and contains a list of Species objects.
2. Each Species contains role, state, and molecule. Roles may include reactant, reagent, catalyst, intermediate, transition_state, and product.
3. A Molecule contains one or more Substructure objects.
4. Substructure.atomdict maps immutable atom labels to Atom objects, for example 'C1': Atom('C', Hcount=1, charge=-1). Atom labels are unique identifiers and must never be renamed or invented.
5. Atom.element gives the element. Hcount is serialized only when nonzero; omitted Hcount therefore means the stored Hcount is zero. Ordinary implicit valence hydrogens may still be inferred chemically, but they are distinct from a nonzero recorded Hcount. Formal charge is serialized only when nonzero.
6. Substructure.bondlist describes connectivity using Bond(atom1, atom2, kind). Common kinds include single, double, triple, aromatic, and transition.
7. The initial ReactionState is the complete reactant-side system. The target ReactionState is the required product-side system. The current ReactionState is the only state on which the next action can operate.
8. All explicitly present reactants, reagents, catalysts, intermediates, ions, and byproducts remain part of the current chemical system unless an environment action changes their representation."""


PGL_ACTION_GUIDE = """PGL ReactionTransformation reference:

1. ReactionTransformation describes one elementary event and contains one or more ElectronTransfer objects.
2. ElectronTransfer(source, target, electrons, kind) describes electron flow. The source may be an atom lone pair or an existing sigma/pi bond. The target may be an atom, a new bond, or a bond whose order is increased.
3. Use electrons=2 for ordinary paired-electron movement unless single-electron chemistry is explicitly required.
4. Common kinds include bond_forming, bond_breaking_or_pi_shift, pi_shift, pi_to_atom, sigma_shift, proton_transfer, single_electron_transfer, redox, coordination, and unknown.
5. One ReactionTransformation may contain multiple ElectronTransfers when they are coupled parts of the same elementary event, such as nucleophilic attack on a pi bond together with movement of the original pi electrons.
6. Every atom label in an action must occur in the current ReactionState. A new bond may be created only between existing labeled atoms.
7. ElectronTransfer endpoints must always use structural tuples. Never use shorthand such as source='O1' or target='C1'. Use source=('O1', 'lone_pair'), source=('C1', 'N1', 'pi_bond'), target=('N1', 'lone_pair'), or target=('O1', 'C1', 'new_sigma_bond') as appropriate.
8. The concrete action format is:
ReactionTransformation(
    transformation_type='reaction|resonance|ts_collapse',
    electron_transfers=[ElectronTransfer(...), ...]
)"""


PLANNER_DECOMPOSITION_GUIDE = """Planner elementary-step decomposition, grounding, and scope rules:

1. Each elementary_steps item must describe exactly one chemically elementary event that can be executed as one ReactionTransformation and produce one new ReactionState. Multiple electron movements belong in one step only when they are coupled parts of the same event. Split events that are separated by a distinct intermediate.
2. Represent sequential proton transfer, reagent activation, metal coordination or dissociation, leaving-group departure, elimination, pi-bond restoration, and workup as separate steps when they change the represented state. Do not compress an entire textbook reaction phase into one step.
3. Account for every chemically relevant change in heavy-atom connectivity, bond order, formal charge, explicit-hydrogen ownership, and every nonzero recorded Hcount. Before finalizing the plan, mentally follow the proposed steps from the supplied initial PGL toward the supplied target PGL and add any missing elementary event.
4. Before emitting elementary_steps, explicitly perform an internal coverage audit. Check, in order, whether the plan covers every applicable item: reactant or reagent activation; generation of the actual nucleophile or electrophile; formation and collapse of key intermediates; every required bond formation, bond cleavage, and bond-order change; necessary proton transfers; leaving-group departure; catalyst or reagent regeneration; and final neutralization needed to reach the stable product. If an applicable item changes the represented PGL state and is separated by an intermediate, it must appear as its own elementary step. Do not print this checklist or hidden analysis; use it to make the output plan complete.
5. Select the mechanism variant from the actual structures, reagents, and conditions in the supplied PGL rather than relying only on a reaction-family name. Never invent an atom, bond, proton source, or proton acceptor.
6. Elementary-step descriptions must remain natural language. They may use functional roles such as carbonyl carbon, nucleophile, leaving group, or proton donor; they are not required to enumerate fixed atom labels. When a description does cite a specific atom, preserve its exact PGL label. The later StepDirective must use current-state atom labels precisely enough for the Executor.
7. The target PGL may omit species that participated earlier but no longer participate in the principal reaction, including transient reagents, spectators, leaving fragments, and byproducts. Do not add cleanup steps solely because such a species is absent from the target. In particular, H2 formed when hydride deprotonates an alcohol or phenol may be omitted by the annotated target; do not break the H-H bond, delete H2, or invent another proton transfer to remove it. Once the principal product has the required connectivity, bond orders, formal charges, and chemically necessary hydrogens, treat these omitted species as a possible dataset-scope or annotation boundary, record them in uncertainties, and regard the chemical mechanism as complete.
8. When initial, current, and target PGL disagree about hydrogen, first decide whether a real proton transfer is chemically required. Plan an additional proton transfer only when target valence or charge requires it, a valid represented donor and acceptor exist, and the transfer conserves the represented hydrogen. If the only discrepancy is a missing or extra explicit-H label, an unexplained disappearing H, or an omitted zero Hcount with no chemically required transfer, treat it as a likely representation or annotation issue and record it in uncertainties.
9. Hcount is normally absent when its stored value is zero. Do not infer an executable proton transfer merely because Hcount is not printed. Ordinary implicit valence hydrogen, nonzero recorded Hcount, and an explicitly labeled H atom are different representations and must not be conflated.
10. Do not force the number of planned steps to match any unknown annotated path. Determine the sequence from elementary-event boundaries and chemical necessity. Do not use a gold path, reward, hidden intermediate, or precomputed target-change summary."""


REQUIRED_CHANGES_CONTRACT = """Required-change accounting:

1. The runtime supplies a deterministic list of every structural change separating the two states: bonds to form, bonds to break, bond-order changes, formal-charge changes, and recorded-Hcount changes. This list is computed from the PGL graphs, not inferred, and it is complete.
2. Every listed change must be produced by some elementary step in your plan. Before emitting elementary_steps, check each listed change against your plan and add the steps needed to cover the ones you missed. A plan that leaves a listed change unaccounted for is incomplete.
3. The list states what must change, not how many steps it takes. One elementary step may cover several listed changes when they are coupled parts of one event, and one listed change may require several steps. Do not read the list as a step count.
4. The list does not contain activation, coordination, proton shuttling, intermediate collapse, catalyst regeneration, or neutralization events that cancel out between the two endpoint states. Those still belong in the plan when chemistry requires them.
5. Atoms reported as present now but absent from the target are a possible annotation boundary. Record them in uncertainties. Do not invent a deletion or an unsupported proton transfer to remove them."""


PLAN_SCHEMA = """{
  "plan_version": 1,
  "reaction_family": "short reaction-family label",
  "reaction_summary": "one or two sentence overall summary",
  "global_mechanism": "coherent natural-language description of the full mechanism",
  "elementary_steps": [
    {"step_id": "E1", "description": "one natural-language elementary-event step"}
  ],
  "initial_elementary_step_id": "E1",
  "alternative_mechanisms": [],
  "uncertainties": []
}"""


DIRECTIVE_SCHEMA = """{
  "directive_decision": "EXECUTE|REPLAN",
  "active_elementary_step": {"step_id": "E1", "description": "the authoritative natural-language elementary step supplied by the runtime"},
  "current_position": "where the real current state lies in the global mechanism",
  "expected_electron_transfer_event": "the electron-transfer event that should happen now, using current atom labels",
  "must_preserve": [],
  "avoid": [],
  "uncertainty": ""
}"""


REVIEW_SCHEMA = """{
  "decision": "APPROVE|REVISE|REPLAN",
  "global_consistency": true,
  "elementary_step_consistency": true,
  "electron_event_consistency": true,
  "chemical_assessment": "concise assessment grounded in the current PGL",
  "required_correction": "empty for APPROVE; precise correction for REVISE",
  "preserve_from_proposal": [],
  "forbid_in_next_attempt": [],
  "plan_status": "KEEP|REVISE_REMAINDER"
}"""


COMPLETION_SCHEMA = """{
  "decision": "CONFIRM_COMPLETE|CONFIRM_COMPLETE_WITH_BOUNDARY_DIFFERENCE|CONTINUE",
  "atom_connectivity_match": true,
  "bond_order_match": true,
  "formal_charge_match": true,
  "explicit_h_match": true,
  "chemical_assessment": "concise comparison of the complete current and target systems",
  "required_next_event": "empty for either completion decision; next necessary chemical event for CONTINUE",
  "ignored_boundary_differences": ["omitted byproduct, spectator, reagent fragment, or representation-only difference"]
}"""


POSTHOC_GT_ASSESSMENT_SCHEMA = """{
  "planner_global_mechanism_judgment": "correct|partially_correct|incorrect",
  "planner_elementary_steps_judgment": "correct|partially_correct|incorrect",
  "predicted_events_judgment": "correct|partially_correct|incorrect",
  "gt_mechanism_summary": "a concise Chinese summary of the mechanism expressed by the GT events",
  "planner_global_assessment": "concise Chinese explanation of the global-mechanism judgment",
  "planner_elementary_steps_assessment": "concise Chinese explanation of strict normalized event coverage and order",
  "predicted_events_assessment": "concise Chinese explanation of the executed-event judgment",
  "concise_conclusion": "two to four Chinese sentences comparing the Planner and executed events with GT"
}"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _history(history_actions_pgl: List[str]) -> str:
    if not history_actions_pgl:
        return "None"
    return "\n\n".join(
        f"Historical ReactionTransformation {index}:\n{action}"
        for index, action in enumerate(history_actions_pgl)
    )


ORACLE_STEP_COUNT_CONTRACT = """Step-count reference (ablation input):

1. The runtime supplies the number of elementary steps used by the reference annotation for this reaction. Produce exactly that many elementary_steps items.
2. This fixes only how finely the mechanism is divided. It tells you nothing about which bonds move, which atoms react, or in what order. Derive all of that from the structures as usual.
3. If your chemistry needs a different number of steps, adjust where you draw concerted-versus-stepwise boundaries to reach the supplied count. Never invent a chemical event to pad the count, and never delete a required event to reach it. If the count cannot be met without inventing or deleting an event, keep the chemistry correct and record the conflict in uncertainties."""


def _oracle_step_block(task: MechanismTaskContext) -> str:
    if task.oracle_step_count is None:
        return ""
    return f"\n<reference_step_count>\n{task.oracle_step_count}\n</reference_step_count>\n"


def _required_changes_block(required_changes: Optional[str], tag: str) -> str:
    if not required_changes:
        return ""
    return f"\n<{tag}>\n{required_changes}\n</{tag}>\n"


def create_plan_messages(
    task: MechanismTaskContext,
    required_changes: Optional[str] = None,
) -> List[Dict[str, str]]:
    contract = f"\n\n{REQUIRED_CHANGES_CONTRACT}" if required_changes else ""
    if task.oracle_step_count is not None:
        contract += f"\n\n{ORACLE_STEP_COUNT_CONTRACT}"
    system = f"""{COMMON_MECHANISM_TASK}

{PGL_STATE_GUIDE}

You are the Global Planner. Compare the complete initial and target PGL structures yourself and infer a chemically plausible global mechanism under the stated conditions.

{PLANNER_DECOMPOSITION_GUIDE}{contract}

Describe every elementary step only in natural language. Do not output a concrete PGL ReactionTransformation, executable predicates, gold-path claims, or unobserved intermediates as facts. Use E1, E2, ... step identifiers. Do not merge sequential events merely to shorten the plan.

Output exactly one <plan>...</plan> block containing valid JSON with this schema:
{PLAN_SCHEMA}"""
    user = f"""<task_identity>
reaction_id: {task.reaction_id}
</task_identity>

<mechanism_task>
{task.task_description}
</mechanism_task>

<reaction_conditions>
{_json(task.conditions)}
</reaction_conditions>

<initial_state_pgl>
{task.initial_state_pgl}
</initial_state_pgl>

<target_state_pgl>
{task.target_state_pgl}
</target_state_pgl>
{_required_changes_block(required_changes, "required_structural_changes")}{_oracle_step_block(task)}
Generate the GlobalPlan now. Do not use skills or any precomputed target-change summary."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def replan_messages(
    task: MechanismTaskContext,
    plan: GlobalPlan,
    current_state_pgl: str,
    history_actions_pgl: List[str],
    reason: Any,
    remaining_changes: Optional[str] = None,
) -> List[Dict[str, str]]:
    contract = f"\n\n{REQUIRED_CHANGES_CONTRACT}" if remaining_changes else ""
    if task.oracle_step_count is not None:
        contract += f"\n\n{ORACLE_STEP_COUNT_CONTRACT}"
    system = f"""{COMMON_MECHANISM_TASK}

{PGL_STATE_GUIDE}

You are the Global Planner revising only the unexecuted remainder of a mechanism. Executed history and the real current ReactionState are immutable facts. Build a new natural-language global mechanism and ordered remaining elementary steps from the current state to the original target. The first returned step must be executable next from the supplied current state.

{PLANNER_DECOMPOSITION_GUIDE}{contract}

Output exactly one <plan>...</plan> valid JSON block using this schema:
{PLAN_SCHEMA}"""
    user = f"""<original_task>
{task.task_description}
</original_task>

<reaction_conditions>
{_json(task.conditions)}
</reaction_conditions>

<previous_global_plan>
{_json(plan.to_dict())}
</previous_global_plan>

<executed_action_history>
{_history(history_actions_pgl)}
</executed_action_history>

<current_state_pgl>
{current_state_pgl}
</current_state_pgl>

<target_state_pgl>
{task.target_state_pgl}
</target_state_pgl>

<replan_reason>
{_json(reason)}
</replan_reason>
{_required_changes_block(remaining_changes, "remaining_structural_changes")}{_oracle_step_block(task)}
Replan the remaining mechanism. Do not use skills, gold actions, or future annotated states."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def directive_messages(
    task: MechanismTaskContext,
    plan: GlobalPlan,
    active_elementary_step: Dict[str, str],
    mechanism_step_idx: int,
    current_state_pgl: str,
    history_actions_pgl: List[str],
    last_env_feedback: Optional[Dict[str, Any]],
    remaining_changes: Optional[str] = None,
) -> List[Dict[str, str]]:
    system = f"""{COMMON_MECHANISM_TASK}

{PGL_STATE_GUIDE}

You are the Global Planner grounding the one authoritative elementary step selected by the runtime. The runtime, not the model, owns step advancement. One successfully executed, non-repeated environment action completes exactly one elementary step and the runtime automatically selects the next step.

{PLANNER_DECOMPOSITION_GUIDE}

Return directive_decision as exactly EXECUTE or REPLAN, with no explanation inside that field. Echo the supplied active_elementary_step without changing its id or description. Choose EXECUTE only when that step can be represented as exactly one next ReactionTransformation from the current PGL. Choose REPLAN if the authoritative step is no longer executable or improperly combines multiple sequential events. If a previous environment edit was rejected, the same elementary step remains active. If a remaining difference is only a likely transient-species or hydrogen-annotation mismatch, report it in uncertainty and do not direct the Executor to invent a deletion or unsupported proton transfer.

Do not output a concrete ReactionTransformation. Output exactly one <directive>...</directive> valid JSON block with this schema:
{DIRECTIVE_SCHEMA}

Environment feedback contains only observable edit/state information; no reward or annotated-path score is available."""
    user = f"""<global_plan>
{_json(plan.to_dict())}
</global_plan>

<mechanism_task_description>
{task.task_description}
</mechanism_task_description>

<reaction_conditions>
{_json(task.conditions)}
</reaction_conditions>

<active_elementary_step>
{_json(active_elementary_step)}
</active_elementary_step>

<mechanism_step_idx>
{mechanism_step_idx}
</mechanism_step_idx>

<initial_state_pgl>
{task.initial_state_pgl}
</initial_state_pgl>

<target_state_pgl>
{task.target_state_pgl}
</target_state_pgl>

<current_state_pgl>
{current_state_pgl}
</current_state_pgl>

<executed_action_history>
{_history(history_actions_pgl)}
</executed_action_history>

<last_environment_feedback>
{_json(last_env_feedback) if last_env_feedback is not None else "None"}
</last_environment_feedback>
{_required_changes_block(remaining_changes, "remaining_structural_changes")}
Produce the StepDirective for exactly the supplied active_elementary_step. Its global context will be copied from the authoritative GlobalPlan before transmission to the Executor."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def completion_check_messages(
    task: MechanismTaskContext,
    plan: GlobalPlan,
    mechanism_step_idx: int,
    current_state_pgl: str,
    history_actions_pgl: List[str],
    *,
    strict_product_match: bool,
    plan_exhausted: bool,
    remaining_changes: Optional[str] = None,
    boundary_gate: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    gate_clause = ""
    if boundary_gate is not None and not boundary_gate.get("allowed", True):
        gate_clause = (
            "\n\nCONFIRM_COMPLETE_WITH_BOUNDARY_DIFFERENCE is not available for this state. "
            "The runtime has deterministically established that structural changes remain on "
            "atoms shared with the target: "
            f"{boundary_gate.get('reason', '')} "
            "Return CONTINUE and name the next necessary chemical event. A boundary difference "
            "is an omitted species, never an unfinished principal product."
        )
    system = f"""{COMMON_MECHANISM_TASK}

{PGL_STATE_GUIDE}

You are the Global Planner performing a mandatory final-state audit after either a strict graph match or exhaustion of all planned chemical steps. Independently compare the complete current and target PGL systems and distinguish a missing real reaction event from an annotation-boundary difference.

{PLANNER_DECOMPOSITION_GUIDE}

Return CONFIRM_COMPLETE when the complete strict structures agree and no further electron-transfer event is chemically required.

Return CONFIRM_COMPLETE_WITH_BOUNDARY_DIFFERENCE only when the principal product already has the target connectivity, bond orders, formal charges, and chemically necessary hydrogens, and every remaining mismatch is solely an omitted species or representation boundary that requires no real electron transfer. Species that participated earlier but no longer participate may be absent from the target. A canonical example is H2 formed by NaH deprotonation: if H2 is absent from the target, ignore that omission rather than breaking H-H, deleting atoms, or inventing proton transfer. List every ignored mismatch in ignored_boundary_differences. Never use this decision to excuse an error on the principal product.

Return CONTINUE when any chemically necessary transformation remains. Ignore bookkeeping-only Species.role, Species.state, comments, molecule names, and ordering. Do not use a reward, gold action, annotated intermediate, or hidden future state.{gate_clause}

Output exactly one <completion_review>...</completion_review> valid JSON block with this schema:
{COMPLETION_SCHEMA}"""
    user = f"""<mechanism_task_description>
{task.task_description}
</mechanism_task_description>

<reaction_conditions>
{_json(task.conditions)}
</reaction_conditions>

<global_plan>
{_json(plan.to_dict())}
</global_plan>

<mechanism_step_idx>
{mechanism_step_idx}
</mechanism_step_idx>

<deterministic_strict_product_match>
{_json(strict_product_match)}
</deterministic_strict_product_match>

<all_planned_steps_exhausted>
{_json(plan_exhausted)}
</all_planned_steps_exhausted>

<target_state_pgl>
{task.target_state_pgl}
</target_state_pgl>

<current_state_pgl>
{current_state_pgl}
</current_state_pgl>

<executed_action_history>
{_history(history_actions_pgl)}
</executed_action_history>
{_required_changes_block(remaining_changes, "remaining_structural_changes")}
<boundary_completion_gate>
{_json(boundary_gate) if boundary_gate is not None else "not evaluated"}
</boundary_completion_gate>

Audit completion now. The deterministic signal is only context, not permission to skip your independent comparison."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def executor_messages(
    task: MechanismTaskContext,
    plan: GlobalPlan,
    directive: StepDirective,
    current_state_pgl: str,
    history_actions_pgl: List[str],
    legal_action_shapes: List[Dict[str, Any]],
    planner_feedback: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    system = f"""{COMMON_MECHANISM_TASK}

{PGL_STATE_GUIDE}

{PGL_ACTION_GUIDE}

You are the Local Executor. Convert the Planner's expected electron-transfer event into exactly one concrete PGL ReactionTransformation for the current ReactionState. Every atom and bond in the action must exist or be formable from the current state. Never invent atom labels.

If the directive cannot be realized, explain the conflict in <plan_objection> instead of forcing an invalid action. Output exactly three blocks in this order and no other text:
<event_summary>a concise description of the action actually proposed</event_summary>
<plan_objection></plan_objection>
<action>ReactionTransformation(...)</action>

If a real conflict prevents a reliable action, plan_objection may be non-empty and <action></action> may be empty. Do not output markdown fences or JSON."""
    user = f"""<mechanism_task_description>
{task.task_description}
</mechanism_task_description>

<reaction_conditions>
{_json(task.conditions)}
</reaction_conditions>

<complete_reaction_boundary>
Initial ReactionState:
{task.initial_state_pgl}

Target ReactionState:
{task.target_state_pgl}
</complete_reaction_boundary>

<planner_global_context>
{_json(directive.global_context)}
</planner_global_context>

<current_elementary_step>
{_json(directive.active_elementary_step)}
</current_elementary_step>

<planner_expected_event>
{directive.expected_electron_transfer_event}
</planner_expected_event>

<planner_constraints>
must_preserve: {_json(directive.must_preserve)}
avoid: {_json(directive.avoid)}
uncertainty: {directive.uncertainty or "None"}
</planner_constraints>

<completed_action_history>
{_history(history_actions_pgl)}
</completed_action_history>

<current_state_pgl>
{current_state_pgl}
</current_state_pgl>

<legal_local_action_shapes>
{_json(legal_action_shapes)}
</legal_local_action_shapes>

<planner_feedback_from_previous_attempt>
{_json(planner_feedback) if planner_feedback is not None else "None"}
</planner_feedback_from_previous_attempt>

Generate the corrected next action. No skills are available."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def review_messages(
    task: MechanismTaskContext,
    plan: GlobalPlan,
    directive: StepDirective,
    current_state_pgl: str,
    history_actions_pgl: List[str],
    proposal: ExecutorProposal,
    deterministic_validation: Dict[str, Any],
    previous_reviews: List[Dict[str, Any]],
    remaining_changes: Optional[str] = None,
    action_evidence: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    evidence_clause = ""
    if remaining_changes or action_evidence:
        evidence_clause = """

The runtime supplies two independent pieces of evidence that do not come from the plan you wrote. Use them, and state in chemical_assessment what each one implied.

1. remaining_structural_changes: the deterministic list of changes still separating the current state from the target. A proposal is suspect when it addresses none of them and is not a recognized activation, coordination, resonance, or proton-shuttle step that enables a later listed change.
2. action_evidence: which of those outstanding changes the proposed action actually touches. `addresses_any_required_change: false` is not by itself a rejection, but approving such a proposal requires naming, in chemical_assessment, the later listed change it enables. If you cannot name one, return REVISE.

Agreeing with your own plan is not evidence. The plan under review and the standard you are reviewing against were written by you, so plan agreement alone must never be the stated reason for APPROVE."""
    system = f"""{COMMON_MECHANISM_TASK}

{PGL_STATE_GUIDE}

{PGL_ACTION_GUIDE}

You are the Global Planner and semantic reviewer. Review the Executor's proposed PGL ReactionTransformation before execution. Judge alignment with the global mechanism, authoritative active elementary step, expected electron-transfer event, current PGL, and deterministic validation.

{PLANNER_DECOMPOSITION_GUIDE}{evidence_clause}

APPROVE only when the proposal implements exactly one event and fully completes the active elementary step without beginning a later step. Accept chemically equivalent PGL encodings and coupled transfers required by the current state. Do not use or assume any ground-truth action, annotated intermediate, or future state. If correction is possible under the current plan, return REVISE with precise feedback. If the active step combines sequential events or conflicts with the real state, return REPLAN.

Reject a proposal that invents an unsupported proton transfer or atom-deletion-like cleanup merely to reconcile a likely annotation boundary. Conversely, require a proton transfer when it is chemically necessary and a valid represented donor and acceptor exist.

Output exactly one <review>...</review> valid JSON block with this schema:
{REVIEW_SCHEMA}"""
    user = f"""<mechanism_task_description>
{task.task_description}
</mechanism_task_description>

<reaction_conditions>
{_json(task.conditions)}
</reaction_conditions>

<target_state_pgl>
{task.target_state_pgl}
</target_state_pgl>

<global_plan>
{_json(plan.to_dict())}
</global_plan>

<step_directive>
{_json(directive.to_dict())}
</step_directive>

<current_state_pgl>
{current_state_pgl}
</current_state_pgl>

<recent_executed_history>
{_history(history_actions_pgl)}
</recent_executed_history>

<executor_proposal>
event_summary: {proposal.event_summary}
plan_objection: {proposal.plan_objection or "None"}
action:
{proposal.action_pgl or "None"}
</executor_proposal>

<deterministic_validation>
{_json(deterministic_validation)}
</deterministic_validation>
{_required_changes_block(remaining_changes, "remaining_structural_changes")}
<action_evidence>
{_json(action_evidence) if action_evidence is not None else "not evaluated"}
</action_evidence>

<previous_reviews_for_this_state>
{_json(previous_reviews) if previous_reviews else "None"}
</previous_reviews_for_this_state>

Return APPROVE, REVISE, or REPLAN. Never compare with a gold answer."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def posthoc_gt_assessment_messages(
    task: MechanismTaskContext,
    initial_plan: GlobalPlan,
    predicted_actions_pgl: List[str],
    gold_actions_pgl: List[str],
    rollout_outcome: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Build a read-only evaluator prompt used only after rollout termination."""

    system = f"""You are a post-hoc chemical mechanism evaluator. The Planner-Executor rollout has already ended. Ground-truth information in this request is for evaluation only and must never be treated as guidance for generating or repairing the completed trajectory.

Make three independent judgments against the ordered ground-truth ReactionTransformations:

1. planner_global_mechanism_judgment: use the same broad semantic standard as a textbook mechanism assessment. Decide whether the initial Planner plan identified the reaction family, overall transformation, major mechanistic stages, and key intermediates. Minor wording and step-boundary differences do not make the global mechanism wrong.

2. planner_elementary_steps_judgment: evaluate only the Planner's elementary_steps under a strict normalized-event standard. Normalize both the natural-language plan and GT into ordered real chemical events, including reagent activation, coordination, proton transfer, bond formation, bond cleavage, bond-order change, leaving-group handling, intermediate formation/collapse, regeneration, and final neutralization. Return correct only if every GT event is covered in the same causal order and the plan contains no additional real chemical event. Textual splitting or merging is allowed: one GT event may span multiple plan items and one plan item may describe multiple adjacent GT events, but after normalization nothing may be added, omitted, duplicated, or reordered. Splitting/merging must not introduce a chemically distinct intermediate or erase a required event. Allow substitution of atom labels only when the atoms are genuinely symmetry- or chemical-environment-equivalent: same element, formal charge/valence environment, molecular role, and interchangeable local graph context. Do not accept merely choosing another atom of the same element. A plausible alternative mechanism that differs from the annotated event sequence is partially_correct, not correct. Judge the steps independently of whether the global explanation sounds right.

3. predicted_events_judgment: decide whether the accepted, executed ReactionTransformations reproduce the ordered GT elementary electron-transfer events. Do not infer correctness merely from reaching the target state.

For either strict event judgment, explicitly detect missing or extra activation, proton transfer, leaving-group departure, bond-order change, intermediate collapse, catalyst/reagent regeneration, or final neutralization. Use partially_correct when the core route is understood but at least one real event, order relation, non-equivalent atom, or intermediate differs; use incorrect when the core pathway is wrong or multiple essential events are absent. Write concise Chinese and keep the three assessment explanations separate.

Output exactly one <gt_assessment>...</gt_assessment> valid JSON block with this schema:
{POSTHOC_GT_ASSESSMENT_SCHEMA}"""
    user = f"""<mechanism_task_description>
{task.task_description}
</mechanism_task_description>

<reaction_conditions>
{_json(task.conditions)}
</reaction_conditions>

<initial_state_pgl>
{task.initial_state_pgl}
</initial_state_pgl>

<target_state_pgl>
{task.target_state_pgl}
</target_state_pgl>

<initial_planner_plan>
{_json(initial_plan.to_dict())}
</initial_planner_plan>

<executed_predicted_events>
{_history(predicted_actions_pgl)}
</executed_predicted_events>

<ground_truth_events>
{_history(gold_actions_pgl)}
</ground_truth_events>

<rollout_outcome>
{_json(rollout_outcome)}
</rollout_outcome>

Return the concise post-hoc comparison now."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
