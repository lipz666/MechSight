from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .planner_skills import pgl_structure_change_facts
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
)

9. The current ReactionState is an exact resource ledger. Every source sigma/pi bond must exist in the current state before the event. Never reuse a bond already broken or decreased by an accepted historical action, and never create a `new_sigma_bond` that already exists.
10. For a represented explicit H atom transferred from a donor to a specified acceptor, encode exactly two ordinary arrows in the same ReactionTransformation: (a) acceptor lone pair -> new acceptor-H bond with kind='bond_forming'; (b) donor-H sigma bond -> donor lone pair with kind='sigma_shift'. If the authoritative step instead releases a free proton/catalyst and names no acceptor, use only the donor-H sigma bond -> donor lone pair arrow and leave that H unbound; never invent water, solvent, or another spectator as an acceptor. Do not label these arrows kind='proton_transfer'. The environment reserves kind='proton_transfer' for implicit-H shorthand. Each explicit H must have at most one covalent partner after the event.
11. Do not add charge-neutralization, atom removal, bond cleanup, or a second copy of an earlier arrow unless that change is part of the authoritative active step and its source exists in the current state.
12. Encode the complete arrow chain of the elementary event, not merely its net graph edit. When attack, elimination, rearrangement, or collapse propagates through a conjugated system, include every contiguous pi-bond shift needed to carry the electron pair to its chemically valid sink. A beta attack on an alpha,beta-unsaturated carbonyl, for example, requires the full conjugated electron relay rather than placing an unexplained lone pair on the adjacent carbon. Do not omit an arrow merely because a later resonance-restoration arrow makes one bond order unchanged in the net post-state.
13. Any resonance stabilization explicitly named by the active step must be encoded as electron transfer, not left as prose. In an enol ether, protonation of the alkene that forms an oxocarbenium includes donation of the adjacent ether-oxygen lone pair into the C-O pi bond; subsequent nucleophilic attack at that carbon includes shifting the C-O pi electrons back to oxygen. Apply the same principle to iminium, allylic, and conjugated intermediates."""


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


FORWARD_TASK_NOTE = """Forward-prediction setting: no target ReactionState is provided. You are given only the complete initial ReactionState, which already contains every reactant and reagent that takes part (including reagents introduced by later operations such as workup), and the reaction conditions. Wherever these instructions mention the target, the target PGL, or agreement with the target, use instead the final products you predict from the structures and conditions. Carry out every operation the conditions describe, in order, until the principal products are stable species appropriate to the final workup. Reagents or spectators that do not react simply remain in the state; never delete atoms or invent cleanup events for them."""


def _is_forward(task: MechanismTaskContext) -> bool:
    """Experiment B withholds the product side; see ChemistryBridge.task_context."""

    return (task.reaction_metadata or {}).get("input_mode") == "forward"


def _task_framing(task: MechanismTaskContext) -> str:
    if _is_forward(task):
        return f"{COMMON_MECHANISM_TASK}\n\n{FORWARD_TASK_NOTE}"
    return COMMON_MECHANISM_TASK


def _target_block(task: MechanismTaskContext) -> str:
    if _is_forward(task):
        return (
            "<target_state_pgl>\nNot provided. Predict the products from the initial "
            "state and the reaction conditions.\n</target_state_pgl>"
        )
    return f"<target_state_pgl>\n{task.target_state_pgl}\n</target_state_pgl>"


GT_BOND_CHANGE_NOTE = """Ground-truth format for this dataset: the annotation records only the states between elementary steps, not curved arrows. Each ground-truth step is therefore given as its net bond formations, bond cleavages, bond-order changes, and formal-charge changes between consecutive annotated states, using the atom labels of the initial PGL. Treat each step's change list as that step's elementary event. Judge whether the Planner steps and the executed ReactionTransformations produce the same changes in the same causal order, applying every equivalence class above to the changes. Never penalize a choice of arrow source or direction that a change list cannot determine, and never require an arrow the change list does not imply."""


def _gt_format_note(gt_format: str) -> str:
    if gt_format == "arrows":
        return ""
    if gt_format == "bond_changes":
        return f"\n\n{GT_BOND_CHANGE_NOTE}"
    raise ValueError(f"unsupported ground-truth format: {gt_format}")


def _gt_block(gold: List[str], gt_format: str) -> str:
    # Rendered change lists already carry their own step headings.
    return _history(gold) if gt_format == "arrows" else "\n\n".join(gold)


def _forward_completion_messages(
    task: MechanismTaskContext,
    plan: GlobalPlan,
    mechanism_step_idx: int,
    current_state_pgl: str,
    history_actions_pgl: List[str],
) -> tuple[str, str]:
    """Completion audit with no target: decide from chemistry and conditions alone."""

    system = f"""{_task_framing(task)}

{PGL_STATE_GUIDE}

You are the Global Planner performing a mandatory final-state audit after every planned elementary step has been executed. No target state exists. Decide from chemistry alone whether the mechanism is finished under the stated conditions.

Return CONFIRM_COMPLETE when every operation the conditions describe has taken effect, no reactive intermediate that the conditions would still transform remains, and the principal products are stable species appropriate to the final workup. In that case set all four structural fields to true; here they mean that the connectivity, bond orders, formal charges, and hydrogens of the current principal products agree with the products you predict.

Return CONTINUE when any chemically necessary transformation remains, and state it in required_next_event. Do not return CONFIRM_COMPLETE_WITH_BOUNDARY_DIFFERENCE. Unreacted reagents, spectators, and byproducts remaining in the state are not a reason to continue.

Output exactly one <completion_review>...</completion_review> valid JSON block with this schema:
{COMPLETION_SCHEMA}"""
    user = f"""<mechanism_task_description>
{task.task_description}
</mechanism_task_description>

<reaction_conditions>
{_json(task.conditions)}
</reaction_conditions>

<global_plan>
{_json(_plan_prompt_payload(plan))}
</global_plan>

<mechanism_step_idx>
{mechanism_step_idx}
</mechanism_step_idx>

<current_state_pgl>
{current_state_pgl}
</current_state_pgl>

<executed_action_history>
{_history(history_actions_pgl)}
</executed_action_history>

Audit completion now."""
    return system, user


def _plan_prompt_payload(plan: GlobalPlan) -> Dict[str, Any]:
    """Keep retrieval telemetry in saved state, not downstream reasoning prompts."""

    payload = plan.to_dict()
    payload.pop("planner_skill_selection", None)
    return payload


def _history(history_actions_pgl: List[str]) -> str:
    if not history_actions_pgl:
        return "None"
    return "\n\n".join(
        f"Historical ReactionTransformation {index}:\n{action}"
        for index, action in enumerate(history_actions_pgl)
    )


def planner_reaction_feature_messages(
    task: MechanismTaskContext,
) -> List[Dict[str, str]]:
    """Infer reusable reaction features before exposing any skill metadata."""

    system = f"""You are the structure-understanding stage of mechanism-skill retrieval. No skill catalog, skill name, reaction label, or gold mechanism is available.

Compare the complete initial and target PGL states and produce a compact chemical abstraction, not an exhaustive graph listing. Identify functional-group roles, reagent motifs, the principal center-aligned net transformation, plausible reaction-family hypotheses, and genuine ambiguities. Use the mechanical PGL delta only to locate changes; interpret chemically equivalent resonance/Kekule forms flexibly. Do not infer a cleanup reaction from components or explicit-H labels that merely cross the annotation boundary.

Output exactly one <reaction_features>...</reaction_features> valid JSON block:
{{
  "reaction_center_summary": "semantic description of the principal changed center",
  "initial_functional_roles": ["substrate and reagent roles supported by initial PGL"],
  "core_net_changes": ["chemically grouped center-aligned changes"],
  "candidate_families": ["one or more structure-derived hypotheses"],
  "distinguishing_evidence": ["evidence that separates close alternatives"],
  "ambiguities": ["uncertainties caused by representation or scope"]
}}

Do not plan elementary steps and do not mention or guess a skill name."""
    user = f"""<mechanism_task>
{task.task_description}
</mechanism_task>

<reaction_conditions>
{_json(task.conditions)}
</reaction_conditions>

<initial_state_pgl>
{task.initial_state_pgl}
</initial_state_pgl>

{_target_block(task)}

<observed_pgl_delta>
{_json(pgl_structure_change_facts(task.initial_state_pgl, task.target_state_pgl))}
</observed_pgl_delta>

Extract reaction features now. Do not plan the mechanism."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def planner_skill_selection_messages(
    task: MechanismTaskContext,
    catalog: List[Dict[str, str]],
    max_skills: int,
    reaction_features: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """Expose only skill triggers; full skill contents remain undisclosed."""

    feature_guidance = (
        "A separately generated semantic feature summary is supplied as a fallible hint. "
        "Cross-check and correct it using the original PGL; never let its family label "
        "override literal structure."
        if reaction_features is not None
        else
        "No prior family hypothesis is supplied. Infer candidates directly from the original "
        "PGL and mechanical delta so that an earlier classification cannot anchor retrieval."
    )
    system = f"""You are a mechanism-skill selector operating before mechanism planning. This is an isolated retrieval request; no skill body, mechanism plan, reaction label, or gold path is available.

{feature_guidance} The catalog contains only trigger descriptions; it does not contain a gold answer.

Before deciding, audit every catalog entry against an explicit structural checklist: (1) required initial substrate motif, (2) required reagent motif, (3) bonds made/broken or changed in order at the same center, and (4) exclusions stated by when_to_use. Do not stop after assigning a tentative reaction-family name, and do not reject an exact structural match merely because the complete molecule is complex or the transformation is intramolecular. PGL often uses alternating single/double Kekule bonds rather than an `aromatic` bond kind; determine whether a carbon is in a closed conjugated ring from connectivity. Conversely, an sp3 carbon with two represented hydrogens and only one skeletal neighbor is not an aromatic ipso carbon.

Use these high-value distinctions when applicable: C=C-C(=O) is an alpha,beta-unsaturated carbonyl, whereas C=C-O is an enol ether; a metal-bound former carbonyl oxygen may remain in the system even after the substrate C-O bond has been cleaved; and an N=C=O or N=C=S cumulene must be recognized directly from its two adjacent double bonds.

This is a high-recall candidate stage, not the final trigger decision. Select at most {max_skills} skills from this catalog batch. Return a candidate when the available structural evidence plausibly supports its central motif and net transformation; the later verifier will reject false positives. Omit a skill only when its required center is clearly contradicted, not merely because another family currently seems more likely. All required substrate motifs and net changes must plausibly map to the same reacting center. In each selection reason, identify the mapped reactant center and corresponding target change. When two skills describe complementary phases or two literal functional-group readings remain plausible, use both available candidate slots. Do not let the first family hypothesis suppress a second plausible trigger. Never select by reaction id, dataset identity, atom numbering pattern, or an unsupported reaction-family guess.

Output exactly one <skill_selection>...</skill_selection> valid JSON block:
{{
  "inferred_reaction_pattern": "concise structure-derived hypothesis",
  "structural_evidence": ["reactant/product evidence without hidden metadata"],
  "selected_skills": [{{"name": "catalog skill name", "reason": "why its trigger matches"}}],
  "no_match_reason": "required when selected_skills is empty; otherwise empty"
}}"""
    feature_block = (
        f"<reaction_features>\n{_json(reaction_features)}\n</reaction_features>\n\n"
        if reaction_features is not None
        else ""
    )
    user = f"""<mechanism_task>
{task.task_description}
</mechanism_task>

{feature_block}<initial_state_pgl>
{task.initial_state_pgl}
</initial_state_pgl>

{_target_block(task)}

<observed_pgl_delta>
{_json(pgl_structure_change_facts(task.initial_state_pgl, task.target_state_pgl))}
</observed_pgl_delta>

<planner_skill_catalog>
{_json(catalog)}
</planner_skill_catalog>

Select relevant Planner skills now. Do not plan the mechanism yet."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def planner_skill_verification_messages(
    task: MechanismTaskContext,
    candidate_catalog: List[Dict[str, str]],
    candidate_selection: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Audit candidate triggers before any skill body is disclosed."""

    system = """You are the conservative verification stage of mechanism-skill retrieval. No skill body is available. Decide only whether each candidate's complete when_to_use condition is proven by the supplied initial-to-target structural change.

For each candidate, independently locate the actual changed center and map every required reactant motif, reagent, and target signature to that same center. Use only observed_pgl_delta.shared_atom_changes as direct reaction-center evidence. Treat annotation_boundary_changes as scope uncertainty, not proof of transformation, and never require cleanup of a component merely because it disappears. Reject the candidate if a required target change is absent, occurs only at a different remote functional group, or is merely a plausible future step beyond the supplied target. A reaction-family guess or the presence of a reagent alone is insufficient. Treat contradictions in the candidate's own structural evidence as grounds for rejection.

Verify literal connectivity rather than relying on the candidate's family label. Look for both supporting and contradicting evidence in the original PGL; reject a candidate when its support depends on a bond, ring membership, reagent role, or target change that the PGL does not establish. A bond explicitly present in PGL cannot be cited as absent; internally reread any such claim once before deciding. When a trigger depends on the size of a newly closed local ring, identify the two closure atoms and inspect the shortest simple path between them that excludes the newly attached center. Do not use the first path encountered or a longer alternate path to reject the trigger; fused, bridged, and pre-existing larger rings may provide multiple paths. Kekule single/double bonds may encode an aromatic ring, but an sp3 carbon with represented hydrogens must not be called an aromatic ipso carbon. A former substrate oxygen can remain as a metal-bound or byproduct atom after its substrate C-O bond breaks; this satisfies a trigger requiring loss of oxygen from the substrate. Distinguish C=C-C(=O) conjugate addition from C=C-O enol-ether addition. If the required initial motif and the exact center-aligned net changes are present, accept even when the molecule contains additional rings or the reaction is intramolecular.

Output exactly one <skill_verification>...</skill_verification> valid JSON block:
{
  "skill_checks": [
    {
      "name": "candidate name",
      "trigger_satisfied": true,
      "supporting_evidence": ["original-PGL evidence"],
      "contradicting_evidence": [],
      "bond_checks": [
        {"state": "initial|target", "atoms": ["A1", "A2"], "exists": true, "kind": "single|double|triple|aromatic|null", "purpose": "trigger clause supported by this bond fact"}
      ],
      "path_checks": [
        {"state": "initial|target", "atoms": ["A1", "A2", "A3", "A1"], "purpose": "complete closed path supporting a claimed ring"}
      ],
      "reason": "center-aligned verification or specific failed condition"
    }
  ]
}
Check every candidate exactly once. For every connectivity or bond-order clause used to satisfy when_to_use, add a separate bond_checks entry; include both initial and target facts when the trigger depends on a bond change. Whenever acceptance depends on a ring, include the shortest relevant complete closed path in path_checks with the first atom repeated at the end; do not abbreviate the middle of the path. Before rejecting a ring-dependent trigger for the wrong ring size, explicitly check for a shorter alternate path between the closure atoms. These are targeted evidence queries, not an exhaustive neighbor list. Set trigger_satisfied=true only when the required support is present, the targeted checks cover the distinguishing structural claims, and contradicting_evidence is empty."""
    user = f"""<mechanism_task>
{task.task_description}
</mechanism_task>

<initial_state_pgl>
{task.initial_state_pgl}
</initial_state_pgl>

{_target_block(task)}

<observed_pgl_delta>
{_json(pgl_structure_change_facts(task.initial_state_pgl, task.target_state_pgl))}
</observed_pgl_delta>

<candidate_skill_triggers>
{_json(candidate_catalog)}
</candidate_skill_triggers>

<candidate_retrieval_result>
{_json(candidate_selection)}
</candidate_retrieval_result>

Verify trigger satisfaction now. Do not plan the mechanism and do not infer unshown later products."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def planner_skill_evidence_repair_messages(
    messages: List[Dict[str, str]],
    audited_checks: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Request one focused repair when cited PGL facts fail deterministic audit."""

    repaired = [dict(message) for message in messages]
    repaired.append(
        {
            "role": "user",
            "content": f"""<deterministic_evidence_audit>
{_json(audited_checks)}
</deterministic_evidence_audit>

Your previous positive trigger decision cited at least one bond or path fact that is false in the named state. Re-read the original PGL and return a complete corrected <skill_verification> block. Preserve a positive decision only if valid targeted evidence still proves every trigger clause; otherwise reject it. For a ring created by target-state bonds, check the completed ring in the target state. When multiple paths connect the closure atoms, use the shortest relevant local path rather than a longer alternate route. Do not enumerate all neighbors.""",
        }
    )
    return repaired


def planner_skill_resolution_messages(
    reaction_features: Dict[str, Any],
    accepted_catalog: List[Dict[str, str]],
    verification: List[Dict[str, Any]],
    max_skills: int,
) -> List[Dict[str, str]]:
    """Remove redundant generic skills after every trigger has been verified."""

    system = f"""You are the final nonredundancy resolver for mechanism skills. Every supplied candidate has independently passed its when_to_use verification. Select at most {max_skills} names and choose the smallest set needed for guidance.

When two accepted skills cover the same reaction stage, retain the more structurally specific trigger and remove the generic one. Retain multiple skills only when their verified evidence shows that they cover genuinely complementary activation and product-forming phases. Do not see or infer skill bodies, mechanism steps, reaction IDs, or gold data.

Output exactly one <skill_resolution>...</skill_resolution> valid JSON block:
{{
  "selected_skill_names": ["accepted skill name"],
  "reason": "specificity or complementarity decision"
}}"""
    user = f"""<reaction_features>
{_json(reaction_features)}
</reaction_features>

<accepted_skill_triggers>
{_json(accepted_catalog)}
</accepted_skill_triggers>

<trigger_verification>
{_json(verification)}
</trigger_verification>

Resolve the minimal nonredundant skill set now."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _planner_skill_guidance(skills: Optional[List[Dict[str, str]]]) -> str:
    if not skills:
        return ""
    rendered = "\n\n".join(
        f"<planner_skill name={_json(item['name'])}>\n{item['content']}\n</planner_skill>"
        for item in skills
    )
    return f"""

Retrieved Planner skills follow. A separate structure-only retrieval call has already matched their when-to-use conditions. They are reusable mechanistic knowledge, not ground truth for this sample.

Before writing the plan, internally map every abstract role in each retrieved skill (substrate, reagent, nucleophile, electrophile, leaving group, catalyst, and relevant intermediate) to the supplied PGL. Use the skill's Mechanistic Strategy as the default ordered causal backbone. Cover every applicable Coverage Check in global_mechanism and elementary_steps; do not silently omit, reverse, or replace a required stage with a generic reaction template. You may adapt concerted/stepwise boundaries and current atom labels. Deviate only when the actual PGL directly contradicts a skill requirement; in that case derive the structure-supported alternative and state the conflict in uncertainties. Do not mention skill names in the plan.

<retrieved_planner_skills>
{rendered}
</retrieved_planner_skills>"""


def create_plan_messages(
    task: MechanismTaskContext,
    planner_skills: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    system = f"""{_task_framing(task)}

{PGL_STATE_GUIDE}

You are the Global Planner. Compare the complete initial and target PGL structures yourself and infer a chemically plausible global mechanism under the stated conditions.

{PLANNER_DECOMPOSITION_GUIDE}
{_planner_skill_guidance(planner_skills)}

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

{_target_block(task)}

Generate the GlobalPlan now. {"Use only the retrieved Planner skills whose triggers remain supported by the PGL." if planner_skills else "No Planner skill was selected; solve from the structures without skills."} Do not use any precomputed target-change summary."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def replan_messages(
    task: MechanismTaskContext,
    plan: GlobalPlan,
    current_state_pgl: str,
    history_actions_pgl: List[str],
    reason: Any,
    planner_skills: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    system = f"""{_task_framing(task)}

{PGL_STATE_GUIDE}

You are the Global Planner revising only the unexecuted remainder of a mechanism. Executed history and the real current ReactionState are immutable facts. Build a new natural-language global mechanism and ordered remaining elementary steps from the current state to the original target. The first returned step must be executable next from the supplied current state.

{PLANNER_DECOMPOSITION_GUIDE}
{_planner_skill_guidance(planner_skills)}

Output exactly one <plan>...</plan> valid JSON block using this schema:
{PLAN_SCHEMA}"""
    user = f"""<original_task>
{task.task_description}
</original_task>

<reaction_conditions>
{_json(task.conditions)}
</reaction_conditions>

<previous_global_plan>
{_json(_plan_prompt_payload(plan))}
</previous_global_plan>

<executed_action_history>
{_history(history_actions_pgl)}
</executed_action_history>

<current_state_pgl>
{current_state_pgl}
</current_state_pgl>

{_target_block(task)}

<replan_reason>
{_json(reason)}
</replan_reason>

Replan the remaining mechanism. Use retrieved skill guidance only when it remains supported by the current state. Do not use gold actions or future annotated states."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def directive_messages(
    task: MechanismTaskContext,
    plan: GlobalPlan,
    active_elementary_step: Dict[str, str],
    mechanism_step_idx: int,
    current_state_pgl: str,
    history_actions_pgl: List[str],
    last_env_feedback: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    system = f"""{_task_framing(task)}

{PGL_STATE_GUIDE}

You are the Global Planner grounding the one authoritative elementary step selected by the runtime. The runtime, not the model, owns step advancement. One successfully executed, non-repeated environment action completes exactly one elementary step and the runtime automatically selects the next step.

{PLANNER_DECOMPOSITION_GUIDE}

Return directive_decision as exactly EXECUTE or REPLAN, with no explanation inside that field. Echo the supplied active_elementary_step without changing its id or description. Choose EXECUTE only when that step can be represented as exactly one next ReactionTransformation from the current PGL. Choose REPLAN if the authoritative step is no longer executable or improperly combines multiple sequential events. If a previous environment edit was rejected, the same elementary step remains active. If a remaining difference is only a likely transient-species or hydrogen-annotation mismatch, report it in uncertainty and do not direct the Executor to invent a deletion or unsupported proton transfer.

When returning EXECUTE, expected_electron_transfer_event must enumerate the complete ordered arrow chain using current atom labels: identify every electron source, destination, bond formed, bond broken or decreased, and coupled pi shift. Do not summarize a conjugated attack as only "attack and form an enolate"; spell out the contiguous relay that makes the stated intermediate.

If the authoritative step names a resonance-stabilized intermediate, translate that stabilization into explicit arrows in expected_electron_transfer_event. Check adjacent heteroatoms and pi systems; for an enol-ether-derived oxocarbenium, require oxygen lone-pair donation into the C-O pi bond, and require the reverse pi shift when a nucleophile subsequently attacks that carbon.

Do not output a concrete ReactionTransformation. Output exactly one <directive>...</directive> valid JSON block with this schema:
{DIRECTIVE_SCHEMA}

Environment feedback contains only observable edit/state information; no reward or annotated-path score is available."""
    user = f"""<global_plan>
{_json(_plan_prompt_payload(plan))}
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

{_target_block(task)}

<current_state_pgl>
{current_state_pgl}
</current_state_pgl>

<executed_action_history>
{_history(history_actions_pgl)}
</executed_action_history>

<last_environment_feedback>
{_json(last_env_feedback) if last_env_feedback is not None else "None"}
</last_environment_feedback>

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
) -> List[Dict[str, str]]:
    system = f"""{_task_framing(task)}

{PGL_STATE_GUIDE}

You are the Global Planner performing a mandatory final-state audit after either a strict graph match or exhaustion of all planned chemical steps. Independently compare the complete current and target PGL systems and distinguish a missing real reaction event from an annotation-boundary difference.

{PLANNER_DECOMPOSITION_GUIDE}

Return CONFIRM_COMPLETE when the complete strict structures agree and no further electron-transfer event is chemically required.

Return CONFIRM_COMPLETE_WITH_BOUNDARY_DIFFERENCE only when the principal product already has the target connectivity, bond orders, formal charges, and chemically necessary hydrogens, and every remaining mismatch is solely an omitted species or representation boundary that requires no real electron transfer. Species that participated earlier but no longer participate may be absent from the target. A canonical example is H2 formed by NaH deprotonation: if H2 is absent from the target, ignore that omission rather than breaking H-H, deleting atoms, or inventing proton transfer. List every ignored mismatch in ignored_boundary_differences. Never use this decision to excuse an error on the principal product.

Return CONTINUE when any chemically necessary transformation remains. Ignore bookkeeping-only Species.role, Species.state, comments, molecule names, and ordering. Do not use a reward, gold action, annotated intermediate, or hidden future state.

Output exactly one <completion_review>...</completion_review> valid JSON block with this schema:
{COMPLETION_SCHEMA}"""
    user = f"""<mechanism_task_description>
{task.task_description}
</mechanism_task_description>

<reaction_conditions>
{_json(task.conditions)}
</reaction_conditions>

<global_plan>
{_json(_plan_prompt_payload(plan))}
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

{_target_block(task)}

<current_state_pgl>
{current_state_pgl}
</current_state_pgl>

<executed_action_history>
{_history(history_actions_pgl)}
</executed_action_history>

Audit completion now. The deterministic signal is only context, not permission to skip your independent comparison."""
    if _is_forward(task):
        system, user = _forward_completion_messages(
            task, plan, mechanism_step_idx, current_state_pgl, history_actions_pgl
        )
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
    system = f"""{_task_framing(task)}

{PGL_STATE_GUIDE}

{PGL_ACTION_GUIDE}

You are the Local Executor. Convert the Planner's expected electron-transfer event into exactly one concrete PGL ReactionTransformation for the current ReactionState. Every atom and bond in the action must exist or be formable from the current state. Never invent atom labels.

Treat completed_action_history as a ledger of consumed electron sources and current_state_pgl as authoritative. Before answering, audit each proposed source against the current bond list, then mentally apply all transfers once. The result must fully complete the current elementary step, must change the state, and must not begin a later step. In particular, never repeat one half of a proton transfer that a prior accepted action already completed.

For deprotonation, obey the named acceptor exactly. If the Planner describes regeneration/release of a free proton and does not name an acceptor, emit only heterolytic donor-H cleavage; do not route the proton onto a nearby water, solvent, leaving group, or catalyst merely because it could act as a base.

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
{'Not provided (forward prediction).' if _is_forward(task) else task.target_state_pgl}
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
) -> List[Dict[str, str]]:
    system = f"""{_task_framing(task)}

{PGL_STATE_GUIDE}

{PGL_ACTION_GUIDE}

You are the Global Planner and semantic reviewer. Review the Executor's proposed PGL ReactionTransformation before execution. Judge alignment with the global mechanism, authoritative active elementary step, expected electron-transfer event, current PGL, and deterministic validation.

{PLANNER_DECOMPOSITION_GUIDE}

APPROVE only when the proposal implements exactly one event and fully completes the active elementary step without beginning a later step. The deterministic validation includes a transactional preview_state_pgl and preview_delta when parsing succeeds. Compare that predicted post-state against the current state, active step, and history. Reject if the preview leaves the step half-complete, reuses an absent/consumed bond, gives an explicit H multiple partners, produces no state change, or includes an unrelated cleanup. Accept chemically equivalent PGL encodings and coupled transfers required by the current state. Do not use or assume any ground-truth action, annotated intermediate, or future state. If correction is possible under the current plan, return REVISE with precise feedback. If the active step combines sequential events or conflicts with the real state, return REPLAN.

Electron-event consistency is stricter than net-state similarity: verify that the proposal contains the full chemically connected arrow path described by expected_electron_transfer_event. For conjugated additions, eliminations, and rearrangements, reject a shortcut that reaches a plausible charge pattern while omitting intervening pi shifts.

Reject a proposal that mentions a resonance-stabilized oxocarbenium, iminium, or allylic intermediate but omits the corresponding lone-pair/pi-bond arrow from the concrete action. Also reject the subsequent attack if it fails to move that stabilizing pi pair back to its heteroatom or other chemically valid sink.

For proton events, reject an acceptor that is not specified or required by the active step. In particular, when the step calls for release/regeneration of a free proton, an otherwise plausible transfer to water is an extra event and must be revised to donor-H cleavage only.

Reject a proposal that invents an unsupported proton transfer or atom-deletion-like cleanup merely to reconcile a likely annotation boundary. Conversely, require a proton transfer when it is chemically necessary and a valid represented donor and acceptor exist.

Output exactly one <review>...</review> valid JSON block with this schema:
{REVIEW_SCHEMA}"""
    user = f"""<mechanism_task_description>
{task.task_description}
</mechanism_task_description>

<reaction_conditions>
{_json(task.conditions)}
</reaction_conditions>

{_target_block(task)}

<global_plan>
{_json(_plan_prompt_payload(plan))}
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
    gt_format: str = "arrows",
) -> List[Dict[str, str]]:
    """Build a read-only evaluator prompt used only after rollout termination."""

    system = f"""You are a post-hoc chemical mechanism evaluator. The Planner-Executor rollout has already ended. Ground-truth information in this request is for evaluation only and must never be treated as guidance for generating or repairing the completed trajectory.

Make three independent judgments against the ordered ground-truth ReactionTransformations:

1. planner_global_mechanism_judgment: use the same broad semantic standard as a textbook mechanism assessment. Decide whether the initial Planner plan identified the reaction family, overall transformation, major mechanistic stages, and key intermediates. Minor wording and step-boundary differences do not make the global mechanism wrong.

2. planner_elementary_steps_judgment: evaluate only the Planner's elementary_steps under a strict normalized-event standard. Normalize both the natural-language plan and GT into ordered real chemical events, including reagent activation, coordination, proton transfer, bond formation, bond cleavage, bond-order change, leaving-group handling, intermediate formation/collapse, regeneration, and final neutralization. Return correct only if every GT event is covered in the same causal order and the plan contains no additional net chemical event. Textual splitting or merging is allowed: one GT event may span multiple plan items and one plan item may describe multiple adjacent GT events, but after normalization nothing may be added, omitted, duplicated, or reordered. Apply the mandatory equivalence classes below before deciding that an intermediate or event is additional. Allow substitution of atom labels only when the atoms are genuinely symmetry- or chemical-environment-equivalent: same element, formal charge/valence environment, molecular role, and interchangeable local graph context. Do not accept merely choosing another atom of the same element. A plausible alternative mechanism that still differs after all mandatory equivalence normalization is partially_correct, not correct. Judge the steps independently of whether the global explanation sounds right.

3. predicted_events_judgment: decide whether the accepted, executed ReactionTransformations reproduce the ordered GT elementary electron-transfer events. Do not infer correctness merely from reaching the target state.

For predicted_events_judgment, normalize adjacent textual/action splitting or merging exactly as for Planner steps: it is acceptable when the same complete ordered chemical transformation remains, with no extra, missing, duplicated, or reordered net event after applying all equivalence classes below. Compare chemically meaningful electron flow and cumulative state changes rather than the raw number of ReactionTransformation blocks or arrows.

Mandatory equivalence classes for both strict judgments:

A. Concerted-versus-stepwise equivalence. Treat a concerted GT event and a sequence of adjacent predicted events as equivalent when concatenating the adjacent electron flow gives the same bond formation, bond cleavage, bond-order/charge changes, causal order, and catalyst balance. The reverse normalization also applies when the prediction combines adjacent GT events. Do not penalize protonation followed by attack versus coupled proton-assisted attack, separate C-C then O-P bond formation versus an asynchronous coupled formation, or explicit aromaticity/resonance restoration versus the same relay compressed into one block, provided that no net event is gained or lost.

B. Reasonable transient-intermediate equivalence. A chemically standard short-lived intermediate may be explicit on one side and implicit on the other when it only resolves the same local mechanistic stage into adjacent steps and then collapses to the same next chemically meaningful state. Examples include a resonance-stabilized acylium/oxocarbenium contributor, a Wheland complex, a tetrahedral adduct, a betaine preceding ring closure, or an explicitly protonated form. Such an intermediate is not an additional real event merely because GT compresses it. It is non-equivalent only if it creates a different branch, changes which non-equivalent site reacts, requires an additional reagent transformation, changes the causal order of net events, or persists to alter a later chemically meaningful state.

C. Acid-base boundary equivalence. Treat H3O+ and H2O + H+ as equivalent catalyst-boundary representations, and likewise allow an explicit water/solvent/base proton acceptor when GT records only cleavage of the substrate X-H bond and omits the acceptor. The principal substrate protonation state, heavy-atom chemistry, net catalyst regeneration, and total chemically relevant proton balance must agree. Do not count formation of an explicit O-H bond on water as an extra proton-transfer event when it merely completes the same deprotonation encoded by GT. This allowance does not excuse a proton moved between non-equivalent substrate sites, a missing chemically necessary protonation/deprotonation, or failure to regenerate the acid/base catalyst.

D. Delocalized-pi/resonance equivalence. Within the same aromatic or conjugated electron system, allow alternative Kekule/resonance localization and the use of a different adjacent pi bond as the drawn electron source when the same labeled atom is the actual bond-forming or bond-breaking site and the cumulative relay restores the same connectivity, bond orders, formal charges, and aromaticity. The neighboring atoms that localize the drawn pi bond need not themselves be symmetry-equivalent because the electrons are delocalized. This does not allow substitution at a different labeled ring position, migration to a different conjugated system, or a different persistent regioisomer.

Explicit proton labels require special normalization because mechanism annotations may introduce, omit, or renumber transient H atoms. Treat two H labels as equivalent only when they occupy the same chemically interchangeable proton pool at that stage (for example, the two hydrogens on the same water molecule after a proton relay), the same heavy-atom donor/acceptor process occurs, and no distinct isotope/site history is chemically required. A GT H label absent from the supplied initial state is not by itself evidence that an otherwise identical proton-transfer event is wrong. This allowance never applies to non-equivalent heavy atoms or to a missing/extra proton-transfer event.

Mandatory decision rule: if every observed difference is fully explained by concerted/stepwise decomposition, a reasonable transient intermediate, acid-base boundary representation, adjacent action splitting/merging, delocalized-pi/resonance placement, or chemically interchangeable transient-H relabeling, return `correct`. Never return `partially_correct` or `incorrect` merely because the normalized-equivalent execution uses a different number of ReactionTransformation blocks, makes a standard transient intermediate explicit, restores a resonance bond in an adjacent block, draws an aromatic attack from another adjacent pi bond while keeping the same attacked atom, uses H3O+ instead of H2O + H+, supplies the omitted proton acceptor, or selects an interchangeable H label. Use `partially_correct` only when at least one chemically real net event, non-equivalent reacting site, causal order, persistent intermediate, or catalyst balance remains different after normalization, and name that non-equivalent difference explicitly.

For either strict event judgment, explicitly detect missing or extra activation, proton transfer, leaving-group departure, bond-order change, intermediate collapse, catalyst/reagent regeneration, or final neutralization. Use partially_correct when the core route is understood but at least one real event, order relation, non-equivalent atom, or intermediate differs; use incorrect when the core pathway is wrong or multiple essential events are absent. Write concise Chinese and keep the three assessment explanations separate.{_gt_format_note(gt_format)}

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

{_target_block(task)}

<initial_planner_plan>
{_json(_plan_prompt_payload(initial_plan))}
</initial_planner_plan>

<executed_predicted_events>
{_history(predicted_actions_pgl)}
</executed_predicted_events>

<ground_truth_events>
{_gt_block(gold_actions_pgl, gt_format)}
</ground_truth_events>

<rollout_outcome>
{_json(rollout_outcome)}
</rollout_outcome>

Return the concise post-hoc comparison now."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
