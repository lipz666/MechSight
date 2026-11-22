"""Verbatim prompt templates from Supporting Information Section S12.

Every template below matches the exact strings evaluated in the manuscript and
Supporting Information (Sections S12.1-S12.8).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import ReactionCase

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"

SYSTEM_PROMPT = (
    "You are an expert organic chemist who writes rigorous stepwise arrow-pushing mechanisms."
)


def load_omebench_prompt(kind: str = "default") -> str:
    """Load the official oMeBench prompt template (``'default'`` or ``'cot'``)."""
    name = "omebench_default.txt" if kind == "default" else "omebench_cot.txt"
    return (_ASSETS_DIR / name).read_text()


# Section S12.1: MechSight-Auto hidden-intermediate-directed pathway enumeration
AUTO_ENUMERATE_BLOCK = (
    "\n\n## Different task for this reaction\n\n"
    "Instead of a single mechanism, write THREE mechanistically distinct candidate mechanisms for the "
    "reaction below. They must differ in their intermediates. Include the most direct pathway, and consider "
    "pathways through short-lived species that do not appear in the overall equation: bonds that form and "
    "later break again, rearrangements, onium or ylide intermediates, tetrahedral adducts, neighbouring-group "
    "participation, catalyst- or reagent-bound species, pericyclic versus stepwise courses. Reason first. Then "
    "give each candidate as a JSON list in the format above, between [PATH1] and [/PATH1], [PATH2] and "
    "[/PATH2], [PATH3] and [/PATH3].\n\n## Input"
)

# Section S12.2: MechSight-Auto targeted intermediate completion at flagged shortcuts
AUTO_COMPLETE_TEMPLATE = (
    "Reactants and reagents (SMILES): {reactants}\n"
    "Products (SMILES): {products}\n"
    "Reaction conditions: {conditions}\n\n"
    "Candidate mechanisms for this reaction (intermediate SMILES after each step). An automatic atom-mapping "
    "check marks steps where several heavy-atom bonds change at once; such a step usually hides one or more "
    "short-lived intermediates (a tetrahedral adduct, an onium or ylide, a zwitterion, a rearranging cation, "
    "a catalyst- or reagent-bound species, ...), unless it is a genuinely concerted (e.g. pericyclic) step.\n\n"
    "{cands}\n\n"
    "For each flagged step, decide whether it hides intermediates. If it does, write the missing intermediate "
    "species that would split it into elementary steps. Also add any other intermediate you think every "
    "candidate misses. Give one species per line between <states> and </states>, as `C<candidate>.S<step>: "
    "SMILES` (the step it belongs before). Write nothing for steps you judge concerted."
)

# Section S12.3: MechSight-Auto cross-draft splicing pathway revision
AUTO_REVISE_TEMPLATE = (
    "Reactants and reagents (SMILES): {reactants}\n"
    "Products (SMILES): {products}\n"
    "Reaction conditions: {conditions}\n\n"
    "Candidate mechanisms proposed for this reaction (intermediate SMILES after each step; steps flagged by "
    "an atom-mapping check change several bonds at once):\n\n"
    "{cands}\n\n"
    "Intermediates proposed for the flagged steps (valid structures; not all of them need to be right):\n"
    "{states}\n\n"
    "First judge each candidate: discard those with unsound elementary steps or missing intermediates. Then "
    "write the final mechanism: keep the most plausible candidate or repair it, inserting the proposed "
    "intermediates where they belong and only where they are chemically justified. Return the final mechanism "
    "between [ANSWER] and [/ANSWER] as a JSON list of steps, each "
    '{{"step": n, "type": ..., "subtype": ..., "intermediate_smiles": ...}}.'
)

# Section S12.4: MechSight-Guided observation-driven intermediate hypothesis generation (4 calls)
GUIDED_HYPOTHESIS_TEMPLATE = (
    "Reactants and reagents (SMILES): {reactants}\n"
    "Products (SMILES): {products}\n"
    "Reaction conditions: {conditions}\n\n"
    "Experimental observations:\n{obs}\n\n"
    "Propose up to four distinct mechanistic intermediate structures supported by these observations on the "
    "way from the reactants to the products. Distinguish a trapped adduct from its precursor, and an inferred "
    "intermediate from a directly detected species. Respect any explicitly reported formula, isotope location "
    "or structure, but do not invent an observed formula or mass. Explain briefly how each proposal accounts "
    "for the observations. Give one valid SMILES per line between <states> and </states>. If no structure can "
    "reasonably be proposed, leave the block empty."
)

# Section S12.5: MechSight-Guided consensus-anchored pathway revision
GUIDED_REVISE_TEMPLATE = (
    "Reactants and reagents (SMILES): {reactants}\n"
    "Products (SMILES): {products}\n"
    "Reaction conditions: {conditions}\n\n"
    "Experimental observations:\n{obs}\n\n"
    "Intermediates corroborated by several independent proposals (support = number of the {nsrc} independent "
    "proposal sets that contain the structure; calculated formula in brackets). These are inferences to assess "
    "against the observations, not experimental facts:\n{consensus}\n\n"
    "Candidate mechanisms{cand_note} (flagged steps change several bonds):\n{cands}\n\n"
    "Checked intermediates proposed for flagged steps:\n{states}\n\n"
    "Write the final mechanism consistent with the actual observations. Include every corroborated "
    "intermediate in its proper place unless an elementary-step or observation argument rules it out; for each "
    "one you exclude, state the reason in one sentence before the answer. Preserve the other sound "
    "intermediates, add the species needed so that every step is elementary, and do not treat a trapped "
    "adduct as a native intermediate unless justified. Return the final mechanism between [ANSWER] and "
    '[/ANSWER] as a JSON list of steps, each {{"step": n, "type": ..., "subtype": ..., '
    '"intermediate_smiles": ...}}.'
)

# Section S12.6: MechSight-Guided non-destructive consensus anchor insertion
GUIDED_INSERT_TEMPLATE = (
    "Reactants and reagents (SMILES): {reactants}\n"
    "Products (SMILES): {products}\n"
    "Reaction conditions: {conditions}\n\n"
    "Experimental observations:\n{obs}\n\n"
    "Current mechanism (JSON list of steps):\n{mech}\n\n"
    "Intermediates proposed by at least 3 of 7 independent mechanism proposals but missing from the current "
    "mechanism:\n{missing}\n\n"
    "Insert each missing intermediate at its chemically proper place, adjusting neighbouring steps so that "
    "every step stays elementary. Do not delete or change the existing intermediates. Return the final "
    "mechanism between [ANSWER] and [/ANSWER] as a JSON list of steps, each "
    '{{"step": n, "type": ..., "subtype": ..., "intermediate_smiles": ...}}.'
)

# Section S12.7: Baseline and control prompt blocks
SELF_REFINE_TEMPLATE = (
    "Reactants and reagents (SMILES): {reactants}\n"
    "Products (SMILES): {products}\n"
    "Reaction conditions: {conditions}\n\n"
    "A proposed stepwise mechanism (intermediate SMILES after each step):\n{mech}\n\n"
    "Critically check it step by step: does any step skip a short-lived intermediate or change several bonds "
    "at once? Is every step a sound elementary step? Then write the improved mechanism (or the same one if "
    "it is already right) between [ANSWER] and [/ANSWER] as a JSON list of steps, each "
    '{{"step": n, "type": ..., "subtype": ..., "intermediate_smiles": ...}}.'
)

DIRECT_CLUES_BLOCK = (
    "\n\n## Experimental observation made while running this reaction\n\n"
    "{obs}\n\nThe mechanism must be consistent with this observation.\n\n## Input"
)

ORACLE_INTERMEDIATES_BLOCK = (
    "\n\n## Known intermediates\n\n"
    "The following species are known intermediates of this reaction (order not given):\n"
    "{species}\n\nThe mechanism must pass through them.\n\n## Input"
)

STEP_LABEL_TEMPLATE = (
    "Below is a stepwise mechanism (intermediate SMILES after each step). Do not change the mechanism. For "
    "every step, choose the type/subtype pair from the allowed list that describes it.\n\n"
    "Allowed type/subtype pairs:\n{allowed}\n\n"
    "Mechanism:\n{steps}\n\n"
    "Return between [ANSWER] and [/ANSWER] a JSON list with one entry per step, in order: "
    '{{"step": n, "type": ..., "subtype": ..., "intermediate_smiles": ...}} (copy the SMILES).'
)

# Section S12.8: Multimodal full-PDF literature extraction prompt
PDF_EXTRACTION_PROMPT = """The attached PDF is a 2026 paper with mechanistic studies. Read the text AND the schemes/figures.

Build a machine-checkable record of the mechanism the authors propose for their model reaction.

1. "model_reaction": the specific reaction for which the mechanism is drawn (usually the model substrates, e.g. 1a + 2a -> 3a).
   - "reactants": SMILES of the substrates and stoichiometric reagents (each a separate list item).
   - "catalysts": SMILES of catalysts, photocatalysts, ligands and additives that are not consumed (separate list).
   - "products": SMILES of the main product and stoichiometric by-products.
   - "conditions": short text (catalyst loading, solvent, light, temperature, atmosphere).
   - "compound_labels": map from the paper's labels (1a, 2a, 3a, ...) to SMILES for the compounds used above.
2. "intermediates": every species of the authors' proposed mechanism (scheme of the catalytic cycle / proposed mechanism),
   in mechanistic order. For each:
   - "label": the paper's label (e.g. "I", "Int-A", "B", "7"); "name": short description.
   - "smiles": the FULL structure as the authors draw it, with every generic group (R, Ar, L, X) instantiated with the
     model substrates / actual ligand of the model reaction. Radicals with explicit radical centres in brackets
     (e.g. [CH2]C, [O]), charges explicit, metals with their drawn ligands. Never use text, abbreviations or "*".
   - "role": one of
       "substrate_intermediate"  - contains ANY atoms of the substrates/stoichiometric reagents and lies on the proposed
                                   pathway, INCLUDING metal/catalyst complexes that carry a substrate fragment
                                   (e.g. oxidative-addition complexes L-Pd(Ar)X, substrate-bound catalyst, acyl-metal);
       "catalyst_state"          - species made ONLY of catalyst / photocatalyst / ligand / additive atoms, with no
                                   substrate atoms at all, e.g. Pd(0)L2, PC*, PC.-, Cu(I)L;
       "trapped_adduct"          - a species formed by a trap (TEMPO, BHT, DMPO, ...) or a detected off-path adduct;
       "off_cycle"               - resting states, decomposition, side products.
   - "evidence": "detected" (observed: HRMS, NMR, EPR, X-ray, isolated) or "proposed".
   - "source": where it is drawn (e.g. "Scheme 4", "Fig. 5c").
3. "clues": what the mechanistic experiments OBSERVED (species detected with m/z or spectra, trapped adducts, isolated
   intermediates, isotope-label positions, KIE values, radical-clock / TEMPO results). Observations with numbers only,
   never the authors' interpretation.
4. Self-check before answering: every substrate_intermediate must be derivable from the reactants (its non-metal atoms
   come from the reactants/reagents; elements must be present in reactants or catalysts), must differ from the reactants
   and products, and must use the same substituents as the model substrates. Fix any inconsistency.

If the paper proposes no mechanism with drawn intermediates, return "intermediates": [].
Answer with ONE JSON object between [ANSWER] and [/ANSWER]."""


def reaction_fields(case: ReactionCase) -> dict[str, str]:
    """Return ``reactants``, ``products``, ``conditions`` strings for prompt formatting."""
    return {
        "reactants": ".".join(case.reactants),
        "products": ".".join(case.products),
        "conditions": case.conditions,
    }


def format_direct_prompt(case: ReactionCase, kind: str = "default") -> str:
    """Format the official oMeBench direct or CoT prompt for a ``ReactionCase``."""
    prompt = load_omebench_prompt(kind)
    replacements = {
        "{ reactants_smiles }": case.reactants,
        "{ products_smiles }": case.products,
        "{ conditions }": case.conditions,
    }
    for token, value in replacements.items():
        prompt = prompt.replace(token, json.dumps(value))
    return prompt


def format_enumerate_prompt(case: ReactionCase) -> str:
    """Format the MechSight-Auto 3-pathway hidden-intermediate enumeration prompt (S12.1)."""
    base = format_direct_prompt(case, kind="default").replace(
        "Return only valid JSON. Do not include explanatory text, markdown, or code fences.",
        "Format each mechanism as valid JSON.",
    )
    return base.replace("\n\n## Input", AUTO_ENUMERATE_BLOCK, 1)


def format_auto_complete_prompt(case: ReactionCase, cands_text: str) -> str:
    """Format the MechSight-Auto targeted shortcut completion prompt (S12.2)."""
    return AUTO_COMPLETE_TEMPLATE.format(cands=cands_text, **reaction_fields(case))


def format_auto_revise_prompt(
    case: ReactionCase,
    cands_text: str,
    bridging_states: list[tuple[str, str]],
) -> str:
    """Format the MechSight-Auto cross-draft splicing revision prompt (S12.3)."""
    states_text = "\n".join(f"- ({loc}) {smi}" for loc, smi in bridging_states) or "- none"
    return AUTO_REVISE_TEMPLATE.format(cands=cands_text, states=states_text, **reaction_fields(case))


def format_direct_clues_prompt(case: ReactionCase) -> str:
    """Format the Direct + Clues / evidence-conditioned pathway prompt (S12.7)."""
    base = format_direct_prompt(case, kind="default")
    return base.replace("\n\n## Input", DIRECT_CLUES_BLOCK.format(obs=case.observations), 1)


def format_guided_hypothesis_prompt(case: ReactionCase) -> str:
    """Format the observation-driven intermediate hypothesis prompt (S12.4)."""
    return GUIDED_HYPOTHESIS_TEMPLATE.format(obs=case.observations, **reaction_fields(case))


def format_guided_revise_prompt(
    case: ReactionCase,
    *,
    consensus_lines: str,
    cands_text: str,
    checked_states: list[str],
    n_sources: int = 7,
    cand_note: str = " (candidate 1 is the Auto final path)",
) -> str:
    """Format the MechSight-Guided consensus-anchored pathway revision prompt (S12.5)."""
    states_text = "\n".join(dict.fromkeys(checked_states)) or "- none"
    return GUIDED_REVISE_TEMPLATE.format(
        obs=case.observations,
        nsrc=n_sources,
        consensus=consensus_lines or "- none",
        cand_note=cand_note,
        cands=cands_text,
        states=states_text,
        **reaction_fields(case),
    )


def format_guided_insert_prompt(
    case: ReactionCase,
    current_mech: list[dict[str, Any]],
    missing_anchors: list[str],
) -> str:
    """Format the MechSight-Guided non-destructive consensus anchor insertion prompt (S12.6)."""
    return GUIDED_INSERT_TEMPLATE.format(
        obs=case.observations,
        mech=json.dumps(current_mech, ensure_ascii=False),
        missing="\n".join(f"- {smi}" for smi in missing_anchors),
        **reaction_fields(case),
    )
