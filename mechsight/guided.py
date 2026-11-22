"""MechSight-Guided: consensus experimental-evidence anchoring (SI Section S10.1).

Integrates raw experimental observations onto the verified MechSight-Auto prior
scaffold in three steps:
  1. Seven independent evidence-conditioned proposals (4 observation-driven
     intermediate hypothesis calls at T=0.7 + 3 evidence-conditioned pathway
     samples: 1 at T=0.2 and 2 at T=0.7). RDKit-validated species supported by
     >= 3 of the 7 proposal sets form the consensus anchor set A_i.
  2. Anchor-constrained scaffold revision (T=0.2) combining the Auto scaffold,
     unguided candidate drafts, evidence-conditioned pathways, RDKit-checked
     shortcut bridging states, and the consensus anchors (with support & formula).
  3. Non-destructive consensus anchor insertion (T=0.2) if any consensus anchor
     in A_i remains absent from the revised pathway.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .auto import MechSightAuto
from .bond_checker import BondChangeChecker
from .chemistry import (
    filter_proposed_states,
    molecular_formula,
    parse_mechanism_json,
    sigma_graph_keys,
)
from .llm import LLMClient
from .prompts import (
    SYSTEM_PROMPT,
    format_direct_clues_prompt,
    format_guided_hypothesis_prompt,
    format_guided_insert_prompt,
    format_guided_revise_prompt,
)
from .schema import AutoResult, GuidedResult, ReactionCase


@dataclass(frozen=True)
class MechSightGuidedConfig:
    """Hyperparameters matching the frozen MechSight-Guided configuration (SI S10.1)."""

    n_hypotheses: int = 4
    n_clue_paths: int = 3
    consensus_threshold: int = 3
    hypothesis_temperature: float = 0.7
    clue_primary_temperature: float = 0.2
    clue_sample_temperature: float = 0.7
    revision_temperature: float = 0.2
    insertion_temperature: float = 0.2
    max_auto_candidates_in_revision: int = 4


class MechSightGuided:
    """Experimental-evidence anchoring layer built on top of ``MechSightAuto``."""

    def __init__(
        self,
        llm: LLMClient,
        auto_runner: MechSightAuto | None = None,
        bond_checker: BondChangeChecker | None = None,
        config: MechSightGuidedConfig | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or MechSightGuidedConfig()
        self.bond_checker = bond_checker or (
            auto_runner.bond_checker if auto_runner is not None else BondChangeChecker()
        )
        self.auto_runner = auto_runner or MechSightAuto(llm=llm, bond_checker=self.bond_checker)

    def _proposal_support(
        self,
        case: ReactionCase,
        raw_answers: dict[str, str | None],
    ) -> tuple[Counter[str], list[list[dict[str, Any]]]]:
        support: Counter[str] = Counter()
        hyp_prompt = format_guided_hypothesis_prompt(case)
        for k in range(self.config.n_hypotheses):
            tag = f"G7L_hyp{k}"
            ans = self.llm.complete(
                system=SYSTEM_PROMPT,
                prompt=hyp_prompt,
                temperature=self.config.hypothesis_temperature,
                tag=tag,
                cache_key=f"{case.case_id}__{tag}",
            )
            raw_answers[tag] = ans
            for _, smi in filter_proposed_states(case.reactants, case.products, ans):
                support[smi] += 1

        clue_prompt = format_direct_clues_prompt(case)
        clue_paths: list[list[dict[str, Any]]] = []
        for k in range(self.config.n_clue_paths):
            tag = "O_llm" if k == 0 else f"G7L_clue{k}"
            temp = (
                self.config.clue_primary_temperature
                if k == 0
                else self.config.clue_sample_temperature
            )
            ans = self.llm.complete(
                system=SYSTEM_PROMPT,
                prompt=clue_prompt,
                temperature=temp,
                tag=tag,
                cache_key=f"{case.case_id}__{tag}",
            )
            raw_answers[tag] = ans
            mech = parse_mechanism_json(ans)
            if not mech:
                continue
            clue_paths.append(mech)
            dedup_smis = "\n".join(dict.fromkeys(s.get("intermediate_smiles", "") for s in mech))
            tagged_block = f"<states>\n{dedup_smis}\n</states>"
            for _, smi in filter_proposed_states(case.reactants, case.products, tagged_block):
                support[smi] += 1

        return support, clue_paths

    @staticmethod
    def _missing_anchors(
        consensus_anchors: list[tuple[str, int, str | None]],
        pathway: list[dict[str, Any]] | None,
    ) -> list[str]:
        have = [k for s in (pathway or []) if (k := sigma_graph_keys(s.get("intermediate_smiles")))]
        missing: list[str] = []
        for smi, _, _ in consensus_anchors:
            k = sigma_graph_keys(smi)
            if k and not any(k & h for h in have):
                missing.append(smi)
        return missing

    def run(
        self,
        case: ReactionCase,
        *,
        auto_result: AutoResult | None = None,
    ) -> GuidedResult:
        """Run MechSight-Guided on ``case`` (requires ``case.observations``)."""
        if not case.observations.strip():
            raise ValueError(f"ReactionCase {case.case_id!r} has empty observations; use MechSightAuto instead.")

        if auto_result is None:
            auto_result = self.auto_runner.run(case)

        raw_answers = dict(auto_result.raw_answers)
        support, clue_paths = self._proposal_support(case, raw_answers)

        consensus_anchors: list[tuple[str, int, str | None]] = [
            (smi, count, molecular_formula(smi))
            for smi, count in support.most_common()
            if count >= self.config.consensus_threshold
        ]
        consensus_lines = "\n".join(
            f"- {smi} [support {count}; {formula}]" for smi, count, formula in consensus_anchors
        )

        cands: list[list[dict[str, Any]]] = []
        if auto_result.final_pathway:
            cands.append(auto_result.final_pathway)
        cands.extend(auto_result.candidates[: self.config.max_auto_candidates_in_revision])
        cands.extend(clue_paths)

        cands_text, _ = self.bond_checker.render_annotated_candidates(case, cands) if cands else ("- none", [])
        checked_states = [
            smi
            for _, smi in (auto_result.round1_bridging_states + auto_result.round2_bridging_states)
        ]

        n_sources = self.config.n_hypotheses + self.config.n_clue_paths
        rev_prompt = format_guided_revise_prompt(
            case,
            consensus_lines=consensus_lines,
            cands_text=cands_text,
            checked_states=checked_states,
            n_sources=n_sources,
        )
        rev_ans = self.llm.complete(
            system=SYSTEM_PROMPT,
            prompt=rev_prompt,
            temperature=self.config.revision_temperature,
            tag="G71_rev",
            cache_key=f"{case.case_id}__G71_rev",
        )
        raw_answers["G71_rev"] = rev_ans
        revised_pathway = parse_mechanism_json(rev_ans)

        missing = self._missing_anchors(consensus_anchors, revised_pathway)
        insertion_triggered = False
        final_pathway = revised_pathway

        if revised_pathway and missing:
            insertion_triggered = True
            ins_prompt = format_guided_insert_prompt(case, revised_pathway, missing)
            ins_ans = self.llm.complete(
                system=SYSTEM_PROMPT,
                prompt=ins_prompt,
                temperature=self.config.insertion_temperature,
                tag="A1_place",
                cache_key=f"{case.case_id}__A1_place",
            )
            raw_answers["A1_place"] = ins_ans
            inserted_pathway = parse_mechanism_json(ins_ans)
            if inserted_pathway:
                final_pathway = inserted_pathway

        return GuidedResult(
            case_id=case.case_id,
            auto_result=auto_result,
            proposal_support=dict(support),
            consensus_anchors=consensus_anchors,
            revised_pathway=revised_pathway,
            missing_anchors=missing,
            insertion_triggered=insertion_triggered,
            final_pathway=final_pathway,
            raw_answers=raw_answers,
        )
