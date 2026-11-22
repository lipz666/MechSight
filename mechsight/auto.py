"""MechSight-Auto: autonomous prior-scaffold construction (SI Section S7.1).

Three sequential stages using no external experimental clues:
  1. Diverse candidate drafting (1 deterministic Direct call at T=0.2,
     4 stochastic samples at T=0.7, 1 hidden-intermediate-directed 3-pathway
     enumeration call at T=0.7), canonical deduplication (<= 8 unique drafts),
     and RXNMapper elementary bond-change gap flagging (Delta_bond >= 3).
  2. Targeted intermediate completion (T=0.2) + deterministic RDKit valence,
     heavy-atom budget, and non-endpoint filtering.
  3. Cross-draft splicing pathway revision (T=0.2), with one optional second
     completion-revision cycle if Delta_bond >= 3 shortcuts remain and new valid
     bridging intermediates are proposed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .bond_checker import BondChangeChecker
from .chemistry import (
    deduplicate_candidate_pathways,
    filter_proposed_states,
    parse_enumerated_paths,
    parse_mechanism_json,
)
from .llm import LLMClient
from .prompts import (
    SYSTEM_PROMPT,
    format_auto_complete_prompt,
    format_auto_revise_prompt,
    format_direct_prompt,
    format_enumerate_prompt,
)
from .schema import AutoResult, ReactionCase


@dataclass(frozen=True)
class MechSightAutoConfig:
    """Hyperparameters matching the frozen MechSight-Auto configuration (SI S1.1, S7.1)."""

    n_stochastic_samples: int = 4
    direct_temperature: float = 0.2
    sample_temperature: float = 0.7
    enumerate_temperature: float = 0.7
    completion_temperature: float = 0.2
    revision_temperature: float = 0.2
    gap_threshold: int = 3
    max_candidates: int = 8
    max_rounds: int = 2


class MechSightAuto:
    """Autonomous clue-free hidden-intermediate recovery layer."""

    def __init__(
        self,
        llm: LLMClient,
        bond_checker: BondChangeChecker | None = None,
        config: MechSightAutoConfig | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or MechSightAutoConfig()
        self.bond_checker = bond_checker or BondChangeChecker(gap_threshold=self.config.gap_threshold)

    def _draft_candidates(self, case: ReactionCase, raw_answers: dict[str, str | None]) -> list[list[dict[str, Any]]]:
        direct_prompt = format_direct_prompt(case, kind="default")
        enum_prompt = format_enumerate_prompt(case)

        direct_ans = self.llm.complete(
            system=SYSTEM_PROMPT,
            prompt=direct_prompt,
            temperature=self.config.direct_temperature,
            tag="base_default",
            cache_key=f"{case.case_id}__base_default",
        )
        raw_answers["base_default"] = direct_ans
        drafts: list[list[dict[str, Any]] | None] = [parse_mechanism_json(direct_ans)]

        for idx in range(1, self.config.n_stochastic_samples + 1):
            tag = f"samp{idx}"
            ans = self.llm.complete(
                system=SYSTEM_PROMPT,
                prompt=direct_prompt,
                temperature=self.config.sample_temperature,
                tag=tag,
                cache_key=f"{case.case_id}__{tag}",
            )
            raw_answers[tag] = ans
            drafts.append(parse_mechanism_json(ans))

        enum_ans = self.llm.complete(
            system=SYSTEM_PROMPT,
            prompt=enum_prompt,
            temperature=self.config.enumerate_temperature,
            tag="enum3",
            cache_key=f"{case.case_id}__enum3",
        )
        raw_answers["enum3"] = enum_ans
        drafts.extend(parse_enumerated_paths(enum_ans, count=3))
        return deduplicate_candidate_pathways(drafts, max_candidates=self.config.max_candidates)

    def run(self, case: ReactionCase) -> AutoResult:
        """Run the full MechSight-Auto pipeline on a ``ReactionCase``."""
        raw_answers: dict[str, str | None] = {}
        candidates = self._draft_candidates(case, raw_answers)
        if not candidates:
            return AutoResult(
                case_id=case.case_id,
                candidates=[],
                round1_gaps=[],
                round1_bridging_states=[],
                round1_pathway=None,
                round2_triggered=False,
                round2_bridging_states=[],
                final_pathway=None,
                final_gaps=[],
                raw_answers=raw_answers,
            )

        cands_text, round1_gaps = self.bond_checker.render_annotated_candidates(case, candidates)
        comp1_ans = self.llm.complete(
            system=SYSTEM_PROMPT,
            prompt=format_auto_complete_prompt(case, cands_text),
            temperature=self.config.completion_temperature,
            tag="V5_comp1",
            cache_key=f"{case.case_id}__V5_comp1",
        )
        raw_answers["V5_comp1"] = comp1_ans
        r1_states = filter_proposed_states(case.reactants, case.products, comp1_ans)

        rev1_ans = self.llm.complete(
            system=SYSTEM_PROMPT,
            prompt=format_auto_revise_prompt(case, cands_text, r1_states),
            temperature=self.config.revision_temperature,
            tag="V5_rev1",
            cache_key=f"{case.case_id}__V5_rev1",
        )
        raw_answers["V5_rev1"] = rev1_ans
        r1_pathway = parse_mechanism_json(rev1_ans)

        final_pathway = r1_pathway
        r2_triggered = False
        r2_states: list[tuple[str, str]] = []

        if self.config.max_rounds >= 2 and r1_pathway:
            r1_final_gaps = self.bond_checker.detect_gaps(case, r1_pathway)
            if r1_final_gaps:
                r2_triggered = True
                r1_text, _ = self.bond_checker.render_annotated_candidates(case, [r1_pathway])
                comp2_ans = self.llm.complete(
                    system=SYSTEM_PROMPT,
                    prompt=format_auto_complete_prompt(case, r1_text),
                    temperature=self.config.completion_temperature,
                    tag="V5_comp2",
                    cache_key=f"{case.case_id}__V5_comp2",
                )
                raw_answers["V5_comp2"] = comp2_ans
                r2_states = filter_proposed_states(case.reactants, case.products, comp2_ans)
                if r2_states:
                    rev2_ans = self.llm.complete(
                        system=SYSTEM_PROMPT,
                        prompt=format_auto_revise_prompt(case, r1_text, r2_states),
                        temperature=self.config.revision_temperature,
                        tag="V5_rev2",
                        cache_key=f"{case.case_id}__V5_rev2",
                    )
                    raw_answers["V5_rev2"] = rev2_ans
                    r2_pathway = parse_mechanism_json(rev2_ans)
                    if r2_pathway:
                        final_pathway = r2_pathway

        final_gaps = self.bond_checker.detect_gaps(case, final_pathway) if final_pathway else []
        return AutoResult(
            case_id=case.case_id,
            candidates=candidates,
            round1_gaps=round1_gaps,
            round1_bridging_states=r1_states,
            round1_pathway=r1_pathway,
            round2_triggered=r2_triggered,
            round2_bridging_states=r2_states,
            final_pathway=final_pathway,
            final_gaps=final_gaps,
            raw_answers=raw_answers,
        )
