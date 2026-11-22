"""Typed data structures for MechSight reaction inputs, pathways, gaps, and outputs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MechanismStep:
    """Single elementary step in a predicted or reference mechanism."""

    step: int
    intermediate_smiles: str
    type: str = ""
    subtype: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to the standard oMeBench-compatible JSON step dictionary."""
        return {
            "step": int(self.step),
            "type": self.type,
            "subtype": self.subtype,
            "intermediate_smiles": self.intermediate_smiles,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_step: int = 1) -> MechanismStep:
        """Build from a step dictionary."""
        raw_step = data.get("step", default_step)
        try:
            step_num = int(raw_step) if raw_step is not None else default_step
        except (TypeError, ValueError):
            step_num = default_step
        return cls(
            step=step_num,
            type=str(data.get("type") or ""),
            subtype=str(data.get("subtype") or ""),
            intermediate_smiles=str(data.get("intermediate_smiles") or ""),
        )


@dataclass
class ReactionCase:
    """Input specification for a reaction mechanism elucidation task."""

    case_id: str
    reactants: list[str]
    products: list[str]
    conditions: str = ""
    observations: str = ""
    reference_intermediates: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReactionCase:
        """Construct from a dict (accepts either ``observations`` or ``clues``)."""
        reactants = data.get("reactants", [])
        if isinstance(reactants, str):
            reactants = [x for x in reactants.split(".") if x]
        products = data.get("products", [])
        if isinstance(products, str):
            products = [x for x in products.split(".") if x]
        return cls(
            case_id=str(data.get("case_id") or data.get("id") or "reaction"),
            reactants=list(reactants),
            products=list(products),
            conditions=str(data.get("conditions") or ""),
            observations=str(data.get("observations") or data.get("clues") or ""),
            reference_intermediates=list(data.get("reference_intermediates") or data.get("author") or []),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class GapFlag:
    """Detected multi-bond shortcut (Delta_bond >= 3) or unparsable transition."""

    step_index: int
    reason: str
    delta_bond: int | None = None
    step_rxn: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AutoResult:
    """Complete output and diagnostics from MechSight-Auto."""

    case_id: str
    candidates: list[list[dict[str, Any]]]
    round1_gaps: list[list[GapFlag]]
    round1_bridging_states: list[tuple[str, str]]
    round1_pathway: list[dict[str, Any]] | None
    round2_triggered: bool
    round2_bridging_states: list[tuple[str, str]]
    final_pathway: list[dict[str, Any]] | None
    final_gaps: list[GapFlag]
    raw_answers: dict[str, str | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["round1_gaps"] = [[g.to_dict() for g in cg] for cg in self.round1_gaps]
        data["final_gaps"] = [g.to_dict() for g in self.final_gaps]
        return data


@dataclass
class GuidedResult:
    """Complete output and diagnostics from MechSight-Guided."""

    case_id: str
    auto_result: AutoResult | None
    proposal_support: dict[str, int]
    consensus_anchors: list[tuple[str, int, str | None]]
    revised_pathway: list[dict[str, Any]] | None
    missing_anchors: list[str]
    insertion_triggered: bool
    final_pathway: list[dict[str, Any]] | None
    raw_answers: dict[str, str | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "auto_result": self.auto_result.to_dict() if self.auto_result is not None else None,
            "proposal_support": dict(self.proposal_support),
            "consensus_anchors": [list(item) for item in self.consensus_anchors],
            "revised_pathway": self.revised_pathway,
            "missing_anchors": list(self.missing_anchors),
            "insertion_triggered": self.insertion_triggered,
            "final_pathway": self.final_pathway,
            "raw_answers": dict(self.raw_answers),
        }
