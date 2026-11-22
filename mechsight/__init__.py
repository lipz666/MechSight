"""MechSight: Knowledge-Driven Reaction Mechanism Elucidation.

Public package exposing the two training-free MechSight layers:
  - ``MechSightAuto``: autonomous prior-scaffold construction via candidate
    drafting, RXNMapper elementary bond-change flagging (Delta_bond >= 3),
    RDKit validation, and cross-draft splicing.
  - ``MechSightGuided``: experimental-evidence anchoring onto the verified
    Auto scaffold via >= 3/7 consensus proposals and non-destructive insertion.
"""
from .auto import MechSightAuto, MechSightAutoConfig
from .bond_checker import BondChangeChecker, count_heavy_bond_changes
from .chemistry import (
    canonicalize_smiles,
    equiv_connectivity_keys,
    filter_proposed_states,
    fixed_h_inchi_keys,
    molecular_formula,
    parse_mechanism_json,
    sigma_graph_keys,
)
from .evaluation import (
    bootstrap_ci,
    evaluate_pathway,
    evaluate_single_step,
    paired_permutation_pvalue,
)
from .guided import MechSightGuided, MechSightGuidedConfig
from .llm import LLMClient, OpenAICompatibleClient
from .schema import AutoResult, GapFlag, GuidedResult, MechanismStep, ReactionCase

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "AutoResult",
    "BondChangeChecker",
    "GapFlag",
    "GuidedResult",
    "LLMClient",
    "MechSightAuto",
    "MechSightAutoConfig",
    "MechSightGuided",
    "MechSightGuidedConfig",
    "MechanismStep",
    "OpenAICompatibleClient",
    "ReactionCase",
    "bootstrap_ci",
    "canonicalize_smiles",
    "count_heavy_bond_changes",
    "equiv_connectivity_keys",
    "evaluate_pathway",
    "evaluate_single_step",
    "filter_proposed_states",
    "fixed_h_inchi_keys",
    "molecular_formula",
    "paired_permutation_pvalue",
    "parse_mechanism_json",
    "sigma_graph_keys",
]
