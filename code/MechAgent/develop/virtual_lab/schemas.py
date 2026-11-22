"""Data structures for VirtualLab universal reaction states, steps, hypotheses, and probes."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .. import standalone_chemistry as sc

EXPERIMENT_TYPES = (
    "kie",
    "radical_trap",
    "radical_clock",
    "light_on_off",
    "quantum_yield",
    "crossover",
    "isotope_label",
    "hammett",
    "kinetic_order",
    "nonlinear_effect",
    "control",
    "stern_volmer",
    "electrochemistry",
    "spectroscopy",
    "stoichiometric_intermediate",
    "stereochemical_probe",
    "competition",
    "other",
)


@dataclass
class AtomMeta:
    """Extended physical-organic state tracked per PGL atom label."""

    isotope: Optional[str] = None          # e.g. "2H", "13C", "18O", "15N"
    radicals: int = 0                      # unpaired electron count (0=closed-shell, 1=doublet, 2=triplet/biradical)
    excited: bool = False                  # electronically excited center (*PC or *Substrate)
    stereo_parity: Optional[int] = None    # +1 or -1 for defined configuration, 0 for racemic/scrambled
    fragment_origin: Optional[str] = None  # provenance tag ("A", "B", "solvent", "reagent") for crossover
    oxidation_state: Optional[int] = None  # optional formal oxidation state for transition-metal centers
    role_tag: Optional[str] = None         # optional tag: "catalyst", "ligand", "base", "photocatalyst", "substrate", "solvent"

    def clone(self) -> "AtomMeta":
        return copy.copy(self)


@dataclass
class LabState:
    """Complete quantum/topological reaction state extending PGL ReactionState."""

    pgl_state: sc.ReactionState
    atom_meta: Dict[str, AtomMeta] = field(default_factory=dict)
    net_electrode_electrons: int = 0  # +n if n electrons injected from cathode, -n if removed at anode
    photons_absorbed: int = 0

    def __post_init__(self) -> None:
        for name, _ in sc.pgl.iter_atoms(self.pgl_state):
            self.atom_meta.setdefault(name, AtomMeta())

    def clone(self) -> "LabState":
        return LabState(
            pgl_state=copy.deepcopy(self.pgl_state),
            atom_meta={k: v.clone() for k, v in self.atom_meta.items()},
            net_electrode_electrons=self.net_electrode_electrons,
            photons_absorbed=self.photons_absorbed,
        )

    @classmethod
    def from_pgl(
        cls,
        pgl_input: Union[str, sc.ReactionState],
        *,
        atom_meta: Optional[Dict[str, AtomMeta]] = None,
    ) -> "LabState":
        state = sc.parse_pgl_literal(pgl_input) if isinstance(pgl_input, str) else copy.deepcopy(pgl_input)
        if not isinstance(state, sc.ReactionState):
            raise TypeError(f"Expected ReactionState, got {type(state).__name__}")
        meta = {k: v.clone() for k, v in (atom_meta or {}).items()}
        return cls(pgl_state=state, atom_meta=meta)


@dataclass
class MechanisticStep:
    """Universal elementary step covering polar, radical, photoredox, electro, organometallic, and pericyclic chemistry.

    Can be driven either by PGL curly-arrow ``electron_transfers`` (1e or 2e) OR by declarative
    bond/charge/spin/redox deltas (essential for d-orbital organometallics, EnT, and electrode steps),
    or both.
    """

    step_class: str = "polar"
    name: str = ""
    # Option 1: PGL curly arrows (ReactionTransformation, list of ElectronTransfer, or PGL string)
    pgl_action: Optional[Union[str, sc.ReactionTransformation, Dict[str, Any]]] = None
    # Option 2: Declarative universal graph & electronic edits
    bonds_formed: List[Tuple[str, str, float]] = field(default_factory=list)
    bonds_broken: List[Tuple[str, str]] = field(default_factory=list)
    bond_order_changes: List[Tuple[str, str, float]] = field(default_factory=list)
    charge_changes: Dict[str, int] = field(default_factory=dict)
    radical_changes: Dict[str, int] = field(default_factory=dict)
    oxidation_state_changes: Dict[str, int] = field(default_factory=dict)
    excited_Set: List[str] = field(default_factory=list)
    excited_Clear: List[str] = field(default_factory=list)
    electrode_electrons: int = 0   # >0 cathodic reduction (+e-), <0 anodic oxidation (-e-)
    photons: int = 0               # >0 photochemical excitation step (hv)

    # Kinetic, stereochemical & physical-organic annotations
    reversible: bool = False
    is_rds: bool = False
    is_chain_propagation: bool = False
    is_chain_initiation: bool = False
    cage_escaped: Optional[bool] = None         # True = diffusible free radical/ion; False = tight solvent cage
    rebound_rate_s: Optional[float] = None      # inner-sphere rebound rate k_reb (s^-1) for radical clock competition
    bimolecular_trap_rate_s: float = 1.0e6      # effective pseudo-first-order trapping rate k_bimol[M] (s^-1)
    stereo_course: Dict[str, str] = field(default_factory=dict)  # atom_id -> "inversion"|"retention"|"racemization"
    pericyclic_electrons: Optional[int] = None  # e.g. 4 or 6 for Woodward-Hoffmann electrocyclic stereo
    aryl_probe_atom: Optional[str] = None       # reaction-center atom conjugated to variable Ar substituent (Hammett)
    ts_charge_shift: Optional[float] = None     # explicit override for delta_q_ddagger at aryl reaction center


@dataclass
class CatalyticSpec:
    """Macro-kinetic, photophysical, electrochemical, and nonlinear-effect specification of a pathway."""

    is_catalytic: bool = False
    tdi_index: int = 0                          # state index of Turnover-Determining Intermediate (resting state)
    tdts_step_index: Optional[int] = None       # step index of Turnover-Determining Transition State (RDS)
    # Component metadata: component_name -> {"atoms": [...], "role": "substrate"|"catalyst"|"ligand"|"base"|"additive"|"oxidant"|"reductant"|"solvent", "essential": bool, "background_yield_fraction": float}
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Off-cycle resting-state equilibria (e.g. dimeric precatalyst [M2L2] <=> 2 [ML], or L dissociation [ML2] <=> [ML] + L)
    catalyst_resting_nuclearity: int = 1        # 2 => half-order (0.5) in total [Cat] when monomer is active in TDTS
    active_catalyst_nuclearity: int = 1
    dissociated_ligands_before_tdts: List[str] = field(default_factory=list)  # gives inverse (-1) order
    saturation_components: List[str] = field(default_factory=list)            # pre-equilibrium binding saturation
    # Asymmetric catalysis Kagan ML_n nonlinear effect parameters
    chiral_ligands_in_active: int = 1
    chiral_ligands_in_reservoir: int = 1
    heterochiral_reservoir_K: float = 0.0       # K_hetero / K_homo: >> 2 gives +NLE (meso sink); < 0.5 or homochiral active dimer gives -NLE/+NLE
    meso_reservoir_inactive: bool = True
    # Photoredox & Electrochemistry parameters
    photocatalyst: Optional[str] = None         # key in PHOTOCATALYST_DATA or custom name
    photocatalyst_potentials: Optional[Dict[str, float]] = None
    species_redox: Dict[str, Dict[str, float]] = field(default_factory=dict)  # species_name -> {"E_ox": ..., "E_red": ..., "E_T": ...}
    applied_potential_V: Optional[float] = None # vs SCE for electrolysis
    chain_length_estimate: float = 1.0          # propagation efficiency nu when chain propagation exists
    dark_thermal_initiation: bool = False       # whether radical chain also initiates thermally without light


@dataclass
class MechanismHypothesis:
    """A complete, executable competing mechanistic hypothesis H_k."""

    hypothesis_id: str
    name: str
    initial_state: LabState
    steps: List[MechanisticStep]
    target_state: Optional[LabState] = None
    catalytic_spec: CatalyticSpec = field(default_factory=CatalyticSpec)
    prior: float = 1.0
    description: str = ""
    claims: List[str] = field(default_factory=list)

    def resolved_tdts_index(self) -> int:
        if self.catalytic_spec.tdts_step_index is not None:
            return self.catalytic_spec.tdts_step_index
        for idx, step in enumerate(self.steps):
            if step.is_rds:
                return idx
        return 0

    @classmethod
    def from_pgl_path(
        cls,
        path_input: Union[str, sc.ReactionPath],
        *,
        hypothesis_id: Optional[str] = None,
        name: Optional[str] = None,
        atom_meta: Optional[Dict[str, AtomMeta]] = None,
        step_classes: Optional[Sequence[str]] = None,
        catalytic_spec: Optional[CatalyticSpec] = None,
        prior: float = 1.0,
    ) -> "MechanismHypothesis":
        """Build an executable MechanismHypothesis directly from a PGL ReactionPath object or string."""
        rpath = sc.parse_pgl_literal(path_input) if isinstance(path_input, str) else copy.deepcopy(path_input)
        if not isinstance(rpath, sc.ReactionPath):
            raise TypeError(f"Expected ReactionPath, got {type(rpath).__name__}")

        states = [x for x in rpath.sequence if isinstance(x, sc.ReactionState)]
        transforms = [x for x in rpath.sequence if isinstance(x, sc.ReactionTransformation)]
        if len(states) < 2:
            raise ValueError("ReactionPath must contain at least 2 ReactionStates")

        init_lab = LabState.from_pgl(states[0], atom_meta=atom_meta)
        target_lab = LabState.from_pgl(states[-1])
        built_steps: List[MechanisticStep] = []

        for idx in range(len(states) - 1):
            s_prev = states[idx]
            s_next = states[idx + 1]
            tr = transforms[idx] if idx < len(transforms) else None
            s_cls = step_classes[idx] if (step_classes and idx < len(step_classes)) else "polar"

            if tr is not None and tr.electron_transfers:
                has_1e = any(int(et.electrons) == 1 for et in tr.electron_transfers)
                if s_cls == "polar" and has_1e:
                    s_cls = "radical_1e"
                built_steps.append(
                    MechanisticStep(
                        step_class=s_cls,
                        name=f"{rpath.name}_step_{idx + 1}",
                        pgl_action=tr,
                    )
                )
            else:
                # Declarative state-to-state diff for non-arrow steps (e.g. pericyclic/organometallic/ts_collapse)
                prev_bonds = sc.pgl.bond_map(s_prev)
                next_bonds = sc.pgl.bond_map(s_next)
                prev_atoms = dict(sc.pgl.iter_atoms(s_prev))
                next_atoms = dict(sc.pgl.iter_atoms(s_next))
                formed = [
                    (k[0], k[1], float(next_bonds[k].order or 1.0))
                    for k in sorted(set(next_bonds) - set(prev_bonds))
                ]
                broken = [
                    (k[0], k[1])
                    for k in sorted(set(prev_bonds) - set(next_bonds))
                ]
                order_chg = [
                    (k[0], k[1], float(next_bonds[k].order or 1.0))
                    for k in sorted(set(prev_bonds) & set(next_bonds))
                    if abs(float(prev_bonds[k].order or 1.0) - float(next_bonds[k].order or 1.0)) > 1e-9
                ]
                q_chg = {
                    aid: int(next_atoms[aid].charge or 0) - int(prev_atoms[aid].charge or 0)
                    for aid in sorted(set(prev_atoms) & set(next_atoms))
                    if int(next_atoms[aid].charge or 0) != int(prev_atoms[aid].charge or 0)
                }
                built_steps.append(
                    MechanisticStep(
                        step_class=s_cls,
                        name=f"{rpath.name}_step_{idx + 1}",
                        bonds_formed=formed,
                        bonds_broken=broken,
                        bond_order_changes=order_chg,
                        charge_changes=q_chg,
                    )
                )

        return cls(
            hypothesis_id=hypothesis_id or (rpath.name or "H_pgl"),
            name=name or (rpath.name or "PGL Pathway"),
            initial_state=init_lab,
            steps=built_steps,
            target_state=target_lab,
            catalytic_spec=catalytic_spec or CatalyticSpec(),
            prior=prior,
        )


@dataclass
class ExperimentalProbe:
    """Parameterized query for a virtual mechanistic experiment."""

    experiment_type: str
    perturbation: str = ""
    observable: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    probe_id: str = ""

    def __post_init__(self) -> None:
        if self.experiment_type not in EXPERIMENT_TYPES:
            raise ValueError(f"Unsupported experiment_type: {self.experiment_type!r}")


@dataclass
class ObservablePrediction:
    """Deterministic prediction produced by VirtualLab for (H_k, ExperimentalProbe)."""

    hypothesis_id: str
    experiment_type: str
    outcome_bin: Optional[str]
    quantitative_value: Any = None
    structural_details: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.90
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "experiment_type": self.experiment_type,
            "outcome_bin": self.outcome_bin,
            "quantitative_value": self.quantitative_value,
            "structural_details": self.structural_details,
            "confidence": round(self.confidence, 4),
            "rationale": self.rationale,
        }


@dataclass
class ExperimentDesignCandidate:
    """Candidate next experiment ranked by Expected Information Gain (EIG)."""

    probe: ExperimentalProbe
    expected_info_gain_bits: float
    predictions_by_hypothesis: Dict[str, ObservablePrediction]
    max_bayes_factor: float
    discriminating_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_id": self.probe.probe_id,
            "experiment_type": self.probe.experiment_type,
            "perturbation": self.probe.perturbation,
            "observable": self.probe.observable,
            "expected_info_gain_bits": round(self.expected_info_gain_bits, 4),
            "max_bayes_factor": round(self.max_bayes_factor, 2),
            "discriminating_summary": self.discriminating_summary,
            "predictions": {
                hid: pred.outcome_bin for hid, pred in self.predictions_by_hypothesis.items()
            },
        }
