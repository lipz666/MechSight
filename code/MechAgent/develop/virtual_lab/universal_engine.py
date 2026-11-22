"""Universal Mechanistic Execution Engine covering all organic, radical, photoredox, electro, organometallic, and pericyclic reactions."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from .. import standalone_chemistry as sc
from .chem_graph import (
    base_element,
    canonical_isotope_tag,
    connected_molecules,
    extract_graph,
    normalize_pgl_species,
)
from .reference_data import TRANSITION_METALS
from .schemas import AtomMeta, LabState, MechanismHypothesis, MechanisticStep


class MechanisticConservationError(ValueError):
    """Raised when an elementary step violates atom, isotope, or charge+electrode conservation."""


@dataclass
class StepExecutionRecord:
    """Detailed physical-organic record of executing a single MechanisticStep."""

    step_index: int
    step_class: str
    name: str
    bonds_formed: List[Tuple[str, str, float]]
    bonds_broken: List[Tuple[str, str, float]]
    bond_order_changes: List[Tuple[str, str, float, float]]
    charge_changes: Dict[str, Tuple[int, int]]
    radical_changes: Dict[str, Tuple[int, int]]
    stereo_events: Dict[str, str]
    touched_atoms: Tuple[str, ...]
    operations: List[str] = field(default_factory=list)


@dataclass
class TrajectoryResult:
    """Full trajectory replay of a MechanismHypothesis."""

    hypothesis_id: str
    states: List[LabState]                # length = len(steps) + 1 (includes initial state at index 0)
    step_records: List[StepExecutionRecord]
    valid: bool = True
    error: Optional[str] = None
    failed_step_index: Optional[int] = None

    @property
    def final_state(self) -> LabState:
        return self.states[-1]


def _bond_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _coerce_pgl_action(
    raw_action: Union[str, sc.ReactionTransformation, Dict[str, Any]],
) -> Dict[str, Any]:
    if isinstance(raw_action, sc.ReactionTransformation):
        return raw_action.to_dict()
    if isinstance(raw_action, dict):
        return copy.deepcopy(raw_action)
    if isinstance(raw_action, str):
        parsed, info = sc.extract_action_payload(raw_action)
        if not info.get("valid"):
            raise ValueError(f"Invalid PGL action string: {info.get('errors')}")
        return parsed
    raise TypeError(f"Unsupported pgl_action type: {type(raw_action).__name__}")


def _infer_fishhook_radical_deltas(
    before_state: LabState,
    action_dict: Dict[str, Any],
) -> Dict[str, int]:
    """Infer unpaired electron changes from 1-electron PGL transfers (homolysis, HAT, XAT, addition, beta-scission, SET)."""
    atoms, _, _ = extract_graph(before_state)
    rad_counts = {
        aid: int(before_state.atom_meta.get(aid, AtomMeta()).radicals)
        for aid in atoms
    }
    transfers = sc._normalized_transfers(action_dict)

    for tr in transfers:
        if int(tr["electrons"]) != 1:
            continue
        s_type, s_atoms = sc._endpoint(tr["source"], atoms)
        t_type, t_atoms = sc._endpoint(tr["target"], atoms)
        if (
            s_type == "atom"
            and t_type == "atom"
            and tr["kind"] in sc._IMPLICIT_BOND_KINDS
        ):
            t_type = "bond"
            t_atoms = (s_atoms[0], t_atoms[0])

        if s_type == "atom":
            u = s_atoms[0]
            # Transition metals may supply d-electrons without organic radical parity flip if specified separately
            if rad_counts[u] > 0:
                rad_counts[u] -= 1
            else:
                rad_counts[u] += 1

        if t_type == "atom":
            v = t_atoms[0]
            if rad_counts[v] > 0:
                rad_counts[v] -= 1
            else:
                rad_counts[v] += 1

    return {
        aid: rad_counts[aid] - int(before_state.atom_meta.get(aid, AtomMeta()).radicals)
        for aid in atoms
        if rad_counts[aid] != int(before_state.atom_meta.get(aid, AtomMeta()).radicals)
    }


def woodward_hoffmann_mode(electrons: int, photochemical: bool) -> str:
    """Return 'conrotatory' or 'disrotatory' for an electrocyclic reaction."""
    if electrons <= 0 or electrons % 2 != 0:
        raise ValueError(f"Electrocyclic electron count must be a positive even integer, got {electrons}")
    is_4n = (electrons % 4) == 0
    if is_4n:
        return "disrotatory" if photochemical else "conrotatory"
    return "conrotatory" if photochemical else "disrotatory"


def _propagate_stereochemistry(
    before: LabState,
    after: LabState,
    step: MechanisticStep,
    bonds_formed: Sequence[Tuple[str, str, float]],
    bonds_broken: Sequence[Tuple[str, str, float]],
) -> Dict[str, str]:
    """Update AtomMeta.stereo_parity across a step and return stereochemical event descriptions."""
    events: Dict[str, str] = {}
    after_atoms, _, after_adj = extract_graph(after)

    # 1. Electrocyclic Woodward-Hoffmann annotation if applicable
    if step.step_class == "electrocyclic" and step.pericyclic_electrons:
        mode = woodward_hoffmann_mode(
            step.pericyclic_electrons,
            photochemical=(step.photons > 0),
        )
        events["__electrocyclic__"] = mode

    formed_by_atom: Dict[str, int] = {}
    broken_by_atom: Dict[str, int] = {}
    for a, b, _ in bonds_formed:
        formed_by_atom[a] = formed_by_atom.get(a, 0) + 1
        formed_by_atom[b] = formed_by_atom.get(b, 0) + 1
    for a, b, _ in bonds_broken:
        broken_by_atom[a] = broken_by_atom.get(a, 0) + 1
        broken_by_atom[b] = broken_by_atom.get(b, 0) + 1

    # 2. Explicit rules take priority
    for aid, rule in step.stereo_course.items():
        if aid not in after.atom_meta:
            continue
        meta = after.atom_meta[aid]
        prev = meta.stereo_parity
        r_low = rule.lower()
        if r_low == "inversion":
            if prev in (-1, 1):
                meta.stereo_parity = -prev
            events[aid] = "inversion"
        elif r_low == "retention":
            events[aid] = "retention"
        elif r_low in {"racemization", "scrambling", "epimerization"}:
            meta.stereo_parity = 0
            events[aid] = r_low
        else:
            events[aid] = rule

    # 3. Automatic physical-organic inference on stereogenic carbons not explicitly annotated
    for aid, atom in after_atoms.items():
        if aid in step.stereo_course:
            continue
        meta = after.atom_meta.get(aid)
        if meta is None or meta.stereo_parity is None:
            continue
        if base_element(atom.element) != "C":
            continue

        deg_total = len(after_adj[aid]) + int(atom.Hcount or 0)
        q = int(atom.charge or 0)
        rads = int(meta.radicals)

        # Tricoordinate planar carbocation or diffusible carbon radical => racemization/planarization
        if deg_total == 3 and (q == 1 or rads >= 1) and step.cage_escaped is not False:
            if meta.stereo_parity != 0:
                meta.stereo_parity = 0
                events[aid] = "racemization"
            continue

        # Concerted substitution at saturated carbon
        if formed_by_atom.get(aid, 0) == 1 and broken_by_atom.get(aid, 0) == 1 and deg_total == 4:
            sc_low = step.step_class.lower()
            if sc_low in {"sn2", "ngp_anchimeric", "ngp_opening", "outer_sphere_attack", "halonium_opening"}:
                if meta.stereo_parity in (-1, 1):
                    meta.stereo_parity = -meta.stereo_parity
                events[aid] = "inversion"
            elif sc_low in {
                "reductive_elimination",
                "migratory_insertion",
                "sigma_bond_metathesis",
                "transmetallation_closed",
                "sigmatropic",
                "1_2_shift",
            }:
                events[aid] = "retention"

    return events


def _verify_conservation(
    before: LabState,
    after: LabState,
    step: MechanisticStep,
) -> None:
    """Enforce strict atom, isotope, implicit-H, charge+electrode, and spin parity conservation."""
    b_atoms, _, _ = extract_graph(before)
    a_atoms, _, _ = extract_graph(after)

    if set(b_atoms.keys()) != set(a_atoms.keys()):
        missing = sorted(set(b_atoms) - set(a_atoms))
        extra = sorted(set(a_atoms) - set(b_atoms))
        raise MechanisticConservationError(
            f"Atom label conservation violated in step {step.name or step.step_class!r}: "
            f"missing={missing}, extra={extra}"
        )

    b_total_h = 0
    a_total_h = 0
    for aid in b_atoms:
        ba, aa = b_atoms[aid], a_atoms[aid]
        if base_element(ba.element) != base_element(aa.element):
            raise MechanisticConservationError(
                f"Element changed for atom {aid}: {ba.element} -> {aa.element}"
            )
        b_iso = canonical_isotope_tag(ba.element, before.atom_meta[aid].isotope)
        a_iso = canonical_isotope_tag(aa.element, after.atom_meta[aid].isotope)
        if b_iso != a_iso:
            raise MechanisticConservationError(
                f"Isotope changed on atom {aid} without transfer: {b_iso} -> {a_iso}"
            )
        b_total_h += (1 if base_element(ba.element) == "H" else 0) + int(ba.Hcount or 0)
        a_total_h += (1 if base_element(aa.element) == "H" else 0) + int(aa.Hcount or 0)

    if b_total_h != a_total_h:
        raise MechanisticConservationError(
            f"Hydrogen count conservation violated: before={b_total_h}, after={a_total_h}"
        )

    q_before = sum(int(x.charge or 0) for x in b_atoms.values())
    q_after = sum(int(x.charge or 0) for x in a_atoms.values())
    expected_dq = -int(step.electrode_electrons)
    if (q_after - q_before) != expected_dq:
        raise MechanisticConservationError(
            f"Charge/electron balance violated in step {step.name or step.step_class!r}: "
            f"q_before={q_before}, q_after={q_after}, electrode_electrons={step.electrode_electrons}"
        )

    # Check spin parity if no transition metal and no electrode transfer
    has_tm = any(base_element(x.element) in TRANSITION_METALS for x in b_atoms.values())
    if not has_tm and step.electrode_electrons == 0:
        rad_before = sum(int(before.atom_meta[aid].radicals) for aid in b_atoms)
        rad_after = sum(int(after.atom_meta[aid].radicals) for aid in a_atoms)
        if (rad_before % 2) != (rad_after % 2):
            raise MechanisticConservationError(
                f"Unpaired electron parity violated in closed-circuit non-metal step: "
                f"radicals {rad_before} -> {rad_after}"
            )


def execute_step(
    lab_state: LabState,
    step: Union[MechanisticStep, sc.ReactionTransformation, str, Dict[str, Any]],
    *,
    step_index: int = 0,
    normalize_species: bool = True,
) -> Tuple[LabState, StepExecutionRecord]:
    """Execute one elementary step (curly-arrow or declarative universal step) on a LabState."""
    if not isinstance(step, MechanisticStep):
        step = MechanisticStep(step_class="polar", pgl_action=step)

    before = lab_state.clone()
    working_pgl = copy.deepcopy(before.pgl_state)
    new_meta = {k: v.clone() for k, v in before.atom_meta.items()}
    operations: List[str] = []

    # 1. Execute PGL curly arrows if supplied
    if step.pgl_action is not None:
        action_dict = _coerce_pgl_action(step.pgl_action)
        inferred_rad_deltas = _infer_fishhook_radical_deltas(before, action_dict)
        working_pgl, pgl_ops = sc.apply_action(working_pgl, action_dict)
        operations.extend(pgl_ops)
        for aid, dr in inferred_rad_deltas.items():
            # Only apply inferred radical change if user didn't explicitly override radical_changes for aid
            if aid not in step.radical_changes:
                new_meta[aid].radicals = max(0, new_meta[aid].radicals + dr)

    # 2. Apply declarative graph & electronic edits (organometallic, EnT, SET, electro, pericyclic)
    atoms = sc._atom_map(working_pgl)
    bonds = sc._bond_locations(working_pgl)

    for a, b in step.bonds_broken:
        key = _bond_key(a, b)
        if key not in bonds:
            raise ValueError(f"Cannot break non-existent bond {key} in step {step.name or step.step_class!r}")
        sub, bond_obj = bonds.pop(key)
        sub.bondlist.remove(bond_obj)
        operations.append(f"break_bond:{key}")

    for a, b, new_order in step.bond_order_changes:
        key = _bond_key(a, b)
        if key not in bonds:
            raise ValueError(f"Cannot change order of non-existent bond {key}")
        sub, bond_obj = bonds[key]
        if new_order <= 1e-9:
            bonds.pop(key)
            sub.bondlist.remove(bond_obj)
            operations.append(f"break_bond:{key}")
        else:
            bond_obj.order = float(new_order)
            bond_obj.kind = sc._order_kind(float(new_order), bond_obj.kind)
            operations.append(f"order_change:{key}:{new_order}")

    for a, b, order in step.bonds_formed:
        if a not in atoms or b not in atoms:
            raise ValueError(f"Unknown atom in bonds_formed: {(a, b)}")
        key = _bond_key(a, b)
        if key in bonds:
            # Increase or overwrite existing bond if specified
            _, bond_obj = bonds[key]
            bond_obj.order = float(order)
            bond_obj.kind = sc._order_kind(float(order), bond_obj.kind)
        else:
            sub = sc._find_substructure(working_pgl, key[0])
            new_bond = sc.Bond(key[0], key[1], sc._order_kind(float(order)), order=float(order))
            sub.bondlist.append(new_bond)
            bonds[key] = (sub, new_bond)
        operations.append(f"form_bond:{key}:{order}")

    for aid, dq in step.charge_changes.items():
        if aid not in atoms:
            raise ValueError(f"Unknown atom in charge_changes: {aid}")
        atoms[aid].charge = int(atoms[aid].charge or 0) + int(dq)
        operations.append(f"charge_delta:{aid}:{dq:+d}")

    for aid, dr in step.radical_changes.items():
        if aid not in new_meta:
            raise ValueError(f"Unknown atom in radical_changes: {aid}")
        new_meta[aid].radicals = max(0, int(new_meta[aid].radicals) + int(dr))
        operations.append(f"radical_delta:{aid}:{dr:+d}")

    for aid, dox in step.oxidation_state_changes.items():
        if aid not in new_meta:
            raise ValueError(f"Unknown atom in oxidation_state_changes: {aid}")
        cur_ox = new_meta[aid].oxidation_state or int(atoms[aid].charge or 0)
        new_meta[aid].oxidation_state = cur_ox + int(dox)

    for aid in step.excited_Clear:
        if aid in new_meta:
            new_meta[aid].excited = False
    for aid in step.excited_Set:
        if aid not in new_meta:
            raise ValueError(f"Unknown atom in excited_Set: {aid}")
        new_meta[aid].excited = True

    if normalize_species:
        working_pgl = normalize_pgl_species(working_pgl)

    after = LabState(
        pgl_state=working_pgl,
        atom_meta=new_meta,
        net_electrode_electrons=before.net_electrode_electrons + int(step.electrode_electrons),
        photons_absorbed=before.photons_absorbed + int(step.photons),
    )

    # 3. Compute exact step diffs & verify physical conservation laws
    _verify_conservation(before, after, step)

    b_atoms, b_bonds, _ = extract_graph(before)
    a_atoms, a_bonds, _ = extract_graph(after)

    formed_rec: List[Tuple[str, str, float]] = []
    broken_rec: List[Tuple[str, str, float]] = []
    order_rec: List[Tuple[str, str, float, float]] = []
    touched: Set[str] = set()

    for key in sorted(set(b_bonds) | set(a_bonds)):
        bo = float(b_bonds[key].order or sc._kind_order(b_bonds[key].kind) or 1.0) if key in b_bonds else 0.0
        ao = float(a_bonds[key].order or sc._kind_order(a_bonds[key].kind) or 1.0) if key in a_bonds else 0.0
        if bo == 0.0 and ao > 0.0:
            formed_rec.append((key[0], key[1], ao))
            touched.update(key)
        elif bo > 0.0 and ao == 0.0:
            broken_rec.append((key[0], key[1], bo))
            touched.update(key)
        elif abs(bo - ao) > 1e-9:
            order_rec.append((key[0], key[1], bo, ao))
            touched.update(key)

    q_rec: Dict[str, Tuple[int, int]] = {}
    rad_rec: Dict[str, Tuple[int, int]] = {}
    for aid in sorted(a_atoms):
        bq = int(b_atoms[aid].charge or 0)
        aq = int(a_atoms[aid].charge or 0)
        if bq != aq:
            q_rec[aid] = (bq, aq)
            touched.add(aid)
        br = int(before.atom_meta[aid].radicals)
        ar = int(after.atom_meta[aid].radicals)
        if br != ar:
            rad_rec[aid] = (br, ar)
            touched.add(aid)

    stereo_events = _propagate_stereochemistry(before, after, step, formed_rec, broken_rec)
    for aid in stereo_events:
        if aid != "__electrocyclic__":
            touched.add(aid)

    record = StepExecutionRecord(
        step_index=step_index,
        step_class=step.step_class,
        name=step.name or f"step_{step_index + 1}",
        bonds_formed=formed_rec,
        bonds_broken=broken_rec,
        bond_order_changes=order_rec,
        charge_changes=q_rec,
        radical_changes=rad_rec,
        stereo_events=stereo_events,
        touched_atoms=tuple(sorted(touched)),
        operations=operations,
    )
    return after, record


def execute_hypothesis(hypothesis: MechanismHypothesis) -> TrajectoryResult:
    """Replay every MechanisticStep of a MechanismHypothesis from its initial LabState."""
    init = hypothesis.initial_state.clone()
    init.pgl_state = normalize_pgl_species(init.pgl_state)
    states: List[LabState] = [init]
    records: List[StepExecutionRecord] = []

    cur = init
    for idx, step in enumerate(hypothesis.steps):
        try:
            nxt, rec = execute_step(cur, step, step_index=idx, normalize_species=True)
        except Exception as exc:
            return TrajectoryResult(
                hypothesis_id=hypothesis.hypothesis_id,
                states=states,
                step_records=records,
                valid=False,
                error=f"{type(exc).__name__}: {exc}",
                failed_step_index=idx,
            )
        states.append(nxt)
        records.append(rec)
        cur = nxt

    return TrajectoryResult(
        hypothesis_id=hypothesis.hypothesis_id,
        states=states,
        step_records=records,
        valid=True,
    )
