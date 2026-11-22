"""Deterministic structural diff between two PGL ReactionStates.

The Planner used to receive two full ~600-token PGL dumps and was asked, in
prose, to "mentally follow the proposed steps from the supplied initial PGL
toward the supplied target PGL and add any missing elementary event". It does
not reliably do that: in the Gemini run 58 of 180 rollouts planned fewer steps
than the annotation and none of those 58 reproduced the annotated events.

The diff below computes the same thing exactly and cheaply, so the required
changes can be stated as a short explicit list instead of being left as an
in-context reading exercise over 35 atoms.

It operates on the strict state signature produced by
:meth:`ChemistryBridge.strict_state_signature`, so it is independent of whether
the runtime uses the private SkillRL chemistry or the standalone one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _is_hydrogen(label: str) -> bool:
    return len(label) > 1 and label[0] == "H" and label[1:].isdigit()


def _bond_atoms(key: str) -> Tuple[str, ...]:
    return tuple(key.split("|"))


@dataclass
class StructuralDiff:
    """What still has to change to turn ``current`` into ``target``."""

    bonds_to_form: List[Dict[str, Any]] = field(default_factory=list)
    bonds_to_break: List[Dict[str, Any]] = field(default_factory=list)
    bond_order_changes: List[Dict[str, Any]] = field(default_factory=list)
    charge_changes: List[Dict[str, Any]] = field(default_factory=list)
    hcount_changes: List[Dict[str, Any]] = field(default_factory=list)
    atoms_only_in_current: List[str] = field(default_factory=list)
    atoms_only_in_target: List[str] = field(default_factory=list)
    # Bond-order changes that redraw a resonance form without changing any
    # atom's total bond order, i.e. a Kekule alternation around an aromatic
    # ring. Reported, but never treated as an outstanding reaction event.
    resonance_bond_order_changes: List[Dict[str, Any]] = field(default_factory=list)
    # Bonds that differ only in which labelled proton occupies the position.
    # Protons are indistinguishable and the annotated H-labels are generated,
    # so these are bookkeeping, not outstanding events.
    hydrogen_relabelings: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def shared_atom_changes(self) -> int:
        """Chemically real changes among atoms present in both states.

        Differences that live entirely on atoms missing from one side are an
        annotation-scope question (an omitted byproduct such as H2). Changes on
        shared atoms are not: they mean the principal product is still wrong.
        Resonance redrawings are excluded - they are the same molecule.
        """

        return (
            len(self.bonds_to_form)
            + len(self.bonds_to_break)
            + len(self.bond_order_changes)
            + len(self.charge_changes)
            + len(self.hcount_changes)
        )

    @property
    def heavy_atom_changes(self) -> int:
        def heavy(atoms) -> bool:
            return not all(_is_hydrogen(atom) for atom in atoms)

        count = 0
        for entry in self.bonds_to_form + self.bonds_to_break + self.bond_order_changes:
            if heavy(entry["atoms"]):
                count += 1
        count += sum(1 for entry in self.charge_changes if heavy([entry["atom"]]))
        return count

    def is_empty(self) -> bool:
        return self.shared_atom_changes == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bonds_to_form": self.bonds_to_form,
            "bonds_to_break": self.bonds_to_break,
            "bond_order_changes": self.bond_order_changes,
            "charge_changes": self.charge_changes,
            "hcount_changes": self.hcount_changes,
            "atoms_only_in_current": self.atoms_only_in_current,
            "atoms_only_in_target": self.atoms_only_in_target,
            "resonance_bond_order_changes": self.resonance_bond_order_changes,
            "hydrogen_relabelings": self.hydrogen_relabelings,
            "shared_atom_change_count": self.shared_atom_changes,
            "heavy_atom_change_count": self.heavy_atom_changes,
        }


def _hydrogen_degrees(bonds: Dict[str, Any]) -> Dict[str, int]:
    degrees: Dict[str, int] = {}
    for key in bonds:
        first, second = _bond_atoms(key)[:2]
        for atom, partner in ((first, second), (second, first)):
            if _is_hydrogen(partner) and not _is_hydrogen(atom):
                degrees[atom] = degrees.get(atom, 0) + 1
    return degrees


def _split_hydrogen_relabelings(
    entries: List[Dict[str, Any]],
    current_degrees: Dict[str, int],
    target_degrees: Dict[str, int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Separate proton-label permutations from real proton transfers.

    A bond ``Cl102-H900000`` that the target wants and the current state lacks
    is not a missing event when the current state already has ``Cl102-H900001``:
    the HCl was formed, with a differently labelled proton. Protons are
    indistinguishable and the annotated H-labels are generated, so the heavy
    atom carrying the same number of hydrogens in both states settles it.

    Without this the gate refuses completion on exactly the cases the metric
    already forgives, and the forced replan then fights a Planner that is
    correctly reporting there is nothing left to do.
    """

    real: List[Dict[str, Any]] = []
    relabelings: List[Dict[str, Any]] = []
    for entry in entries:
        atoms = entry["atoms"][:2]
        hydrogens = [atom for atom in atoms if _is_hydrogen(atom)]
        heavies = [atom for atom in atoms if not _is_hydrogen(atom)]
        if len(hydrogens) == 1 and len(heavies) == 1:
            heavy = heavies[0]
            if current_degrees.get(heavy, 0) == target_degrees.get(heavy, 0):
                relabelings.append(entry)
                continue
        real.append(entry)
    return real, relabelings


def _split_resonance_changes(
    changes: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Separate Kekule redrawings from real bond-order changes.

    A benzene ring drawn with the double bonds in the other three positions is
    the same molecule. Every atom in such a component keeps its total bond
    order, so the whole component sums to zero. A real change - a carbonyl
    reduced to an alcohol - shifts the total on at least one atom.

    Without this split the completion gate fires on 12 rollouts whose only
    remaining difference is which Kekule structure the annotation happened to
    draw, and would push the Planner into inventing steps to "fix" it.
    """

    if not changes:
        return [], []

    parent: Dict[str, str] = {}

    def find(atom: str) -> str:
        parent.setdefault(atom, atom)
        while parent[atom] != atom:
            parent[atom] = parent[parent[atom]]
            atom = parent[atom]
        return atom

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[left_root] = right_root

    for change in changes:
        first, second = change["atoms"][0], change["atoms"][1]
        union(first, second)

    components: Dict[str, List[Dict[str, Any]]] = {}
    for change in changes:
        components.setdefault(find(change["atoms"][0]), []).append(change)

    resonance: List[Dict[str, Any]] = []
    real: List[Dict[str, Any]] = []
    for component in components.values():
        deltas: Dict[str, float] = {}
        for change in component:
            before = change["from_order"]
            after = change["to_order"]
            if before is None or after is None:
                deltas["<unknown>"] = 1.0
                break
            for atom in change["atoms"][:2]:
                deltas[atom] = deltas.get(atom, 0.0) + (float(after) - float(before))
        if all(abs(delta) < 1e-9 for delta in deltas.values()):
            resonance.extend(component)
        else:
            real.extend(component)
    return real, resonance


def diff_signatures(current: Dict[str, Any], target: Dict[str, Any]) -> StructuralDiff:
    """Diff two strict state signatures.

    Only atoms present in both states are compared for bond, charge and Hcount
    changes. An atom present on one side only is reported separately, because
    the annotated target legitimately omits spectators and byproducts.
    """

    current_atoms = current.get("atoms", {})
    target_atoms = target.get("atoms", {})
    shared = set(current_atoms) & set(target_atoms)

    diff = StructuralDiff(
        atoms_only_in_current=sorted(set(current_atoms) - set(target_atoms)),
        atoms_only_in_target=sorted(set(target_atoms) - set(current_atoms)),
    )

    current_bonds = {
        key: value
        for key, value in current.get("bonds", {}).items()
        if set(_bond_atoms(key)) <= shared
    }
    target_bonds = {
        key: value
        for key, value in target.get("bonds", {}).items()
        if set(_bond_atoms(key)) <= shared
    }

    for key in sorted(set(target_bonds) - set(current_bonds)):
        diff.bonds_to_form.append(
            {
                "atoms": list(_bond_atoms(key)),
                "target_kind": target_bonds[key].get("kind"),
                "target_order": target_bonds[key].get("order"),
            }
        )
    for key in sorted(set(current_bonds) - set(target_bonds)):
        diff.bonds_to_break.append(
            {
                "atoms": list(_bond_atoms(key)),
                "current_kind": current_bonds[key].get("kind"),
                "current_order": current_bonds[key].get("order"),
            }
        )
    for key in sorted(set(current_bonds) & set(target_bonds)):
        before = current_bonds[key]
        after = target_bonds[key]
        if before.get("order") != after.get("order"):
            diff.bond_order_changes.append(
                {
                    "atoms": list(_bond_atoms(key)),
                    "from_order": before.get("order"),
                    "to_order": after.get("order"),
                    "from_kind": before.get("kind"),
                    "to_kind": after.get("kind"),
                }
            )

    diff.bond_order_changes, diff.resonance_bond_order_changes = _split_resonance_changes(
        diff.bond_order_changes
    )

    # Hydrogen degrees are counted over the unfiltered bond maps: the proton a
    # mechanism actually moved may carry a label the other state never uses.
    current_degrees = _hydrogen_degrees(current.get("bonds", {}))
    target_degrees = _hydrogen_degrees(target.get("bonds", {}))
    diff.bonds_to_form, formed_relabelings = _split_hydrogen_relabelings(
        diff.bonds_to_form, current_degrees, target_degrees
    )
    diff.bonds_to_break, broken_relabelings = _split_hydrogen_relabelings(
        diff.bonds_to_break, current_degrees, target_degrees
    )
    diff.hydrogen_relabelings = formed_relabelings + broken_relabelings

    for atom in sorted(shared):
        before = current_atoms[atom]
        after = target_atoms[atom]
        if int(before.get("charge", 0)) != int(after.get("charge", 0)):
            diff.charge_changes.append(
                {
                    "atom": atom,
                    "element": after.get("element"),
                    "from_charge": int(before.get("charge", 0)),
                    "to_charge": int(after.get("charge", 0)),
                }
            )
        if int(before.get("Hcount", 0)) != int(after.get("Hcount", 0)):
            diff.hcount_changes.append(
                {
                    "atom": atom,
                    "element": after.get("element"),
                    "from_hcount": int(before.get("Hcount", 0)),
                    "to_hcount": int(after.get("Hcount", 0)),
                }
            )

    return diff


def render_diff(diff: StructuralDiff, *, heading: str = "Required structural changes") -> str:
    """Render the diff as a compact prompt block.

    Roughly 20 lines replace the reading exercise over two 600-token states.
    """

    lines: List[str] = [f"{heading}:"]
    if diff.is_empty() and not diff.atoms_only_in_current and not diff.atoms_only_in_target:
        lines.append("  none; the two states already agree on every shared atom")
        return "\n".join(lines)

    def bond(entry: Dict[str, Any]) -> str:
        return "-".join(entry["atoms"])

    if diff.bonds_to_form:
        lines.append("  bonds to form:")
        for entry in diff.bonds_to_form:
            lines.append(f"    {bond(entry)} as {entry['target_kind']}")
    if diff.bonds_to_break:
        lines.append("  bonds to break:")
        for entry in diff.bonds_to_break:
            lines.append(f"    {bond(entry)} currently {entry['current_kind']}")
    if diff.bond_order_changes:
        lines.append("  bond order changes:")
        for entry in diff.bond_order_changes:
            lines.append(f"    {bond(entry)} {entry['from_kind']} -> {entry['to_kind']}")
    if diff.charge_changes:
        lines.append("  formal charge changes:")
        for entry in diff.charge_changes:
            lines.append(
                f"    {entry['atom']} {entry['from_charge']:+d} -> {entry['to_charge']:+d}"
            )
    if diff.hcount_changes:
        lines.append("  recorded Hcount changes:")
        for entry in diff.hcount_changes:
            lines.append(
                f"    {entry['atom']} Hcount {entry['from_hcount']} -> {entry['to_hcount']}"
            )
    if diff.hydrogen_relabelings:
        lines.append(
            "  proton-label differences only (the heavy atom already carries the right "
            "number of hydrogens; do not plan a transfer for these): "
            + ", ".join(bond(entry) for entry in diff.hydrogen_relabelings)
        )
    if diff.resonance_bond_order_changes:
        lines.append(
            "  resonance-only redrawings (same molecule, no event required; do not plan "
            "a step for these): "
            + ", ".join(bond(entry) for entry in diff.resonance_bond_order_changes)
        )
    if diff.atoms_only_in_current:
        lines.append(
            "  atoms present now but absent from the target (possible omitted byproduct "
            "or spectator; do not invent a deletion step): "
            + ", ".join(diff.atoms_only_in_current)
        )
    if diff.atoms_only_in_target:
        lines.append(
            "  atoms required by the target but absent now (a reagent fragment must still "
            "be introduced by a real event): " + ", ".join(diff.atoms_only_in_target)
        )
    lines.append(
        f"  totals: {diff.shared_atom_changes} change(s) on shared atoms, "
        f"{diff.heavy_atom_changes} of them involving a heavy atom"
    )
    return "\n".join(lines)


def boundary_completion_allowed(diff: StructuralDiff) -> Tuple[bool, str]:
    """Whether a boundary-difference completion may be accepted.

    The Gemini run ended 156 of 180 rollouts on the model's own
    ``CONFIRM_COMPLETE_WITH_BOUNDARY_DIFFERENCE`` verdict, and only 24 on a
    strict structural match. Nothing checked that verdict. This gate does: an
    omitted byproduct is a genuine annotation boundary, an unformed bond on the
    principal product is not.
    """

    if diff.shared_atom_changes == 0:
        return True, "no remaining change on any atom shared with the target"
    return (
        False,
        f"{diff.shared_atom_changes} structural change(s) remain on atoms shared with the "
        f"target ({diff.heavy_atom_changes} involving a heavy atom); this is a missing "
        "reaction event, not an annotation boundary",
    )


def _action_atoms(action: Any) -> List[str]:
    transfers = (
        action.get("electron_transfers")
        if isinstance(action, dict)
        else getattr(action, "electron_transfers", None)
    ) or []
    atoms: List[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
        elif isinstance(value, str) and value and (value[0].isupper() or value[0] == "H"):
            atoms.append(value)

    for transfer in transfers:
        for name in ("source", "target"):
            value = transfer.get(name) if isinstance(transfer, dict) else getattr(transfer, name, None)
            collect(value)
    return atoms


def proposal_relevance(action: Any, diff: StructuralDiff) -> Dict[str, Any]:
    """Which outstanding required changes a proposed action actually touches.

    This is the evidence the reviewer never had. Its only reference used to be
    the plan it had written itself, and it approved 683 of 683 proposals. This
    answers a question the plan cannot: does the proposed action move any change
    that the two PGL graphs say still has to happen?

    Not a veto. Activation, coordination, proton shuttling and resonance steps
    legitimately touch nothing on the list while enabling a later step; the
    reviewer is asked to name that later step when it approves one.
    """

    touched = set(_action_atoms(action))
    addressed: List[str] = []

    for entry in diff.bonds_to_form:
        if touched & set(entry["atoms"]):
            addressed.append(f"form {'-'.join(entry['atoms'])}")
    for entry in diff.bonds_to_break:
        if touched & set(entry["atoms"]):
            addressed.append(f"break {'-'.join(entry['atoms'])}")
    for entry in diff.bond_order_changes:
        if touched & set(entry["atoms"]):
            addressed.append(f"bond order {'-'.join(entry['atoms'])}")
    for entry in diff.charge_changes:
        if entry["atom"] in touched:
            addressed.append(f"charge {entry['atom']}")
    for entry in diff.hcount_changes:
        if entry["atom"] in touched:
            addressed.append(f"Hcount {entry['atom']}")

    return {
        "atoms_touched": sorted(touched),
        "required_changes_addressed": addressed,
        "addresses_any_required_change": bool(addressed),
        "remaining_required_changes": diff.shared_atom_changes,
        "note": (
            "Touching no required change is not by itself an error: activation, "
            "coordination, proton shuttling and resonance steps enable later ones. "
            "Approving such a step requires naming the later step it enables."
        ),
    }
