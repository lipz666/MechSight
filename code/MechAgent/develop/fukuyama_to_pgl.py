"""Convert FukuyamaBench into the PGL task format used by this repo.

FukuyamaBench (HaCTang/ReactionMechanismReasoning) ships 319 mechanisms from
Fukuyama's workbook as per-step reaction SMILES with atom maps. Two things stop
it from being read directly:

1. It records *states*, not electron-transfer arrows. Bond formations and
   cleavages follow from consecutive states; arrow direction does not, and
   guessing it would invent ground truth. The converter therefore emits a
   per-step bond-change ground truth instead of a ReactionTransformation
   sequence, which both sides of a comparison can compute mechanically.
2. Atom maps are assigned per step, so the same species is renumbered between
   steps - measured at 38.2% of adjacent step pairs. The converter stitches a
   global numbering by matching species across each boundary on their
   map-stripped canonical SMILES and then aligning atoms by substructure match.

Requires RDKit, which the agent runtime does not; run this offline to produce
the dataset, then the runtime only ever sees PGL text.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

_BOND_KIND = {
    Chem.BondType.SINGLE: ("single", 1.0),
    Chem.BondType.DOUBLE: ("double", 2.0),
    Chem.BondType.TRIPLE: ("triple", 3.0),
    Chem.BondType.AROMATIC: ("aromatic", 1.5),
}


def _fragments(smiles: str) -> Optional[List[Chem.Mol]]:
    mols = []
    for part in smiles.split("."):
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            return None
        mols.append(mol)
    return mols


def _stripped(mol: Chem.Mol) -> Tuple[str, Chem.Mol]:
    """Canonical SMILES with atom maps removed, plus the stripped molecule."""

    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(copy), copy


class _GlobalMapper:
    """Assign one stable label per atom across the whole mechanism."""

    def __init__(self) -> None:
        self._next = 0
        self._pool: Dict[str, List[Tuple[Chem.Mol, Dict[int, str]]]] = {}
        self.broken_links = 0

    def _fresh(self, element: str) -> str:
        self._next += 1
        return f"{element}{self._next}"

    def assign_step(
        self, reactants: List[Chem.Mol], products: List[Chem.Mol]
    ) -> Dict[int, str]:
        """Map this step's local atom-map numbers onto global labels."""

        assignment: Dict[int, str] = {}
        pool = {key: list(value) for key, value in self._pool.items()}

        for mol in reactants:
            key, stripped = _stripped(mol)
            carried = pool[key].pop() if pool.get(key) else None
            if carried is not None:
                previous, previous_labels = carried
                _, previous_stripped = _stripped(previous)
                match = stripped.GetSubstructMatch(previous_stripped)
                for previous_index, current_index in enumerate(match):
                    old = previous.GetAtomWithIdx(previous_index).GetAtomMapNum()
                    new = mol.GetAtomWithIdx(current_index).GetAtomMapNum()
                    if old in previous_labels and new:
                        assignment[new] = previous_labels[old]
            for atom in mol.GetAtoms():
                number = atom.GetAtomMapNum()
                if number and number not in assignment:
                    assignment[number] = self._fresh(atom.GetSymbol())

        # A product atom with no reactant counterpart means the per-step map is
        # incomplete. Give it a fresh label so the rest of the path survives,
        # and count it so the caller can judge the case's reliability.
        for mol in products:
            for atom in mol.GetAtoms():
                number = atom.GetAtomMapNum()
                if number and number not in assignment:
                    assignment[number] = self._fresh(atom.GetSymbol())
                    self.broken_links += 1

        carried_pool: Dict[str, List[Tuple[Chem.Mol, Dict[int, str]]]] = defaultdict(list)
        for mol in products:
            labels = {
                atom.GetAtomMapNum(): assignment[atom.GetAtomMapNum()]
                for atom in mol.GetAtoms()
                if atom.GetAtomMapNum()
            }
            carried_pool[_stripped(mol)[0]].append((mol, labels))
        self._pool = dict(carried_pool)
        return assignment


_FREE = "__free__"
HSnapshot = Dict[str, List[str]]  # heavy-atom label, or _FREE, -> hydrogen labels


def _unmapped_hydrogens(atom: Chem.Atom) -> int:
    """Hydrogens on a heavy atom that carry no atom map of their own."""

    count = atom.GetTotalNumHs()
    count += sum(
        1
        for neighbour in atom.GetNeighbors()
        if neighbour.GetAtomicNum() == 1 and not neighbour.GetAtomMapNum()
    )
    return count


def _side_hydrogens(
    mols: List[Chem.Mol], labels: Dict[int, str]
) -> Tuple[Dict[str, int], int]:
    """Per heavy atom hydrogen count on one side of a step, plus free protons."""

    counts: Dict[str, int] = {}
    free = 0
    for mol in mols:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1 and not atom.GetAtomMapNum():
                if atom.GetDegree() == 0:
                    free += 1
                continue
            label = labels.get(atom.GetAtomMapNum())
            if label:
                counts[label] = _unmapped_hydrogens(atom)
    return counts, free


class _HydrogenLedger:
    """Track which heavy atom owns each hydrogen across a whole mechanism.

    The source maps carry heavy atoms only, and the environment never edits a
    recorded Hcount: a proton transfer can only be expressed by moving an
    explicit hydrogen atom. Without this ledger every proton transfer in the
    ground truth - 59% of HumanBenchmark steps - was a charge change the agent
    had no atom to write an arrow to.

    Within a step, hydrogens a heavy atom loses are handed to the heavy atoms
    that gain them, so the same labelled proton moves. Protons are
    indistinguishable, so which of several equivalent hydrogens moves is
    arbitrary; scoring anonymizes hydrogen labels for that reason.
    """

    def __init__(self) -> None:
        self.owner: Dict[str, Optional[str]] = {}
        self._next = 900000
        self._seen_heavy: set = set()
        self.moved: set = set()
        self.introduced_free: List[str] = []
        self.unsourced = 0

    def _fresh(self) -> str:
        self._next += 1
        return f"H{self._next}"

    def owned(self, heavy: str) -> List[str]:
        return sorted(h for h, owner in self.owner.items() if owner == heavy)

    def free(self) -> List[str]:
        return sorted(h for h, owner in self.owner.items() if owner is None)

    def step(
        self,
        reactant_counts: Dict[str, int],
        reactant_free: int,
        product_counts: Dict[str, int],
        product_free: int,
    ) -> Tuple[HSnapshot, HSnapshot]:
        # Reconcile the reactant side with what the ledger holds. A heavy atom
        # seen for the first time brings its hydrogens with it; a shortfall on
        # an atom already tracked means a hydrogen with no traceable source.
        for heavy, want in reactant_counts.items():
            have = self.owned(heavy)
            first_time = heavy not in self._seen_heavy
            for _ in range(max(0, want - len(have))):
                hydrogen = self._fresh()
                self.owner[hydrogen] = heavy
                if not first_time:
                    self.unsourced += 1
            for hydrogen in have[want:]:
                self.owner[hydrogen] = None
                self.moved.add(hydrogen)
            self._seen_heavy.add(heavy)
        free_now = self.free()
        for _ in range(max(0, reactant_free - len(free_now))):
            hydrogen = self._fresh()
            self.owner[hydrogen] = None
            self.introduced_free.append(hydrogen)
        present_free = self.free()[:reactant_free]

        before: HSnapshot = {heavy: self.owned(heavy) for heavy in reactant_counts}
        before[_FREE] = list(present_free)

        # Released hydrogens go first so a transfer between heavy atoms moves
        # one proton rather than parking it free and drawing a different one.
        pool: List[str] = []
        assignment: Dict[str, List[str]] = {}
        for heavy, have in before.items():
            if heavy == _FREE:
                continue
            keep = product_counts.get(heavy, 0)
            assignment[heavy] = have[:keep]
            pool.extend(have[keep:])
        pool.extend(present_free)
        for heavy in sorted(product_counts):
            held = assignment.setdefault(heavy, [])
            while len(held) < product_counts[heavy]:
                if pool:
                    held.append(pool.pop(0))
                else:
                    held.append(self._fresh())
                    if heavy in self._seen_heavy:
                        self.unsourced += 1
            self._seen_heavy.add(heavy)

        for heavy, held in assignment.items():
            for hydrogen in held:
                if hydrogen in self.owner and self.owner[hydrogen] != heavy:
                    self.moved.add(hydrogen)
                self.owner[hydrogen] = heavy
        for hydrogen in pool:
            if self.owner.get(hydrogen) is not None:
                self.moved.add(hydrogen)
            self.owner[hydrogen] = None

        after: HSnapshot = {
            heavy: sorted(held) for heavy, held in assignment.items() if heavy in product_counts
        }
        after[_FREE] = sorted(pool)
        return before, after


def _atom_entries(
    mols: List[Chem.Mol],
    labels: Dict[int, str],
    hydrogens: HSnapshot,
    mobile: set,
) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    for mol in mols:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1 and not atom.GetAtomMapNum():
                continue
            label = labels.get(atom.GetAtomMapNum())
            if not label:
                continue
            owned = hydrogens.get(label, [])
            explicit = [h for h in owned if h in mobile]
            args = [repr(atom.GetSymbol())]
            if len(owned) - len(explicit):
                args.append(f"Hcount={len(owned) - len(explicit)}")
            if atom.GetFormalCharge():
                args.append(f"charge={atom.GetFormalCharge()}")
            entries[label] = f"Atom({', '.join(args)})"
            for hydrogen in explicit:
                entries[hydrogen] = "Atom('H')"
    for hydrogen in hydrogens.get(_FREE, []):
        entries[hydrogen] = "Atom('H', charge=1)"
    return entries


def _bond_entries(
    mols: List[Chem.Mol],
    labels: Dict[int, str],
    hydrogens: HSnapshot,
    mobile: set,
) -> Dict[Tuple[str, str], Tuple[str, float]]:
    bonds: Dict[Tuple[str, str], Tuple[str, float]] = {}
    for mol in mols:
        for bond in mol.GetBonds():
            first = labels.get(bond.GetBeginAtom().GetAtomMapNum())
            second = labels.get(bond.GetEndAtom().GetAtomMapNum())
            if not first or not second:
                continue
            kind, order = _BOND_KIND.get(bond.GetBondType(), ("single", 1.0))
            bonds[tuple(sorted((first, second)))] = (kind, order)
        for atom in mol.GetAtoms():
            label = labels.get(atom.GetAtomMapNum())
            if not label or (atom.GetAtomicNum() == 1 and not atom.GetAtomMapNum()):
                continue
            for hydrogen in hydrogens.get(label, []):
                if hydrogen in mobile:
                    bonds[tuple(sorted((label, hydrogen)))] = ("single", 1.0)
    return bonds


def _charges(
    mols: List[Chem.Mol],
    labels: Dict[int, str],
    hydrogens: HSnapshot,
    mobile: set,
) -> Dict[str, int]:
    charges: Dict[str, int] = {}
    for mol in mols:
        for atom in mol.GetAtoms():
            label = labels.get(atom.GetAtomMapNum())
            if not label:
                continue
            charges[label] = atom.GetFormalCharge()
            for hydrogen in hydrogens.get(label, []):
                if hydrogen in mobile:
                    charges[hydrogen] = 0
    for hydrogen in hydrogens.get(_FREE, []):
        charges[hydrogen] = 1
    return charges


def _state_text(
    atoms: Dict[str, str], bonds: Dict[Tuple[str, str], Tuple[str, float]], role: str
) -> str:
    atomdict = ", ".join(f"{label!r}: {entry}" for label, entry in sorted(atoms.items()))
    bondlist = ", ".join(
        f"Bond({first!r}, {second!r}, {kind!r}, order={order})"
        for (first, second), (kind, order) in sorted(bonds.items())
    )
    return (
        "ReactionState(species=[Species(role=%r, state='stable', "
        "molecule=Molecule(name='', substructures=[Substructure(name='main', "
        "atomdict={%s}, bondlist=[%s])]))])" % (role, atomdict, bondlist)
    )


def render_state(
    mols: List[Chem.Mol],
    labels: Dict[int, str],
    hydrogens: HSnapshot,
    mobile: set,
    *,
    role: str,
) -> str:
    """Serialize one side of a step as a PGL ReactionState."""

    return _state_text(
        _atom_entries(mols, labels, hydrogens, mobile),
        _bond_entries(mols, labels, hydrogens, mobile),
        role,
    )


def _merged_state(
    parts: List[Tuple[Chem.Mol, Dict[int, str], HSnapshot]],
    *,
    role: str,
    mobile: set,
    free: List[str],
) -> str:
    """Render several species, each with its own step's labels, as one state."""

    atoms: Dict[str, str] = {}
    bonds: Dict[Tuple[str, str], Tuple[str, float]] = {}
    for mol, labels, hydrogens in parts:
        bound = {k: v for k, v in hydrogens.items() if k != _FREE}
        atoms.update(_atom_entries([mol], labels, bound, mobile))
        bonds.update(_bond_entries([mol], labels, bound, mobile))
    for hydrogen in free:
        atoms.setdefault(hydrogen, "Atom('H', charge=1)")
    return _state_text(atoms, bonds, role)


def step_changes(
    before: Tuple[Dict[Tuple[str, str], Tuple[str, float]], Dict[str, int]],
    after: Tuple[Dict[Tuple[str, str], Tuple[str, float]], Dict[str, int]],
) -> Dict[str, List[Any]]:
    """The bond and charge changes that define one elementary step."""

    before_bonds, before_charges = before
    after_bonds, after_charges = after
    formed = [
        [first, second, after_bonds[(first, second)][0]]
        for (first, second) in sorted(set(after_bonds) - set(before_bonds))
    ]
    broken = [
        [first, second, before_bonds[(first, second)][0]]
        for (first, second) in sorted(set(before_bonds) - set(after_bonds))
    ]
    reordered = [
        [first, second, before_bonds[(first, second)][0], after_bonds[(first, second)][0]]
        for (first, second) in sorted(set(before_bonds) & set(after_bonds))
        if before_bonds[(first, second)][1] != after_bonds[(first, second)][1]
    ]
    charge_changes = [
        [label, before_charges[label], after_charges[label]]
        for label in sorted(set(before_charges) & set(after_charges))
        if before_charges[label] != after_charges[label]
    ]
    return {
        "bonds_formed": formed,
        "bonds_broken": broken,
        "bond_order_changes": reordered,
        "charge_changes": charge_changes,
    }


def _species_labels(mol: Chem.Mol, labels: Dict[int, str]) -> frozenset:
    return frozenset(
        labels[atom.GetAtomMapNum()]
        for atom in mol.GetAtoms()
        if atom.GetAtomMapNum() in labels
    )


def convert_steps(
    steps: List[Tuple[str, str]],
    *,
    case_id: str,
    partition: str,
    confidences: Optional[List[float]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert a mechanism given as ordered (mapped reactants, mapped products).

    Shared by every RMR source: FukuyamaBench stores the pair as one
    ``mapped_rxn`` string, while oMe and HumanBenchmark store the two sides as
    per-species ``mapped_smiles``. Everything after that - stitching a global
    numbering, accumulating the endpoint states, deriving per-step changes - is
    identical, so only the reader differs.
    """

    mapper = _GlobalMapper()
    walked: List[Tuple[List[Chem.Mol], List[Chem.Mol], Dict[int, str]]] = []
    unmapped_heavy = 0
    for index, (reactant_smiles, product_smiles) in enumerate(steps):
        reactants = _fragments(reactant_smiles)
        products = _fragments(product_smiles)
        if reactants is None or products is None:
            raise ValueError(f"step {index + 1} has unparseable SMILES")
        # HumanBenchmark and oMe number only the atoms that end up in the
        # product, so leaving groups, bases and reagents carry no map and
        # would be silently dropped - leaving a carbon whose substituent is
        # gone, and a mechanism the environment can no longer balance.
        unmapped_heavy += sum(
            1
            for mol in reactants + products
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() > 1 and not atom.GetAtomMapNum()
        )
        walked.append((reactants, products, mapper.assign_step(reactants, products)))
    return _assemble(
        walked,
        mapper,
        unmapped_heavy=unmapped_heavy,
        case_id=case_id,
        partition=partition,
        confidences=confidences or [],
        extra=extra or {},
    )


def convert_case(case_dir: Path) -> Dict[str, Any]:
    """Convert one FukuyamaBench case, or raise ValueError with the reason."""

    mapped = json.loads((case_dir / "mapped_rxn.json").read_text())["mechanism_rxn"]
    steps: List[Tuple[str, str]] = []
    confidences: List[float] = []
    for index, step in enumerate(mapped):
        reaction = step.get("mapped_rxn")
        if not reaction or ">>" not in reaction:
            raise ValueError(f"step {index + 1} has no atom-mapped reaction")
        left, right = reaction.split(">>")
        steps.append((left, right))
        confidence = step.get("confidence")
        if confidence is not None:
            confidences.append(float(confidence))
    return convert_steps(
        steps,
        case_id=case_dir.name,
        partition=case_dir.name[0],
        confidences=confidences,
    )


def _assemble(
    walked: List[Tuple[List[Chem.Mol], List[Chem.Mol], Dict[int, str]]],
    mapper: "_GlobalMapper",
    *,
    case_id: str,
    partition: str,
    confidences: List[float],
    extra: Dict[str, Any],
    unmapped_heavy: int = 0,
) -> Dict[str, Any]:
    ledger = _HydrogenLedger()
    snapshots: List[Tuple[HSnapshot, HSnapshot]] = []
    for reactants, products, labels in walked:
        reactant_counts, reactant_free = _side_hydrogens(reactants, labels)
        product_counts, product_free = _side_hydrogens(products, labels)
        snapshots.append(
            ledger.step(reactant_counts, reactant_free, product_counts, product_free)
        )
    mobile = set(ledger.moved)

    states: List[str] = []
    changes: List[Dict[str, Any]] = []
    for index, ((reactants, products, labels), (before, after)) in enumerate(
        zip(walked, snapshots)
    ):
        if index == 0:
            states.append(render_state(reactants, labels, before, mobile, role="reactant"))
        role = "product" if index == len(walked) - 1 else "intermediate"
        states.append(render_state(products, labels, after, mobile, role=role))
        changes.append(
            step_changes(
                (
                    _bond_entries(reactants, labels, before, mobile),
                    _charges(reactants, labels, before, mobile),
                ),
                (
                    _bond_entries(products, labels, after, mobile),
                    _charges(products, labels, after, mobile),
                ),
            )
        )

    # A reagent is introduced at the step that consumes it and a byproduct is
    # dropped once it stops participating, so both endpoint states have to be
    # accumulated: otherwise the Planner is asked to form bonds to atoms it was
    # never shown, or to reach a target that discards what the mechanism made.
    seen: set = set()
    initial_parts: List[Tuple[Chem.Mol, Dict[int, str], HSnapshot]] = []
    introduced_at: List[int] = []
    for index, (reactants, products, labels) in enumerate(walked):
        for mol in reactants:
            species = _species_labels(mol, labels)
            if species and not (species & seen):
                initial_parts.append((mol, labels, snapshots[index][0]))
                if index > 0:
                    introduced_at.append(index + 1)
        for mols in (reactants, products):
            for mol in mols:
                seen |= _species_labels(mol, labels)

    # Hydrogen ownership in the target is read from the ledger's final state,
    # so a proton that moved after a species' last appearance is not drawn twice.
    final: HSnapshot = {}
    for hydrogen, owner in ledger.owner.items():
        if owner is not None:
            final.setdefault(owner, []).append(hydrogen)
    target_parts: List[Tuple[Chem.Mol, Dict[int, str], HSnapshot]] = []
    for index, (_, products, labels) in enumerate(walked):
        later: set = set()
        for reactants_after, _, labels_after in walked[index + 1 :]:
            for mol in reactants_after:
                later |= _species_labels(mol, labels_after)
        for mol in products:
            species = _species_labels(mol, labels)
            if species and not (species & later):
                target_parts.append((mol, labels, final))

    empty = [i + 1 for i, change in enumerate(changes) if not any(change.values())]
    record = {
        "case_id": case_id,
        "partition": partition,
        "num_steps": len(changes),
        "initial_state_pgl": _merged_state(
            initial_parts, role="reactant", mobile=mobile, free=ledger.introduced_free
        ),
        "target_state_pgl": _merged_state(
            target_parts, role="product", mobile=mobile, free=ledger.free()
        ),
        "states_pgl": states,
        "gt_step_changes": changes,
        "reagents_introduced_after_step_1": sorted(set(introduced_at)),
        "quality": {
            "mapping_confidence_min": min(confidences) if confidences else None,
            "mapping_confidence_mean": (
                sum(confidences) / len(confidences) if confidences else None
            ),
            "low_confidence_steps": sum(1 for c in confidences if c < 0.5),
            "broken_map_links": mapper.broken_links,
            "mobile_hydrogens": len(mobile),
            "unsourced_hydrogens": ledger.unsourced,
            "steps_without_change": empty,
            # An atom or proton with no traceable source cannot be in the
            # accumulated initial state, so the target would be unreachable.
            "unmapped_heavy_atoms": unmapped_heavy,
            "initial_state_complete": (
                mapper.broken_links == 0 and ledger.unsourced == 0 and unmapped_heavy == 0
            ),
        },
    }
    record.update(extra)
    return record


def _read_conditions(case_dir: Path) -> List[Dict[str, str]]:
    try:
        mechanism = json.loads((case_dir / "mechanism.json").read_text())
    except Exception:
        return []
    seen: List[Dict[str, str]] = []
    for reaction in mechanism.get("reactions", []):
        for condition in reaction.get("conditions", []):
            entry = {
                "role": str(condition.get("role", "")),
                "text": str(condition.get("text", "")),
            }
            if entry not in seen:
                seen.append(entry)
    return seen


def convert_all(mechanisms_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    records: List[Dict[str, Any]] = []
    failures: Dict[str, int] = defaultdict(int)
    for case_dir in sorted(p for p in mechanisms_dir.iterdir() if p.is_dir()):
        try:
            record = convert_case(case_dir)
        except Exception as exc:
            failures[str(exc).split(" has ")[-1] or repr(exc)] += 1
            continue
        record["conditions"] = _read_conditions(case_dir)
        records.append(record)
    return records, dict(failures)


def write_core_named(records: List[Dict[str, Any]], core_dir: Path) -> Path:
    """Emit the records in the core named-reaction layout the loader reads.

    The path holds only the accumulated initial and target states. Putting the
    per-step states in would be wrong: they are not accumulated, so the
    environment would take a first state that lacks reagents a later step needs.
    No ReactionTransformation is emitted because this dataset has no arrows, so
    the run must pass --disable-gt-assessment and be scored on bond changes.
    """

    core_dir.mkdir(parents=True, exist_ok=True)
    path = core_dir / "named_reactions_val_pgl_paths.core.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            row = {
                "mechanism_id": f"fukuyama::{record['case_id']}",
                "mechanism_name": record["case_id"],
                "reaction_class": f"fukuyama_{record['partition']}",
                "num_states": 2,
                "num_transformations": 0,
                "conditions": record.get("conditions", []),
                "ReactionPath": "ReactionPath(name=%r, sequence=[%s, %s])"
                % (
                    f"fukuyama_{record['case_id']}",
                    record["initial_state_pgl"],
                    record["target_state_pgl"],
                ),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mechanisms_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--partition",
        default=None,
        help="Keep only cases whose id starts with this letter, e.g. A.",
    )
    parser.add_argument(
        "--core-dir",
        type=Path,
        default=None,
        help="Also write the records as a core named-reaction file in this directory.",
    )
    parser.add_argument(
        "--runnable-only",
        action="store_true",
        help=(
            "Keep only cases whose initial state carries every atom the "
            "mechanism uses; the rest cannot be posed as a task."
        ),
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Drop cases whose worst per-step atom-mapping confidence is below this.",
    )
    args = parser.parse_args()

    records, failures = convert_all(args.mechanisms_dir)
    if args.partition:
        records = [r for r in records if r["partition"] == args.partition]
    if args.runnable_only:
        records = [r for r in records if r["quality"]["initial_state_complete"]]
    if args.min_confidence is not None:
        records = [
            r
            for r in records
            if (r["quality"]["mapping_confidence_min"] or 0.0) >= args.min_confidence
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.core_dir:
        core_path = write_core_named(records, args.core_dir)
        print(f"core dataset -> {core_path}")

    steps = sum(r["num_steps"] for r in records)
    print(f"converted {len(records)} cases, {steps} elementary steps -> {args.output}")
    if failures:
        print("skipped:", dict(failures))


if __name__ == "__main__":
    main()
