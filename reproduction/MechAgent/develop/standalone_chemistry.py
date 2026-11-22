from __future__ import annotations

import ast
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class Atom:
    element: str
    Hcount: int = 0
    charge: int = 0

    def __repr__(self) -> str:
        args = [repr(self.element)]
        if self.Hcount:
            args.append(f"Hcount={self.Hcount}")
        if self.charge:
            args.append(f"charge={self.charge}")
        return f"Atom({', '.join(args)})"


@dataclass
class Bond:
    atom1: str
    atom2: str
    kind: str
    order: Optional[float] = None

    def __post_init__(self) -> None:
        if self.order is None:
            self.order = _kind_order(self.kind)

    def __repr__(self) -> str:
        base = f"Bond({self.atom1!r}, {self.atom2!r}, {self.kind!r}"
        default = _kind_order(self.kind)
        if self.order is not None and self.order != default:
            base += f", order={_number(self.order)}"
        elif self.order is not None and self.kind not in _KIND_ORDERS:
            base += f", order={_number(self.order)}"
        return base + ")"


@dataclass
class Substructure:
    name: str
    atomdict: Dict[str, Atom]
    bondlist: List[Bond]

    def __repr__(self) -> str:
        return (
            f"Substructure(name={self.name!r}, atomdict={self.atomdict!r}, "
            f"bondlist={self.bondlist!r})"
        )


@dataclass
class Molecule:
    name: str
    substructures: List[Substructure]

    def __repr__(self) -> str:
        return f"Molecule(name={self.name!r}, substructures={self.substructures!r})"


@dataclass
class Species:
    role: str
    state: str
    molecule: Molecule

    def __repr__(self) -> str:
        return (
            f"Species(role={self.role!r}, state={self.state!r}, "
            f"molecule={self.molecule!r})"
        )


@dataclass
class ReactionState:
    species: List[Species]

    def __repr__(self) -> str:
        return f"ReactionState(species={self.species!r})"


@dataclass
class ElectronTransfer:
    source: Any
    target: Any
    electrons: int
    kind: str

    def __repr__(self) -> str:
        return (
            f"ElectronTransfer(source={self.source!r}, target={self.target!r}, "
            f"electrons={self.electrons!r}, kind={self.kind!r})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "electrons": self.electrons,
            "kind": self.kind,
        }


@dataclass
class ReactionTransformation:
    transformation_type: str
    electron_transfers: List[ElectronTransfer]

    def __repr__(self) -> str:
        return (
            "ReactionTransformation("
            f"transformation_type={self.transformation_type!r}, "
            f"electron_transfers={self.electron_transfers!r})"
        )

    __str__ = __repr__

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transformation_type": self.transformation_type,
            "electron_transfers": [item.to_dict() for item in self.electron_transfers],
        }


@dataclass
class ReactionPath:
    name: str
    sequence: List[Any]


@dataclass
class MechanismTask:
    mechanism_id: str
    mechanism_name: str
    reaction_class: str
    path: ReactionPath
    source_path: str
    split: str
    metadata: Dict[str, Any] = field(default_factory=dict)


_CONSTRUCTORS = {
    "Atom": Atom,
    "Bond": Bond,
    "Substructure": Substructure,
    "Molecule": Molecule,
    "Species": Species,
    "ReactionState": ReactionState,
    "ElectronTransfer": ElectronTransfer,
    "ReactionTransformation": ReactionTransformation,
    "ReactionPath": ReactionPath,
}


def parse_pgl_literal(text: str) -> Any:
    """Parse the restricted constructor-only PGL expression without eval()."""

    root = ast.parse(str(text).strip(), mode="eval")

    def convert(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.List):
            return [convert(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(convert(item) for item in node.elts)
        if isinstance(node, ast.Dict):
            return {convert(key): convert(value) for key, value in zip(node.keys, node.values)}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = convert(node.operand)
            if not isinstance(value, (int, float)):
                raise ValueError("PGL unary operators are allowed only for numbers")
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            constructor = _CONSTRUCTORS.get(node.func.id)
            if constructor is None:
                raise ValueError(f"unsupported PGL constructor: {node.func.id}")
            if any(keyword.arg is None for keyword in node.keywords):
                raise ValueError("PGL **kwargs are not allowed")
            args = [convert(item) for item in node.args]
            kwargs = {str(item.arg): convert(item.value) for item in node.keywords}
            return constructor(*args, **kwargs)
        raise ValueError(f"unsupported PGL syntax: {type(node).__name__}")

    return convert(root.body)


_KIND_ORDERS = {
    "single": 1.0,
    "double": 2.0,
    "triple": 3.0,
    "aromatic": 1.5,
    "transition": 0.5,
    "dative": 1.0,
}


def _kind_order(kind: str) -> Optional[float]:
    return _KIND_ORDERS.get(str(kind))


def _order_kind(order: float, previous: Optional[str] = None) -> str:
    if abs(order - 1.0) < 1e-9:
        return "single"
    if abs(order - 2.0) < 1e-9:
        return "double"
    if abs(order - 3.0) < 1e-9:
        return "triple"
    if abs(order - 1.5) < 1e-9:
        return "aromatic"
    if abs(order - 0.5) < 1e-9:
        return "transition"
    return previous or "single"


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _substructures(state: ReactionState) -> Iterable[Substructure]:
    for species in state.species:
        for substructure in species.molecule.substructures:
            yield substructure


class _PGLHelpers:
    @staticmethod
    def iter_atoms(state: ReactionState) -> Iterable[Tuple[str, Atom]]:
        for substructure in _substructures(state):
            yield from substructure.atomdict.items()

    @staticmethod
    def bond_map(state: ReactionState) -> Dict[Tuple[str, str], Bond]:
        result: Dict[Tuple[str, str], Bond] = {}
        for substructure in _substructures(state):
            for bond in substructure.bondlist:
                result[_bond_key(bond.atom1, bond.atom2)] = bond
        return result

    @staticmethod
    def atom_to_pgl(atom: Atom) -> str:
        return repr(atom)


pgl = _PGLHelpers()


def reaction_state_text(state: ReactionState) -> str:
    return repr(state)


def _bond_key(atom1: str, atom2: str) -> Tuple[str, str]:
    return tuple(sorted((str(atom1), str(atom2))))


def _atom_map(state: ReactionState) -> Dict[str, Atom]:
    result: Dict[str, Atom] = {}
    for name, atom in pgl.iter_atoms(state):
        if name in result:
            raise ValueError(f"duplicate atom label in ReactionState: {name}")
        result[name] = atom
    return result


def _bond_locations(state: ReactionState) -> Dict[Tuple[str, str], Tuple[Substructure, Bond]]:
    result: Dict[Tuple[str, str], Tuple[Substructure, Bond]] = {}
    for substructure in _substructures(state):
        for bond in substructure.bondlist:
            key = _bond_key(bond.atom1, bond.atom2)
            if key in result:
                raise ValueError(f"duplicate bond in ReactionState: {key}")
            result[key] = (substructure, bond)
    return result


def _endpoint(endpoint: Any, atoms: Dict[str, Atom]) -> Tuple[str, Tuple[str, ...]]:
    if isinstance(endpoint, str):
        if endpoint not in atoms:
            raise ValueError(f"unknown atom endpoint: {endpoint}")
        return "atom", (endpoint,)
    if not isinstance(endpoint, (tuple, list)) or not endpoint:
        raise ValueError(f"invalid electron-transfer endpoint: {endpoint!r}")
    values = tuple(str(item) for item in endpoint)
    if len(values) >= 3:
        if values[0] not in atoms or values[1] not in atoms:
            raise ValueError(f"unknown bond endpoint atoms: {endpoint!r}")
        return "bond", values[:2]
    if len(values) == 2 and values[0] in atoms and values[1] in atoms:
        return "bond", values
    if values[0] not in atoms:
        raise ValueError(f"unknown atom endpoint: {values[0]}")
    return "atom", (values[0],)


_IMPLICIT_BOND_KINDS = {
    "bond_forming",
    "bond_forming_or_pi_shift",
    "proton_transfer",
    "coordination",
    "pi_bond_forming",
    "oxidative_addition",
}


def _normalized_transfers(action: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = action.get("electron_transfers")
    if not isinstance(raw, list) or not raw:
        raise ValueError("electron_transfers must be a non-empty list")
    transfers: List[Dict[str, Any]] = []
    for index, item in enumerate(raw):
        if isinstance(item, ElectronTransfer):
            item = item.to_dict()
        if not isinstance(item, dict):
            raise ValueError(f"electron_transfers[{index}] must be an object")
        electrons = item.get("electrons")
        if not isinstance(electrons, int) or electrons not in {1, 2}:
            raise ValueError(f"electron_transfers[{index}] has invalid electrons")
        transfers.append(
            {
                "source": item.get("source"),
                "target": item.get("target"),
                "electrons": electrons,
                "kind": str(item.get("kind") or "unknown"),
            }
        )
    return transfers


def apply_action(state: ReactionState, action: Dict[str, Any]) -> Tuple[ReactionState, List[str]]:
    updated = copy.deepcopy(state)
    atoms = _atom_map(updated)
    bonds = _bond_locations(updated)
    bond_deltas: Dict[Tuple[str, str], float] = {}
    charge_deltas: Dict[str, float] = {name: 0.0 for name in atoms}
    operations: List[str] = []

    for transfer in _normalized_transfers(action):
        electrons = int(transfer["electrons"])
        pair_order = electrons / 2.0
        source_type, source_atoms = _endpoint(transfer["source"], atoms)
        target_type, target_atoms = _endpoint(transfer["target"], atoms)

        if (
            source_type == "atom"
            and target_type == "atom"
            and transfer["kind"] in _IMPLICIT_BOND_KINDS
        ):
            target_type = "bond"
            target_atoms = (source_atoms[0], target_atoms[0])

        if source_type == "atom":
            charge_deltas[source_atoms[0]] += electrons
            operations.append(f"remove_{electrons}e_from_atom:{source_atoms[0]}")
        else:
            key = _bond_key(source_atoms[0], source_atoms[1])
            current_order = (
                bonds[key][1].order if key in bonds else 0.0
            ) or 0.0
            pending_order = current_order + bond_deltas.get(key, 0.0)
            if pending_order + 1e-9 < pair_order:
                raise ValueError(f"source bond does not have enough order: {key}")
            bond_deltas[key] = bond_deltas.get(key, 0.0) - pair_order
            charge_deltas[source_atoms[0]] += pair_order
            charge_deltas[source_atoms[1]] += pair_order
            operations.append(f"decrease_bond:{key}:{_number(pair_order)}")

        if target_type == "atom":
            charge_deltas[target_atoms[0]] -= electrons
            operations.append(f"add_{electrons}e_to_atom:{target_atoms[0]}")
        else:
            if target_atoms[0] == target_atoms[1]:
                raise ValueError("cannot form a bond from an atom to itself")
            key = _bond_key(target_atoms[0], target_atoms[1])
            bond_deltas[key] = bond_deltas.get(key, 0.0) + pair_order
            charge_deltas[target_atoms[0]] -= pair_order
            charge_deltas[target_atoms[1]] -= pair_order
            operations.append(f"increase_bond:{key}:{_number(pair_order)}")

    for name, delta in charge_deltas.items():
        rounded = round(delta)
        if abs(delta - rounded) > 1e-9:
            raise ValueError(f"fractional formal-charge change for {name}: {delta}")
        atoms[name].charge = int(atoms[name].charge or 0) + int(rounded)

    for key, delta in bond_deltas.items():
        if key in bonds:
            substructure, bond = bonds[key]
            old_order = float(bond.order or _kind_order(bond.kind) or 0.0)
        else:
            old_order = 0.0
            substructure = _find_substructure(updated, key[0])
            bond = Bond(key[0], key[1], "single", order=0.0)
        new_order = old_order + delta
        if new_order < -1e-9 or new_order > 3.0 + 1e-9:
            raise ValueError(f"invalid resulting bond order for {key}: {new_order}")
        if new_order <= 1e-9:
            if key in bonds:
                substructure.bondlist.remove(bond)
            continue
        bond.order = new_order
        bond.kind = _order_kind(new_order, bond.kind)
        if key not in bonds:
            substructure.bondlist.append(bond)

    return updated, operations


def _find_substructure(state: ReactionState, atom_name: str) -> Substructure:
    for substructure in _substructures(state):
        if atom_name in substructure.atomdict:
            return substructure
    raise ValueError(f"cannot locate atom substructure: {atom_name}")


def extract_action_payload(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    value = str(text).strip()
    start = value.lower().find("<action>")
    end = value.lower().rfind("</action>")
    if start >= 0 and end > start:
        value = value[start + len("<action>") : end].strip()
    try:
        parsed = parse_pgl_literal(value)
        if not isinstance(parsed, ReactionTransformation):
            raise ValueError("action must be a ReactionTransformation")
        return parsed.to_dict(), {"valid": True, "errors": []}
    except Exception as exc:
        return {}, {
            "valid": False,
            "repairable": True,
            "errors": [{"error": "action_parse_error", "detail": repr(exc)}],
        }


def validate_action(
    action: Dict[str, Any],
    current_state: ReactionState,
    legal_action_shapes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    del legal_action_shapes
    try:
        apply_action(current_state, action)
    except Exception as exc:
        return {
            "valid": False,
            "repairable": True,
            "errors": [{"error": "invalid_local_edit", "detail": repr(exc)}],
        }
    return {"valid": True, "repairable": False, "errors": []}


class LocalEditMechanismEnv:
    def __init__(self, tasks: List[MechanismTask], *, max_steps: int, stop_on_product: bool):
        if len(tasks) != 1:
            raise ValueError("standalone environment requires exactly one task")
        self.task = tasks[0]
        self.max_steps = int(max_steps)
        self.stop_on_product = bool(stop_on_product)
        states = [item for item in self.task.path.sequence if isinstance(item, ReactionState)]
        if len(states) < 2:
            raise ValueError("mechanism task requires initial and final states")
        self.initial_state = states[0]
        self.final_state = states[-1]
        self.current_state = copy.deepcopy(self.initial_state)
        self.step_count = 0

    def reset(self) -> Tuple[ReactionState, Dict[str, Any]]:
        self.current_state = copy.deepcopy(self.initial_state)
        self.step_count = 0
        return self.current_state, {"legal_local_actions": self.legal_local_actions()}

    def legal_local_actions(self) -> List[Dict[str, Any]]:
        return [
            {
                "source": "('AtomLabel', 'lone_pair') or ('A', 'B', 'bond_marker')",
                "target": "('AtomLabel', 'orbital') or ('A', 'B', 'new_sigma_bond')",
                "electrons": "1 or 2",
            }
        ]

    def state_signature(self) -> Dict[str, Any]:
        return state_signature(self.current_state)

    def product_reached(self) -> bool:
        return state_signature(self.current_state) == state_signature(self.final_state)

    def step(self, action: Dict[str, Any]) -> Tuple[ReactionState, float, bool, Dict[str, Any]]:
        self.step_count += 1
        try:
            updated, operations = apply_action(self.current_state, action)
        except Exception as exc:
            done = self.step_count >= self.max_steps
            return self.current_state, 0.0, done, {
                "edit_valid": False,
                "edit_operations": [],
                "edit_error": repr(exc),
            }
        self.current_state = updated
        reached = self.product_reached()
        done = self.step_count >= self.max_steps or (self.stop_on_product and reached)
        return self.current_state, float(reached), done, {
            "edit_valid": True,
            "edit_operations": operations,
        }


class LocalEditEnvAdapter:
    def __init__(self, env: LocalEditMechanismEnv):
        self._env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


def state_signature(state: ReactionState) -> Dict[str, Any]:
    atoms = {
        name: {
            "element": atom.element,
            "Hcount": int(atom.Hcount or 0),
            "charge": int(atom.charge or 0),
        }
        for name, atom in sorted(pgl.iter_atoms(state), key=lambda item: item[0])
    }
    bonds = {
        "|".join(key): {
            "kind": _order_kind(float(bond.order or _kind_order(bond.kind) or 0.0), bond.kind),
            "order": float(bond.order or _kind_order(bond.kind) or 0.0),
        }
        for key, bond in sorted(pgl.bond_map(state).items())
    }
    return {"atoms": atoms, "bonds": bonds}


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _named_task(row: Dict[str, Any], path: Path, split: str) -> MechanismTask:
    return MechanismTask(
        mechanism_id=str(row["mechanism_id"]),
        mechanism_name=str(row["mechanism_name"]),
        reaction_class=str(row["reaction_class"]),
        path=parse_pgl_literal(str(row["ReactionPath"])),
        source_path=str(path),
        split=split,
        # Only converted external datasets carry this; forward mode reads it.
        metadata={"conditions": row["conditions"]} if row.get("conditions") else {},
    )


def load_core_mechanism_tasks(
    *,
    core_dir: str,
    subset: str,
    split: str,
    max_tasks: Optional[int],
    mechfinder_num_classes: Optional[int],
    mechfinder_per_class: int,
) -> List[MechanismTask]:
    root = Path(core_dir).expanduser().resolve()
    tasks: List[MechanismTask] = []

    if subset in {"named_train", "all"}:
        path = root / "named_reactions_train_pgl_paths.core.jsonl"
        tasks.extend(_named_task(row, path, "train") for row in _load_jsonl(path))
    if subset in {"named_val", "all"}:
        path = root / "named_reactions_val_pgl_paths.core.jsonl"
        tasks.extend(_named_task(row, path, "val") for row in _load_jsonl(path))
    if subset in {"mechfinder", "all"}:
        path = root / "mechfinder_merged_pgl_paths.core.jsonl"
        rows_by_class: Dict[str, List[Dict[str, Any]]] = {}
        for row in _load_jsonl(path):
            reaction_class = str(row["reaction_class"])
            bucket = rows_by_class.setdefault(reaction_class, [])
            if len(bucket) < int(mechfinder_per_class):
                bucket.append(row)
        classes = sorted(rows_by_class)
        if mechfinder_num_classes is not None:
            classes = classes[: int(mechfinder_num_classes)]
        for reaction_class in classes:
            for row in rows_by_class[reaction_class]:
                source_index = int(row["source_row_index"])
                tasks.append(
                    MechanismTask(
                        mechanism_id=f"mechfinder::{reaction_class}::{source_index}",
                        mechanism_name=reaction_class,
                        reaction_class=reaction_class,
                        path=parse_pgl_literal(str(row["mechanism_graph"])),
                        source_path=str(path),
                        split=split,
                        metadata={
                            "source_row_index": source_index,
                            "source_reaction_smiles": row.get("source_reaction_smiles"),
                            "merged_step_count": row.get("merged_step_count"),
                        },
                    )
                )
    return tasks[:max_tasks] if max_tasks is not None else tasks


def load_mechanism_tasks(
    *, split: str, max_tasks: Optional[int], extracted_dir: str
) -> List[MechanismTask]:
    del split, max_tasks, extracted_dir
    raise ValueError("standalone runtime supports only --data-source core")


@dataclass
class LocalEditResult:
    """Transactional preview result used by the develop-2 strict adapter.

    The private runtime returns an object rather than raising, so the caller can
    inspect a rejected edit without losing the current state.
    """

    valid: bool
    state: Optional[ReactionState]
    operations: List[str]
    error: Optional[str] = None


def apply_local_reaction_transformation(
    state: ReactionState, action: Dict[str, Any]
) -> LocalEditResult:
    try:
        updated, operations = apply_action(state, action)
    except Exception as exc:  # the preview reports failure, it does not raise
        return LocalEditResult(False, None, [], repr(exc))
    return LocalEditResult(True, updated, operations, None)


class _LocalEditEnvs:
    """Namespace shim matching ``chem_compat.local_edit_envs``."""

    apply_local_reaction_transformation = staticmethod(apply_local_reaction_transformation)


local_edit_envs = _LocalEditEnvs()
