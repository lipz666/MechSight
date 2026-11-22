from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schemas import MechanismTaskContext


DEFAULT_SKILLRL_ROOT = os.environ.get("SKILLRL_ROOT", "./external/SkillRL")


class _StrictEnvAdapter:
    """Delegate to SkillRL and add strict matching plus transactional previews."""

    def __init__(
        self,
        adapter: Any,
        strict_state_matches: Any,
        strict_state_signature: Any,
        state_to_pgl: Any,
        apply_local_reaction_transformation: Any,
        iter_atoms: Any,
        hide_target: bool = False,
    ):
        self._hide_target = hide_target
        self._adapter = adapter
        self._strict_state_matches = strict_state_matches
        self._strict_state_signature = strict_state_signature
        self._state_to_pgl = state_to_pgl
        self._apply_local_reaction_transformation = apply_local_reaction_transformation
        self._iter_atoms = iter_atoms

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter, name)

    def product_reached(self) -> bool:
        # In forward mode a true here would tell the agent it hit the hidden
        # product, so completion is left to the Planner's own audit.
        if self._hide_target:
            return False
        return self._strict_state_matches(self.current_state, self.final_state)

    def step(self, action: Dict[str, Any]) -> Tuple[Any, float, bool, Dict[str, Any]]:
        observation, reward, done, info = self._adapter.step(action)
        normalizations = self._normalize_bonded_explicit_h_charge(self.current_state)
        if normalizations:
            info = dict(info or {})
            info["edit_operations"] = list(info.get("edit_operations") or []) + normalizations
            observation = self._state_to_pgl(self.current_state)
        return observation, reward, done, info

    def preview_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply an action to a copy and expose its exact post-state before commit."""

        result = self._apply_local_reaction_transformation(self.current_state, action)
        if not result.valid:
            return {
                "preview_valid": False,
                "preview_errors": [
                    {"error": "preview_edit_error", "detail": str(result.error)}
                ],
                "preview_operations": [str(item) for item in result.operations],
            }

        normalizations = self._normalize_bonded_explicit_h_charge(result.state)
        before = self._strict_state_signature(self.current_state)
        after = self._strict_state_signature(result.state)
        errors: List[Dict[str, Any]] = []
        if before == after:
            errors.append({"error": "preview_produces_no_state_change"})

        # An explicit hydrogen can have at most one covalent partner. Catching
        # this here prevents the common paired-proton-arrow encoding from
        # producing a three-centre H before the real environment is mutated.
        h_degree: Dict[str, int] = {
            name: 0
            for name, atom in after["atoms"].items()
            if atom["element"] == "H"
        }
        for edge in after["bonds"]:
            atom1, atom2 = edge.split("|", 1)
            if atom1 in h_degree:
                h_degree[atom1] += 1
            if atom2 in h_degree:
                h_degree[atom2] += 1
        for atom_name, degree in h_degree.items():
            if degree > 1:
                errors.append(
                    {
                        "error": "explicit_hydrogen_has_multiple_bonds",
                        "atom": atom_name,
                        "degree": degree,
                    }
                )

        return {
            "preview_valid": not errors,
            "preview_errors": errors,
            "preview_operations": [str(item) for item in result.operations] + normalizations,
            "preview_state_changed": before != after,
            "preview_state_pgl": self._state_to_pgl(result.state),
            "preview_delta": self._signature_delta(before, after),
        }

    def _normalize_bonded_explicit_h_charge(self, state: Any) -> List[str]:
        """A represented free H+ becomes neutral after covalent bond formation."""

        bonded_atoms = {
            atom_name
            for edge in self._strict_state_signature(state)["bonds"]
            for atom_name in edge.split("|", 1)
        }
        operations: List[str] = []
        for atom_name, atom in self._iter_atoms(state):
            if (
                str(atom.element) == "H"
                and atom_name in bonded_atoms
                and int(atom.charge or 0) != 0
            ):
                old_charge = int(atom.charge or 0)
                atom.charge = 0
                operations.append(
                    f"normalized bonded explicit H {atom_name} charge {old_charge} -> 0"
                )
        return operations

    @staticmethod
    def _signature_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        before_atoms = before["atoms"]
        after_atoms = after["atoms"]
        atom_changes = {
            name: {"before": before_atoms.get(name), "after": after_atoms.get(name)}
            for name in sorted(set(before_atoms) | set(after_atoms))
            if before_atoms.get(name) != after_atoms.get(name)
        }
        before_bonds = before["bonds"]
        after_bonds = after["bonds"]
        bond_changes = {
            name: {"before": before_bonds.get(name), "after": after_bonds.get(name)}
            for name in sorted(set(before_bonds) | set(after_bonds))
            if before_bonds.get(name) != after_bonds.get(name)
        }
        return {"atom_changes": atom_changes, "bond_changes": bond_changes}


class ChemistryBridge:
    """Thin dynamic bridge to the existing SkillRL chemistry environment."""

    def __init__(self, skillrl_root: str = DEFAULT_SKILLRL_ROOT):
        root = str(Path(skillrl_root).expanduser().resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        self.skillrl_root = root

        try:
            adapters = importlib.import_module("agent_system.develop.adapters")
            parser = importlib.import_module("agent_system.develop.parser")
            validator = importlib.import_module("agent_system.develop.validator")
            chem = importlib.import_module("agent_system.develop.chem_compat")
            runner = importlib.import_module("agent_system.develop.run_react_agent")
            self.using_standalone_runtime = False
        except ModuleNotFoundError as exc:
            # The private SkillRL chemistry is unavailable outside the lab
            # machine. Fall back to develop/standalone_chemistry.py, loaded by
            # path so develop-2 stays independent of the develop package.
            if not str(exc.name or "").startswith("agent_system"):
                raise
            local_path = Path(__file__).resolve().parent.parent / "develop" / "standalone_chemistry.py"
            spec = importlib.util.spec_from_file_location("standalone_chemistry", local_path)
            if spec is None or spec.loader is None:
                raise ModuleNotFoundError(
                    f"standalone chemistry fallback not found at {local_path}"
                ) from exc
            local = importlib.util.module_from_spec(spec)
            # Register before exec: dataclasses resolve their own module via
            # sys.modules while the class body is being processed.
            sys.modules.setdefault(spec.name, local)
            spec.loader.exec_module(local)
            adapters = parser = validator = chem = runner = local
            self.using_standalone_runtime = True

        self.LocalEditEnvAdapter = adapters.LocalEditEnvAdapter
        self.reaction_state_text = adapters.reaction_state_text
        self.extract_action_payload = parser.extract_action_payload
        self.validate_action_fn = validator.validate_action
        self.LocalEditMechanismEnv = chem.LocalEditMechanismEnv
        self.pgl = chem.pgl
        self.Atom = chem.Atom
        self.Bond = chem.Bond
        self.Molecule = chem.Molecule
        self.Species = chem.Species
        self.Substructure = chem.Substructure
        self.ReactionState = chem.ReactionState
        self.load_mechanism_tasks = chem.load_mechanism_tasks
        self.load_core_mechanism_tasks = runner.load_core_mechanism_tasks
        self.apply_local_reaction_transformation = (
            chem.local_edit_envs.apply_local_reaction_transformation
        )

    def wrap_env(self, env: Any, *, hide_target: bool = False) -> Any:
        return _StrictEnvAdapter(
            self.LocalEditEnvAdapter(env),
            self.strict_state_matches,
            self.strict_state_signature,
            self.state_to_pgl,
            self.apply_local_reaction_transformation,
            self.pgl.iter_atoms,
            hide_target=hide_target,
        )

    def make_env(self, task: Any, *, max_env_calls: int, hide_target: bool = False) -> Any:
        # The upstream environment's success check intentionally ignores H,
        # charge, and bond order. Keep it from terminating on that loose check;
        # the local strict adapter and Planner completion audit own termination.
        env = self.LocalEditMechanismEnv(
            [task],
            max_steps=max_env_calls,
            stop_on_product=False,
        )
        return self.wrap_env(env, hide_target=hide_target)

    def strict_state_signature(self, state: Any) -> Dict[str, Any]:
        atoms = {
            name: {
                "element": str(atom.element),
                "Hcount": int(atom.Hcount or 0),
                "charge": int(atom.charge or 0),
            }
            for name, atom in sorted(self.pgl.iter_atoms(state), key=lambda item: item[0])
        }
        kind_orders = {
            "single": 1.0,
            "double": 2.0,
            "triple": 3.0,
            "aromatic": 1.5,
            "transition": 0.5,
        }
        bonds = {}
        for key, bond in sorted(self.pgl.bond_map(state).items(), key=lambda item: item[0]):
            effective_order = bond.order
            if effective_order is None:
                effective_order = kind_orders.get(str(bond.kind))
            bonds["|".join(key)] = {
                "kind": str(bond.kind),
                "order": float(effective_order) if effective_order is not None else None,
            }
        return {"atoms": atoms, "bonds": bonds}

    def strict_state_matches(self, current: Any, target: Any) -> bool:
        return self.strict_state_signature(current) == self.strict_state_signature(target)

    def state_to_pgl(self, state: Any) -> str:
        text = self.reaction_state_text(state)
        if not text:
            raise ValueError(f"cannot convert state to PGL: {type(state)!r}")
        # The upstream atom serializer already omits zero Hcount and includes
        # every nonzero Hcount. Keep this invariant explicit at the bridge.
        for _, atom in self.pgl.iter_atoms(state):
            atom_text = self.pgl.atom_to_pgl(atom)
            if int(atom.Hcount or 0) == 0 and "Hcount=" in atom_text:
                raise ValueError("PGL serializer emitted a zero Hcount")
            if int(atom.Hcount or 0) != 0 and "Hcount=" not in atom_text:
                raise ValueError("PGL serializer omitted a nonzero Hcount")
        return text

    def parse_action(self, action_pgl: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self.extract_action_payload(f"<action>{action_pgl}</action>")

    def validate_action(
        self,
        action: Dict[str, Any],
        current_state: Any,
        legal_action_shapes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        endpoint_errors = []
        atom_map = {name: atom for name, atom in self.pgl.iter_atoms(current_state)}
        current_bond_map = self.pgl.bond_map(current_state)
        current_bonds = set(current_bond_map)
        atom_site_markers = {
            "lone_pair", "lp", "LP", "radical_electron", "unpaired_electron",
            "atom", "proton", "empty_orbital",
        }
        absent_edge_markers = {"new_sigma_bond", "new_dative_bond"}
        used_source_bonds = set()

        def structural_edge(endpoint: Any) -> Optional[Tuple[str, str]]:
            if not isinstance(endpoint, (list, tuple)) or len(endpoint) < 2:
                return None
            atom1, atom2 = str(endpoint[0]), str(endpoint[1])
            if atom2 in atom_site_markers or atom1 not in atom_map or atom2 not in atom_map:
                return None
            return tuple(sorted((atom1, atom2)))

        for index, transfer in enumerate(action.get("electron_transfers") or []):
            source = transfer.get("source") if isinstance(transfer, dict) else None
            target = transfer.get("target") if isinstance(transfer, dict) else None
            if not isinstance(source, (list, tuple)) or len(source) < 2:
                endpoint_errors.append(
                    {
                        "error": "source_endpoint_requires_structural_tuple",
                        "transfer_index": index,
                        "source": source,
                    }
                )
            if not isinstance(target, (list, tuple)) or len(target) < 2:
                endpoint_errors.append(
                    {
                        "error": "target_endpoint_requires_structural_tuple",
                        "transfer_index": index,
                        "target": target,
                    }
                )
            if not isinstance(transfer, dict):
                continue

            source_edge = structural_edge(source)
            if source_edge is not None:
                if source_edge not in current_bonds:
                    endpoint_errors.append(
                        {
                            "error": "source_bond_absent_in_current_state",
                            "transfer_index": index,
                            "source": source,
                        }
                    )
                else:
                    source_marker = str(source[2]) if len(source) >= 3 else ""
                    source_bond = current_bond_map[source_edge]
                    source_order = source_bond.order
                    if source_order is None:
                        source_order = {
                            "single": 1,
                            "double": 2,
                            "triple": 3,
                            "aromatic": 1.5,
                            "transition": 0.5,
                        }.get(str(source_bond.kind))
                    if source_marker in {"pi_bond", "double"} and float(source_order or 0) < 2:
                        endpoint_errors.append(
                            {
                                "error": "source_pi_bond_absent_in_current_state",
                                "transfer_index": index,
                                "source": source,
                                "current_bond_kind": str(source_bond.kind),
                            }
                        )
                if source_edge in used_source_bonds:
                    endpoint_errors.append(
                        {
                            "error": "source_bond_reused_in_same_event",
                            "transfer_index": index,
                            "source": source,
                        }
                    )
                used_source_bonds.add(source_edge)

            target_edge = structural_edge(target)
            target_marker = (
                str(target[2])
                if isinstance(target, (list, tuple)) and len(target) >= 3
                else ""
            )
            if target_edge is not None and target_marker in absent_edge_markers and target_edge in current_bonds:
                endpoint_errors.append(
                    {
                        "error": "new_target_bond_already_exists",
                        "transfer_index": index,
                        "target": target,
                    }
                )

            explicit_h_involved = any(
                str(item) in atom_map and str(atom_map[str(item)].element) == "H"
                for endpoint in (source, target)
                if isinstance(endpoint, (list, tuple))
                for item in endpoint[:2]
            )
            if str(transfer.get("kind")) == "proton_transfer" and explicit_h_involved:
                endpoint_errors.append(
                    {
                        "error": "explicit_h_requires_two_arrow_encoding",
                        "transfer_index": index,
                        "detail": (
                            "For an explicit H atom, encode acceptor lone-pair -> new A-H "
                            "with kind='bond_forming' and donor-H sigma bond -> donor lone-pair "
                            "with kind='sigma_shift'; do not use kind='proton_transfer'."
                        ),
                    }
                )
        if endpoint_errors:
            return {"valid": False, "repairable": True, "errors": endpoint_errors}
        return self.validate_action_fn(action, current_state, legal_action_shapes)

    def task_context(
        self,
        task: Any,
        *,
        input_mode: str = "structures_only",
    ) -> MechanismTaskContext:
        states = [item for item in task.path.sequence if isinstance(item, self.ReactionState)]
        if len(states) < 2:
            raise ValueError(f"task {task.mechanism_id} does not contain initial and target states")
        mechanism_name = str(getattr(task, "mechanism_name", task.mechanism_id))
        reaction_class = str(getattr(task, "reaction_class", "unknown"))
        if input_mode == "structures_only":
            conditions = None
            description = (
                "Using only the provided initial and target PGL ReactionStates, first infer "
                "the most likely reaction family from the structural transformation, then "
                "plan the complete stepwise elementary electron-transfer mechanism. No "
                "reaction name, reaction-class label, or conditions are provided."
            )
            safe_suffix = str(task.mechanism_id).rsplit("::", 1)[-1]
            prompt_reaction_id = f"structures_only::{safe_suffix}"
            metadata = {"input_mode": "structures_only"}
        elif input_mode == "forward":
            # Experiment B: the product side is withheld, leaving the reactant
            # state and the dataset's condition text, as in the RMR pathway
            # prompt. The class label stays hidden.
            conditions = getattr(task, "metadata", {}).get("conditions") or None
            description = (
                "Using only the provided initial PGL ReactionState and the reaction "
                "conditions, infer what reaction occurs, predict its products, and plan "
                "the complete stepwise elementary electron-transfer mechanism. No product "
                "structure, reaction name, or reaction-class label is provided."
            )
            safe_suffix = str(task.mechanism_id).rsplit("::", 1)[-1]
            prompt_reaction_id = f"forward::{safe_suffix}"
            metadata = {"input_mode": "forward"}
        elif input_mode == "metadata":
            conditions = getattr(task, "metadata", {}).get("conditions") or getattr(
                task, "reaction_class", None
            )
            description = (
                "Infer the stepwise elementary electron-transfer mechanism connecting "
                f"the provided initial and target PGL ReactionStates. Reaction description: {mechanism_name}. "
                f"Reaction class or condition label: {reaction_class}."
            )
            prompt_reaction_id = str(task.mechanism_id)
            # Metadata is deliberately whitelisted. Never pass the annotated path,
            # gold transformations, future states, or skill content into runtime prompts.
            metadata = {
                "mechanism_name": mechanism_name,
                "reaction_class": reaction_class,
                "source_path": str(getattr(task, "source_path", "")),
                "split": str(getattr(task, "split", "")),
                "input_mode": "metadata",
            }
        else:
            raise ValueError(f"unsupported task input mode: {input_mode}")
        return MechanismTaskContext(
            reaction_id=prompt_reaction_id,
            task_description=description,
            initial_state_pgl=self.state_to_pgl(states[0]),
            target_state_pgl="" if input_mode == "forward" else self.state_to_pgl(states[-1]),
            conditions=conditions,
            reaction_metadata=metadata,
        )

    def gold_actions_pgl(self, task: Any) -> List[str]:
        """Return annotated actions for post-hoc evaluation only.

        Runtime task construction deliberately excludes this data; callers
        must invoke this method only after a rollout has terminated.
        """

        return [
            str(item)
            for item in task.path.sequence
            if not isinstance(item, self.ReactionState)
            and getattr(item, "electron_transfers", None)
        ]

    def load_tasks(
        self,
        *,
        data_source: str,
        split: str,
        extracted_dir: str,
        core_dir: str,
        core_subset: str,
        max_tasks: Optional[int],
        reaction_id: Optional[str],
        mechfinder_num_classes: Optional[int],
        mechfinder_per_class: int,
    ) -> List[Any]:
        extracted_path = Path(extracted_dir).expanduser()
        core_path = Path(core_dir).expanduser()
        if not extracted_path.is_absolute():
            extracted_path = Path(self.skillrl_root) / extracted_path
        if not core_path.is_absolute():
            core_path = Path(self.skillrl_root) / core_path
        if data_source == "core":
            tasks = self.load_core_mechanism_tasks(
                core_dir=str(core_path),
                subset=core_subset,
                split=split,
                max_tasks=max_tasks,
                mechfinder_num_classes=mechfinder_num_classes,
                mechfinder_per_class=mechfinder_per_class,
            )
        else:
            tasks = self.load_mechanism_tasks(
                split=split,
                max_tasks=max_tasks,
                extracted_dir=str(extracted_path),
            )
        if reaction_id:
            queries = [item.strip() for item in reaction_id.split(",") if item.strip()]
            tasks = [task for task in tasks if any(query in str(task.mechanism_id) for query in queries)]
        return tasks
