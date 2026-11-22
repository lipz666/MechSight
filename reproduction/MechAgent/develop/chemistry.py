from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schemas import MechanismTaskContext


DEFAULT_SKILLRL_ROOT = os.environ.get("SKILLRL_ROOT", "./external/SkillRL")


class _StrictEnvAdapter:
    """Delegate to SkillRL's adapter but use a chemistry-complete product match."""

    def __init__(self, adapter: Any, strict_state_matches: Any):
        self._adapter = adapter
        self._strict_state_matches = strict_state_matches

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter, name)

    def product_reached(self) -> bool:
        return self._strict_state_matches(self.current_state, self.final_state)


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
            if not str(exc.name or "").startswith("agent_system"):
                raise
            from . import standalone_chemistry as local

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

    def wrap_env(self, env: Any) -> Any:
        return _StrictEnvAdapter(self.LocalEditEnvAdapter(env), self.strict_state_matches)

    def make_env(self, task: Any, *, max_env_calls: int) -> Any:
        # The upstream environment's success check intentionally ignores H,
        # charge, and bond order. Keep it from terminating on that loose check;
        # the local strict adapter and Planner completion audit own termination.
        env = self.LocalEditMechanismEnv(
            [task],
            max_steps=max_env_calls,
            stop_on_product=False,
        )
        return self.wrap_env(env)

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
        if endpoint_errors:
            return {"valid": False, "repairable": True, "errors": endpoint_errors}
        return self.validate_action_fn(action, current_state, legal_action_shapes)

    def task_context(
        self,
        task: Any,
        *,
        input_mode: str = "structures_only",
        oracle_step_count: bool = False,
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
        step_count = None
        if oracle_step_count:
            # Deliberate, opt-in leak for the step-convention ablation. Only the
            # count crosses over; the annotated actions themselves never do.
            step_count = len(self.gold_actions_pgl(task))
            metadata = dict(metadata)
            metadata["oracle_step_count"] = step_count
        return MechanismTaskContext(
            reaction_id=prompt_reaction_id,
            task_description=description,
            initial_state_pgl=self.state_to_pgl(states[0]),
            target_state_pgl=self.state_to_pgl(states[-1]),
            conditions=conditions,
            reaction_metadata=metadata,
            oracle_step_count=step_count,
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
