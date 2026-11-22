from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_ATOM_RE = re.compile(r"'([^']+)'\s*:\s*Atom\('([^']+)'([^)]*)\)")
_BOND_RE = re.compile(
    r"Bond\('([^']+)'\s*,\s*'([^']+)'\s*,\s*'([^']+)'(?:\s*,\s*order=([0-9.]+))?"
)


@dataclass(frozen=True)
class PlannerSkill:
    name: str
    when_to_use: str
    content: str
    source_path: str
    required_elements: tuple[str, ...] = ()

    def catalog_entry(self) -> Dict[str, str]:
        return {"name": self.name, "when_to_use": self.when_to_use}

    def injection_entry(self) -> Dict[str, str]:
        return {"name": self.name, "content": self.content}


def _frontmatter_value(frontmatter: str, key: str) -> str:
    match = re.search(
        rf"(?m)^{re.escape(key)}:\s*(.+?)\s*$",
        frontmatter,
    )
    if not match:
        raise ValueError(f"missing skill frontmatter field: {key}")
    value = match.group(1).strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        value = value[1:-1].strip()
    if not value:
        raise ValueError(f"empty skill frontmatter field: {key}")
    return value


def _optional_frontmatter_value(frontmatter: str, key: str) -> Optional[str]:
    match = re.search(
        rf"(?m)^{re.escape(key)}:\s*(.+?)\s*$",
        frontmatter,
    )
    if not match:
        return None
    value = match.group(1).strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        value = value[1:-1].strip()
    return value if value else None


def load_planner_skill(path: Path) -> PlannerSkill:
    text = path.read_text(encoding="utf-8").strip()
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(f"missing YAML frontmatter in planner skill: {path}")
    frontmatter = match.group(1)
    name = _frontmatter_value(frontmatter, "name")
    try:
        when_to_use = _frontmatter_value(frontmatter, "when_to_use")
    except ValueError:
        # Backward compatibility for skill-creator style frontmatter.
        when_to_use = _frontmatter_value(frontmatter, "description")
    content = text[match.end() :].strip()
    if not re.fullmatch(r"[a-z0-9-]{1,63}", name):
        raise ValueError(f"invalid planner skill name: {name!r}")
    if not content:
        raise ValueError(f"empty planner skill content: {path}")

    # Parse or infer required elements for fast L0 deterministic filtering
    req_elements_raw = _optional_frontmatter_value(frontmatter, "required_elements")
    if req_elements_raw:
        required_elements = tuple(
            el.strip().strip("'\"")
            for el in req_elements_raw.replace("[", "").replace("]", "").split(",")
            if el.strip().strip("'\"")
        )
    else:
        inferred = []
        name_lower = name.lower()
        if any(k in name_lower for k in ("appel", "staudinger", "wittig", "horner")):
            inferred.append("P")
        if any(k in name_lower for k in ("isothiocyanate", "hantzsch-thiazole", "sulfide", "tosylate", "swern")):
            inferred.append("S")
        if any(k in name_lower for k in (
            "amine", "imine", "nitrile", "boc", "cbz", "fmoc", "ing-manske", "dcc-amide",
            "mannich", "sn-ar-amine", "vilsmeier", "weinreb", "lah-amide", "imidazole",
            "knorr-pyrazole", "paal-knorr", "primary-amide", "fischer-indole",
            "bischler-napieralski", "skraup", "shapiro", "curtius-beckmann",
            "reductive-amination", "isocyanate"
        )):
            inferred.append("N")
        required_elements = tuple(inferred)

    return PlannerSkill(
        name=name,
        when_to_use=when_to_use,
        content=content,
        source_path=str(path),
        required_elements=required_elements,
    )


class PlannerSkillLibrary:
    """Two-level Planner skill store: compact catalog, then selected content."""

    def __init__(self, skills: Iterable[PlannerSkill]):
        ordered = sorted(skills, key=lambda item: item.name)
        self._skills = {item.name: item for item in ordered}
        if len(self._skills) != len(ordered):
            raise ValueError("duplicate planner skill name")
        if not self._skills:
            raise ValueError("planner skill library is empty")

    @classmethod
    def from_directory(cls, root: str | Path) -> "PlannerSkillLibrary":
        root_path = Path(root)
        paths = sorted(root_path.glob("*/SKILL.md"))
        if not paths:
            raise ValueError(f"no planner skills found under: {root_path}")
        return cls(load_planner_skill(path) for path in paths)

    @property
    def names(self) -> List[str]:
        return list(self._skills)

    def catalog(self) -> List[Dict[str, str]]:
        return [item.catalog_entry() for item in self._skills.values()]

    def filter_candidates_l0(
        self,
        initial_pgl: str,
        target_pgl: str,
    ) -> List[Dict[str, str]]:
        """L0 deterministic pre-filter: prune skills whose required elements are absent.

        Only the element test is applied. Every surviving skill still goes to
        the batched high-recall LLM scan; ranking the survivors by keywords and
        keeping a fixed number used to drop the expected skill in 49 of 60
        MechFinder cases, because nearly every reaction contains N or O and so
        most skills survive the element test.
        """
        elements = {m[1] for m in _ATOM_RE.findall(initial_pgl)} | {
            m[1] for m in _ATOM_RE.findall(target_pgl)
        }
        return [
            skill.catalog_entry()
            for skill in self._skills.values()
            if not skill.required_elements
            or any(element in elements for element in skill.required_elements)
        ]

    def selected_content(self, names: Iterable[str]) -> List[Dict[str, str]]:
        selected: List[Dict[str, str]] = []
        for name in names:
            if name not in self._skills:
                raise ValueError(f"unknown planner skill selected: {name}")
            selected.append(self._skills[name].injection_entry())
        return selected


def _pgl_graph(pgl: str) -> Dict[str, Any]:
    atoms: Dict[str, Dict[str, Any]] = {}
    for label, element, options in _ATOM_RE.findall(pgl):
        h_match = re.search(r"\bHcount=(-?\d+)", options)
        charge_match = re.search(r"\bcharge=(-?\d+)", options)
        atoms[label] = {
            "element": element,
            "Hcount": int(h_match.group(1)) if h_match else 0,
            "charge": int(charge_match.group(1)) if charge_match else 0,
        }
    bonds: Dict[tuple[str, str], Dict[str, Any]] = {}
    for left, right, kind, raw_order in _BOND_RE.findall(pgl):
        key = tuple(sorted((left, right)))
        bonds[key] = {
            "atoms": list(key),
            "kind": kind,
            "order": float(raw_order) if raw_order else None,
        }
    return {"atoms": atoms, "bonds": bonds}


def _connected_components(
    atom_labels: Iterable[str],
    bonds: Dict[tuple[str, str], Dict[str, Any]],
) -> List[List[str]]:
    adjacency = {label: set() for label in atom_labels}
    for left, right in bonds:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    components: List[List[str]] = []
    unseen = set(adjacency)
    while unseen:
        root = min(unseen)
        stack = [root]
        component = set()
        while stack:
            label = stack.pop()
            if label in component:
                continue
            component.add(label)
            stack.extend(adjacency.get(label, ()))
        unseen -= component
        components.append(sorted(component))
    return sorted(components, key=lambda item: (item[0], len(item)))


def _element_counts(labels: Iterable[str], atoms: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label in labels:
        element = str(atoms[label]["element"])
        counts[element] = counts.get(element, 0) + 1
    return dict(sorted(counts.items()))


def pgl_structure_change_facts(initial_pgl: str, target_pgl: str) -> Dict[str, Any]:
    """Mechanically compare supplied PGLs; no GT path or chemistry inference is used."""

    initial = _pgl_graph(initial_pgl)
    target = _pgl_graph(target_pgl)
    initial_atoms = initial["atoms"]
    target_atoms = target["atoms"]
    initial_bonds = initial["bonds"]
    target_bonds = target["bonds"]
    shared_atoms = set(initial_atoms) & set(target_atoms)
    added_atoms = set(target_atoms) - set(initial_atoms)
    removed_atoms = set(initial_atoms) - set(target_atoms)
    shared_initial_bonds = {
        key: value
        for key, value in initial_bonds.items()
        if set(key) <= shared_atoms
    }
    shared_target_bonds = {
        key: value
        for key, value in target_bonds.items()
        if set(key) <= shared_atoms
    }
    added_keys = sorted(set(shared_target_bonds) - set(shared_initial_bonds))
    removed_keys = sorted(set(shared_initial_bonds) - set(shared_target_bonds))
    changed_keys = sorted(
        key
        for key in set(shared_initial_bonds) & set(shared_target_bonds)
        if shared_initial_bonds[key] != shared_target_bonds[key]
    )
    atom_changes = []
    for label in sorted(set(initial_atoms) & set(target_atoms)):
        if initial_atoms[label] != target_atoms[label]:
            atom_changes.append(
                {
                    "atom": label,
                    "initial": initial_atoms[label],
                    "target": target_atoms[label],
                }
            )
    fully_removed_components = [
        component
        for component in _connected_components(initial_atoms, initial_bonds)
        if set(component) <= removed_atoms
    ]
    fully_added_components = [
        component
        for component in _connected_components(target_atoms, target_bonds)
        if set(component) <= added_atoms
    ]
    return {
        "shared_atom_changes": {
            "added_bonds": [shared_target_bonds[key] for key in added_keys],
            "removed_bonds": [shared_initial_bonds[key] for key in removed_keys],
            "changed_bonds": [
                {
                    "atoms": list(key),
                    "initial": shared_initial_bonds[key],
                    "target": shared_target_bonds[key],
                }
                for key in changed_keys
            ],
            "atom_property_changes": atom_changes,
        },
        "annotation_boundary_changes": {
            "added_atom_count": len(added_atoms),
            "removed_atom_count": len(removed_atoms),
            "added_element_counts": _element_counts(added_atoms, target_atoms),
            "removed_element_counts": _element_counts(removed_atoms, initial_atoms),
            "fully_added_components": [
                {
                    "atom_count": len(component),
                    "element_counts": _element_counts(component, target_atoms),
                }
                for component in fully_added_components
            ],
            "fully_removed_components": [
                {
                    "atom_count": len(component),
                    "element_counts": _element_counts(component, initial_atoms),
                }
                for component in fully_removed_components
            ],
            "interpretation": (
                "These differences may reflect untracked species, leaving fragments, "
                "or transient explicit-H labels. They are not reaction-center evidence "
                "by themselves and must not create cleanup steps."
            ),
        },
    }


def normalize_skill_selection(
    data: Dict[str, Any],
    *,
    allowed_names: Iterable[str],
    max_skills: int,
) -> Dict[str, Any]:
    allowed = set(allowed_names)
    raw_selected = data.get("selected_skills") or []
    if not isinstance(raw_selected, list):
        raise ValueError("selected_skills must be a list")
    selected: List[Dict[str, str]] = []
    seen = set()
    for item in raw_selected:
        if not isinstance(item, dict):
            raise ValueError("each selected skill must be an object")
        name = str(item.get("name") or "").strip()
        reason = str(item.get("reason") or "").strip()
        if name not in allowed:
            raise ValueError(f"selector returned unknown planner skill: {name!r}")
        if not reason:
            raise ValueError(f"selector omitted reason for planner skill: {name}")
        if name not in seen:
            selected.append({"name": name, "reason": reason})
            seen.add(name)
    if len(selected) > max_skills:
        raise ValueError(f"selector returned more than {max_skills} planner skills")
    evidence = data.get("structural_evidence") or []
    if not isinstance(evidence, list):
        raise ValueError("structural_evidence must be a list")
    return {
        "inferred_reaction_pattern": str(
            data.get("inferred_reaction_pattern") or "unresolved"
        ).strip(),
        "structural_evidence": [str(item) for item in evidence],
        "selected_skills": selected,
        "no_match_reason": str(data.get("no_match_reason") or "").strip(),
    }


def normalize_reaction_features(data: Dict[str, Any]) -> Dict[str, Any]:
    fields = (
        "initial_functional_roles",
        "core_net_changes",
        "candidate_families",
        "distinguishing_evidence",
        "ambiguities",
    )
    normalized: Dict[str, Any] = {
        "reaction_center_summary": str(
            data.get("reaction_center_summary") or "unresolved"
        ).strip()
    }
    for field in fields:
        value = data.get(field) or []
        if not isinstance(value, list):
            raise ValueError(f"reaction feature {field} must be a list")
        normalized[field] = [str(item) for item in value]
    return normalized


def normalize_skill_verification(
    data: Dict[str, Any],
    *,
    candidate_names: Iterable[str],
) -> List[Dict[str, Any]]:
    candidates = list(candidate_names)
    allowed = set(candidates)
    raw = data.get("skill_checks") or []
    if not isinstance(raw, list):
        raise ValueError("skill_checks must be a list")
    checks: List[Dict[str, Any]] = []
    seen = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each skill check must be an object")
        name = str(item.get("name") or "").strip()
        if name not in allowed or name in seen:
            raise ValueError(f"invalid or duplicate verified planner skill: {name!r}")
        satisfied = item.get("trigger_satisfied")
        if not isinstance(satisfied, bool):
            raise ValueError(f"trigger_satisfied must be boolean for: {name}")
        supporting = item.get("supporting_evidence") or []
        contradicting = item.get("contradicting_evidence") or []
        raw_bond_checks = item.get("bond_checks") or []
        raw_path_checks = item.get("path_checks") or []
        if not isinstance(supporting, list) or not isinstance(contradicting, list):
            raise ValueError(f"skill evidence fields must be lists for: {name}")
        if not isinstance(raw_bond_checks, list):
            raise ValueError(f"bond_checks must be a list for: {name}")
        if not isinstance(raw_path_checks, list):
            raise ValueError(f"path_checks must be a list for: {name}")
        bond_checks: List[Dict[str, Any]] = []
        for claim in raw_bond_checks:
            if not isinstance(claim, dict):
                raise ValueError(f"each bond check must be an object for: {name}")
            state = str(claim.get("state") or "").strip()
            atoms = claim.get("atoms") or []
            exists = claim.get("exists")
            kind = claim.get("kind")
            if state not in {"initial", "target"}:
                raise ValueError(f"invalid bond-check state for: {name}")
            if not isinstance(atoms, list) or len(atoms) != 2:
                raise ValueError(f"bond-check atoms must have length two for: {name}")
            if not isinstance(exists, bool):
                raise ValueError(f"bond-check exists must be boolean for: {name}")
            bond_checks.append(
                {
                    "state": state,
                    "atoms": [str(atoms[0]), str(atoms[1])],
                    "exists": exists,
                    "kind": str(kind) if kind is not None else None,
                    "purpose": str(claim.get("purpose") or "").strip(),
                }
            )
        path_checks: List[Dict[str, Any]] = []
        for claim in raw_path_checks:
            if not isinstance(claim, dict):
                raise ValueError(f"each path check must be an object for: {name}")
            state = str(claim.get("state") or "").strip()
            atoms = claim.get("atoms") or []
            if state not in {"initial", "target"}:
                raise ValueError(f"invalid path-check state for: {name}")
            if not isinstance(atoms, list) or len(atoms) < 4:
                raise ValueError(f"path-check atoms must contain a closed path for: {name}")
            path_checks.append(
                {
                    "state": state,
                    "atoms": [str(value) for value in atoms],
                    "purpose": str(claim.get("purpose") or "").strip(),
                }
            )
        if contradicting:
            satisfied = False
        reason = str(item.get("reason") or "").strip()
        if not reason:
            raise ValueError(f"skill verification omitted reason for: {name}")
        checks.append(
            {
                "name": name,
                "trigger_satisfied": satisfied,
                "supporting_evidence": [str(value) for value in supporting],
                "contradicting_evidence": [str(value) for value in contradicting],
                "bond_checks": bond_checks,
                "path_checks": path_checks,
                "reason": reason,
            }
        )
        seen.add(name)
    if seen != allowed:
        raise ValueError("skill verification must check every candidate exactly once")
    return checks


def audit_skill_verification_evidence(
    checks: List[Dict[str, Any]],
    *,
    initial_pgl: str,
    target_pgl: str,
) -> List[Dict[str, Any]]:
    """Validate only the bond facts a verifier chose to cite."""

    graphs = {"initial": _pgl_graph(initial_pgl), "target": _pgl_graph(target_pgl)}
    audited: List[Dict[str, Any]] = []
    for original in checks:
        item = dict(original)
        contradictions = list(item.get("contradicting_evidence") or [])
        for claim in item.get("bond_checks") or []:
            graph = graphs[claim["state"]]
            key = tuple(sorted(claim["atoms"]))
            actual = graph["bonds"].get(key)
            actual_exists = actual is not None
            kind_matches = (
                claim.get("kind") is None
                or (actual is not None and actual["kind"] == claim["kind"])
            )
            if actual_exists != claim["exists"] or (
                claim["exists"] and not kind_matches
            ):
                contradictions.append(
                    "Invalid cited bond fact: "
                    f"state={claim['state']} atoms={claim['atoms']} "
                    f"claimed_exists={claim['exists']} claimed_kind={claim.get('kind')} "
                    f"actual={actual}"
                )
        for claim in item.get("path_checks") or []:
            graph = graphs[claim["state"]]
            path = claim["atoms"]
            invalid_edges = [
                [left, right]
                for left, right in zip(path, path[1:])
                if tuple(sorted((left, right))) not in graph["bonds"]
            ]
            if path[0] != path[-1] or invalid_edges:
                contradictions.append(
                    "Invalid cited closed path: "
                    f"state={claim['state']} path={path} invalid_edges={invalid_edges}"
                )
        item["contradicting_evidence"] = contradictions
        if contradictions:
            item["trigger_satisfied"] = False
        audited.append(item)
    return audited


def normalize_skill_resolution(
    data: Dict[str, Any],
    *,
    accepted_names: Iterable[str],
    max_skills: int,
) -> Dict[str, Any]:
    allowed = set(accepted_names)
    raw = data.get("selected_skill_names") or []
    if not isinstance(raw, list):
        raise ValueError("selected_skill_names must be a list")
    names: List[str] = []
    for value in raw:
        name = str(value).strip()
        if name not in allowed:
            raise ValueError(f"resolution returned unaccepted skill: {name!r}")
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("resolution must retain at least one accepted skill")
    if len(names) > max_skills:
        raise ValueError("resolution retained too many skills")
    return {
        "selected_skill_names": names,
        "reason": str(data.get("reason") or "").strip(),
    }
