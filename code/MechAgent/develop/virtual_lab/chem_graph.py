"""Molecular graph algorithms: connected components, exact HRMS masses, WL + backtracking automorphisms, and SMILES.

100% pure Python stdlib; transparently uses RDKit for canonical SMILES when available.
"""
from __future__ import annotations

import collections
import copy
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .. import standalone_chemistry as sc
from .reference_data import (
    ELECTRON_MASS,
    HRMS_ION_OFFSETS,
    ISOTOPE_EXACT_MASS,
    MONOISOTOPIC_MASS,
    PSEUDOATOM_NOMINAL_MASS,
)
from .schemas import AtomMeta, LabState

try:
    from rdkit import Chem as _rdkit_chem  # type: ignore

    HAS_RDKIT = True
except Exception:
    _rdkit_chem = None
    HAS_RDKIT = False


def canonical_isotope_tag(element: str, isotope: Optional[str]) -> Optional[str]:
    """Normalize isotope labels such as 'D' -> '2H', '13' on C -> '13C', etc."""
    if not isotope:
        if element == "D":
            return "2H"
        if element == "T":
            return "3H"
        return None
    raw = str(isotope).strip()
    if not raw:
        return None
    if raw == "D":
        return "2H"
    if raw == "T":
        return "3H"
    if raw.isdigit():
        base_el = "H" if element in {"D", "T"} else element
        return f"{raw}{base_el}"
    return raw


def base_element(element: str) -> str:
    if element in {"D", "T"}:
        return "H"
    return element


def atom_exact_mass(element: str, isotope: Optional[str] = None) -> float:
    tag = canonical_isotope_tag(element, isotope)
    if tag and tag in ISOTOPE_EXACT_MASS:
        return ISOTOPE_EXACT_MASS[tag]
    el = base_element(element)
    if el in MONOISOTOPIC_MASS:
        return MONOISOTOPIC_MASS[el]
    if el in PSEUDOATOM_NOMINAL_MASS:
        return PSEUDOATOM_NOMINAL_MASS[el]
    if el.startswith("R") and el[1:].isdigit():
        return PSEUDOATOM_NOMINAL_MASS["R"]
    raise KeyError(f"Unknown element for monoisotopic mass: {element!r}")


@dataclass
class MolecularEntity:
    """A single connected molecular component extracted from a LabState."""

    atom_ids: Tuple[str, ...]
    atoms: Dict[str, sc.Atom]
    meta: Dict[str, AtomMeta]
    bonds: Dict[Tuple[str, str], sc.Bond]

    @property
    def net_charge(self) -> int:
        return sum(int(a.charge or 0) for a in self.atoms.values())

    @property
    def total_radicals(self) -> int:
        return sum(int(self.meta[aid].radicals) for aid in self.atom_ids if aid in self.meta)

    @property
    def radical_sites(self) -> List[str]:
        return [aid for aid in self.atom_ids if self.meta.get(aid, AtomMeta()).radicals > 0]

    def element_counts(self, *, separate_isotopes: bool = False) -> Dict[str, int]:
        counts: Dict[str, int] = collections.Counter()
        for aid in self.atom_ids:
            atom = self.atoms[aid]
            meta = self.meta.get(aid, AtomMeta())
            el = base_element(atom.element)
            iso = canonical_isotope_tag(atom.element, meta.isotope)
            key = f"[{iso}]" if (separate_isotopes and iso) else el
            counts[key] += 1
            implicit_h = int(atom.Hcount or 0)
            if implicit_h > 0:
                counts["H"] += implicit_h
        return dict(counts)

    def hill_formula(self, *, include_isotopes: bool = False) -> str:
        counts = self.element_counts(separate_isotopes=include_isotopes)
        parts: List[str] = []
        if "C" in counts:
            c = counts.pop("C")
            parts.append("C" if c == 1 else f"C{c}")
            if "H" in counts:
                h = counts.pop("H")
                parts.append("H" if h == 1 else f"H{h}")
        for sym in sorted(counts):
            n = counts[sym]
            parts.append(sym if n == 1 else f"{sym}{n}")
        charge = self.net_charge
        suffix = ""
        if charge == 1:
            suffix = "+"
        elif charge == -1:
            suffix = "-"
        elif charge > 1:
            suffix = f"+{charge}"
        elif charge < -1:
            suffix = str(charge)
        return "".join(parts) + suffix

    @property
    def exact_mass(self) -> float:
        total = 0.0
        for aid in self.atom_ids:
            atom = self.atoms[aid]
            meta = self.meta.get(aid, AtomMeta())
            total += atom_exact_mass(atom.element, meta.isotope)
            implicit_h = int(atom.Hcount or 0)
            if implicit_h > 0:
                total += implicit_h * MONOISOTOPIC_MASS["H"]
        # Subtract/add rest mass of net formal charge electrons
        total -= self.net_charge * ELECTRON_MASS
        return round(total, 6)

    def hrms_mz(self) -> Dict[str, float]:
        """Compute exact HRMS m/z values for standard ionization adducts."""
        q = self.net_charge
        if q != 0:
            return {"[M] (native ion)": round(abs(self.exact_mass / q), 5)}
        m = self.exact_mass
        return {
            ion: round((m + offset) / z, 5)
            for ion, (offset, z) in HRMS_ION_OFFSETS.items()
        }

    def to_smiles(self, *, include_isotopes: bool = True) -> str:
        if HAS_RDKIT:
            try:
                return _rdkit_smiles(self, include_isotopes=include_isotopes)
            except Exception:
                pass
        return _stdlib_canonical_smiles(self, include_isotopes=include_isotopes)


def _bond_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def extract_graph(lab_state: LabState) -> Tuple[Dict[str, sc.Atom], Dict[Tuple[str, str], sc.Bond], Dict[str, Set[str]]]:
    atoms = {name: atom for name, atom in sc.pgl.iter_atoms(lab_state.pgl_state)}
    bonds = dict(sc.pgl.bond_map(lab_state.pgl_state))
    adj: Dict[str, Set[str]] = {name: set() for name in atoms}
    for a, b in bonds:
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    return atoms, bonds, adj


def connected_molecules(lab_state: LabState) -> List[MolecularEntity]:
    """Partition a LabState into true connected molecular components."""
    atoms, bonds, adj = extract_graph(lab_state)
    visited: Set[str] = set()
    entities: List[MolecularEntity] = []

    for root in sorted(atoms):
        if root in visited:
            continue
        comp: List[str] = []
        queue = collections.deque([root])
        visited.add(root)
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nb in sorted(adj[cur]):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        comp_tuple = tuple(sorted(comp))
        comp_set = set(comp_tuple)
        comp_atoms = {aid: copy.deepcopy(atoms[aid]) for aid in comp_tuple}
        comp_meta = {aid: lab_state.atom_meta.get(aid, AtomMeta()).clone() for aid in comp_tuple}
        comp_bonds = {
            k: copy.deepcopy(b)
            for k, b in bonds.items()
            if k[0] in comp_set and k[1] in comp_set
        }
        entities.append(
            MolecularEntity(
                atom_ids=comp_tuple,
                atoms=comp_atoms,
                meta=comp_meta,
                bonds=comp_bonds,
            )
        )
    entities.sort(key=lambda e: (-len(e.atom_ids), e.atom_ids))
    return entities


def normalize_pgl_species(pgl_state: sc.ReactionState) -> sc.ReactionState:
    """Repartition ReactionState.species so each Species equals one connected molecular component."""
    dummy = LabState.from_pgl(pgl_state)
    entities = connected_molecules(dummy)
    old_roles: Dict[str, Tuple[str, str]] = {}
    for sp in pgl_state.species:
        for sub in sp.molecule.substructures:
            for aid in sub.atomdict:
                old_roles[aid] = (sp.role or "intermediate", sp.state or "")
    new_species: List[sc.Species] = []
    for idx, ent in enumerate(entities):
        role, st = old_roles.get(ent.atom_ids[0], ("intermediate", ""))
        sub = sc.Substructure(
            name=f"m{idx + 1}",
            atomdict={aid: copy.deepcopy(ent.atoms[aid]) for aid in ent.atom_ids},
            bondlist=[copy.deepcopy(b) for _, b in sorted(ent.bonds.items())],
        )
        new_species.append(
            sc.Species(
                role=role,
                state=st,
                molecule=sc.Molecule(name="", substructures=[sub]),
            )
        )
    return sc.ReactionState(species=new_species)


def materialize_implicit_hydrogens(
    lab_state: LabState,
    *,
    target_atoms: Optional[Iterable[str]] = None,
    prefix: str = "H_imp_",
) -> Tuple[LabState, Dict[str, List[str]]]:
    """Promote implicit Atom.Hcount on selected (or all) atoms into explicit Atom('H') nodes."""
    updated = lab_state.clone()
    existing_ids = {name for name, _ in sc.pgl.iter_atoms(updated.pgl_state)}
    target_set = set(target_atoms) if target_atoms is not None else None
    counter = 1
    created_by_owner: Dict[str, List[str]] = {}

    for sp in updated.pgl_state.species:
        for sub in sp.molecule.substructures:
            owners = list(sub.atomdict.keys())
            for owner in owners:
                if target_set is not None and owner not in target_set:
                    continue
                atom = sub.atomdict[owner]
                h_n = int(atom.Hcount or 0)
                if h_n <= 0:
                    continue
                atom.Hcount = 0
                owner_meta = updated.atom_meta.get(owner, AtomMeta())
                created: List[str] = []
                for _ in range(h_n):
                    while f"{prefix}{counter}" in existing_ids:
                        counter += 1
                    hid = f"{prefix}{counter}"
                    existing_ids.add(hid)
                    counter += 1
                    sub.atomdict[hid] = sc.Atom("H", Hcount=0, charge=0)
                    sub.bondlist.append(sc.Bond(owner, hid, "single", order=1.0))
                    updated.atom_meta[hid] = AtomMeta(
                        fragment_origin=owner_meta.fragment_origin,
                        role_tag=owner_meta.role_tag,
                    )
                    created.append(hid)
                created_by_owner[owner] = created
    return updated, created_by_owner


# ---------------------------------------------------------------------------
# SMILES Generation (RDKit when available + Pure-Stdlib Canonical Fallback)
# ---------------------------------------------------------------------------

def _fold_simple_hydrogens(
    entity: MolecularEntity,
    *,
    include_isotopes: bool = True,
) -> Tuple[List[str], Dict[str, int]]:
    """Fold terminal uncharged non-radical unlabeled 1H atoms into their heavy-atom parent's H count."""
    adj: Dict[str, Dict[str, float]] = {aid: {} for aid in entity.atom_ids}
    for (a, b), bond in entity.bonds.items():
        order = float(bond.order or sc._kind_order(bond.kind) or 1.0)
        adj[a][b] = order
        adj[b][a] = order

    folded_h: Set[str] = set()
    extra_h: Dict[str, int] = collections.Counter()
    if len(entity.atom_ids) > 1:
        for aid in entity.atom_ids:
            at = entity.atoms[aid]
            mt = entity.meta.get(aid, AtomMeta())
            iso = canonical_isotope_tag(at.element, mt.isotope)
            if (
                base_element(at.element) == "H"
                and int(at.charge or 0) == 0
                and mt.radicals == 0
                and (not include_isotopes or not iso)
                and len(adj[aid]) == 1
            ):
                nb, order = next(iter(adj[aid].items()))
                if base_element(entity.atoms[nb].element) != "H" and abs(order - 1.0) < 1e-9:
                    folded_h.add(aid)
                    extra_h[nb] += 1

    kept = [aid for aid in entity.atom_ids if aid not in folded_h]
    if not kept:
        return list(entity.atom_ids), {}
    return kept, dict(extra_h)


def _rdkit_smiles(entity: MolecularEntity, *, include_isotopes: bool = True) -> str:
    kept, extra_h = _fold_simple_hydrogens(entity, include_isotopes=include_isotopes)
    kept_set = set(kept)
    rw = _rdkit_chem.RWMol()
    idx_map: Dict[str, int] = {}
    bond_types = {
        1.0: _rdkit_chem.BondType.SINGLE,
        1.5: _rdkit_chem.BondType.AROMATIC,
        2.0: _rdkit_chem.BondType.DOUBLE,
        3.0: _rdkit_chem.BondType.TRIPLE,
    }
    for aid in kept:
        pgl_atom = entity.atoms[aid]
        meta = entity.meta.get(aid, AtomMeta())
        el = base_element(pgl_atom.element)
        if el in MONOISOTOPIC_MASS:
            ra = _rdkit_chem.Atom(el)
        else:
            ra = _rdkit_chem.Atom(0)  # Wildcard '*' for generic template pseudoatoms (R, Ar, X)
        ra.SetNoImplicit(True)
        ra.SetNumExplicitHs(int(pgl_atom.Hcount or 0) + int(extra_h.get(aid, 0)))
        ra.SetFormalCharge(int(pgl_atom.charge or 0))
        if meta.radicals > 0:
            ra.SetNumRadicalElectrons(int(meta.radicals))
        iso = canonical_isotope_tag(pgl_atom.element, meta.isotope)
        if include_isotopes and iso:
            m = re.match(r"^(\d+)", iso)
            if m:
                ra.SetIsotope(int(m.group(1)))
        idx_map[aid] = rw.AddAtom(ra)

    for (a, b), bond in sorted(entity.bonds.items()):
        if a not in kept_set or b not in kept_set:
            continue
        order = float(bond.order or sc._kind_order(bond.kind) or 1.0)
        btype = bond_types.get(order, _rdkit_chem.BondType.SINGLE)
        rw.AddBond(idx_map[a], idx_map[b], btype)

    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return _rdkit_chem.MolToSmiles(mol, canonical=True, isomericSmiles=include_isotopes)


def _wl_colors(
    atom_ids: Sequence[str],
    atoms: Dict[str, sc.Atom],
    meta: Dict[str, AtomMeta],
    bonds: Dict[Tuple[str, str], sc.Bond],
    *,
    include_isotopes: bool = False,
    rounds: int = 6,
) -> Dict[str, int]:
    """1D Weisfeiler-Lehman vertex color refinement."""
    adj: Dict[str, List[Tuple[float, str]]] = {aid: [] for aid in atom_ids}
    for (a, b), bond in bonds.items():
        if a in adj and b in adj:
            order = round(float(bond.order or sc._kind_order(bond.kind) or 1.0), 2)
            adj[a].append((order, b))
            adj[b].append((order, a))

    # Collapse explicit unlabelled H count into heavy-atom invariant for electronic symmetry
    init_sig: Dict[str, Any] = {}
    for aid in atom_ids:
        at = atoms[aid]
        mt = meta.get(aid, AtomMeta())
        iso = canonical_isotope_tag(at.element, mt.isotope) if include_isotopes else None
        edge_orders = tuple(sorted(o for o, _ in adj[aid]))
        init_sig[aid] = (
            base_element(at.element),
            int(at.charge or 0),
            int(at.Hcount or 0),
            int(mt.radicals),
            bool(mt.excited),
            iso,
            edge_orders,
        )
    palette = {sig: idx for idx, sig in enumerate(sorted(set(init_sig.values())))}
    colors = {aid: palette[init_sig[aid]] for aid in atom_ids}

    for _ in range(rounds):
        next_sig: Dict[str, Any] = {}
        for aid in atom_ids:
            nb_colors = tuple(sorted((o, colors[nb]) for o, nb in adj[aid]))
            next_sig[aid] = (colors[aid], nb_colors)
        palette = {sig: idx for idx, sig in enumerate(sorted(set(next_sig.values())))}
        new_colors = {aid: palette[next_sig[aid]] for aid in atom_ids}
        if len(set(new_colors.values())) == len(set(colors.values())):
            colors = new_colors
            break
        colors = new_colors
    return colors


def _stdlib_canonical_smiles(entity: MolecularEntity, *, include_isotopes: bool = True) -> str:
    """Deterministic pure-stdlib canonical topological SMILES generator."""
    colors = _wl_colors(
        entity.atom_ids,
        entity.atoms,
        entity.meta,
        entity.bonds,
        include_isotopes=include_isotopes,
    )
    # Collapse simple terminal uncharged unlabelled explicit H atoms onto heavy parent for clean SMILES
    adj: Dict[str, Dict[str, float]] = {aid: {} for aid in entity.atom_ids}
    for (a, b), bond in entity.bonds.items():
        order = float(bond.order or sc._kind_order(bond.kind) or 1.0)
        adj[a][b] = order
        adj[b][a] = order

    kept, extra_h = _fold_simple_hydrogens(entity, include_isotopes=include_isotopes)
    kept_set = set(kept)
    sub_adj = {
        u: {v: o for v, o in adj[u].items() if v in kept_set}
        for u in kept
    }

    organic_valences: Dict[str, Tuple[int, ...]] = {
        "B": (3,),
        "C": (4,),
        "N": (3, 5),
        "O": (2,),
        "P": (3, 5),
        "S": (2, 4, 6),
        "F": (1,),
        "Cl": (1,),
        "Br": (1,),
        "I": (1,),
    }

    def atom_token(aid: str) -> str:
        at = entity.atoms[aid]
        mt = entity.meta.get(aid, AtomMeta())
        el = base_element(at.element)
        iso = canonical_isotope_tag(at.element, mt.isotope) if include_isotopes else None
        iso_num = ""
        if iso:
            m = re.match(r"^(\d+)", iso)
            if m:
                iso_num = m.group(1)
        h_tot = int(at.Hcount or 0) + int(extra_h.get(aid, 0))
        q = int(at.charge or 0)
        h_str = "" if h_tot == 0 else ("H" if h_tot == 1 else f"H{h_tot}")
        q_str = ""
        if q == 1:
            q_str = "+"
        elif q == -1:
            q_str = "-"
        elif q > 1:
            q_str = f"+{q}"
        elif q < -1:
            q_str = str(q)
        bond_sum = round(sum(sub_adj[aid].values()), 2)
        tot_val = bond_sum + h_tot
        std_vals = organic_valences.get(el)
        non_std_val = std_vals is None or not any(abs(tot_val - sv) < 1e-6 for sv in std_vals)
        # Always bracket if isotope, charge, radical, non-organic, or non-standard valence
        if iso_num or q != 0 or mt.radicals > 0 or non_std_val:
            return f"[{iso_num}{el}{h_str}{q_str}]"
        return el

    def bond_symbol(order: float) -> str:
        if abs(order - 2.0) < 1e-9:
            return "="
        if abs(order - 3.0) < 1e-9:
            return "#"
        if abs(order - 1.5) < 1e-9:
            return ":"
        return ""

    root = min(kept, key=lambda u: (colors[u], u))
    visited: Set[str] = set()
    parent: Dict[str, Optional[str]] = {root: None}
    tree_children: Dict[str, List[str]] = {u: [] for u in kept}
    ring_closures: Dict[Tuple[str, str], int] = {}
    ring_id = 0

    def dfs_tree(u: str) -> None:
        nonlocal ring_id
        visited.add(u)
        nbs = sorted(sub_adj[u], key=lambda v: (colors[v], -sub_adj[u][v], v))
        for v in nbs:
            if v == parent.get(u):
                continue
            if v not in visited:
                parent[v] = u
                tree_children[u].append(v)
                dfs_tree(v)
            else:
                edge = _bond_key(u, v)
                if edge not in ring_closures:
                    ring_id += 1
                    ring_closures[edge] = ring_id

    dfs_tree(root)

    ring_tags_at: Dict[str, List[Tuple[int, str]]] = {u: [] for u in kept}
    opened: Set[int] = set()

    def emit(u: str) -> str:
        ring_part = ""
        # Check back-edges incident to u
        nbs = sorted(sub_adj[u], key=lambda v: (colors[v], v))
        for v in nbs:
            edge = _bond_key(u, v)
            if edge in ring_closures:
                rnum = ring_closures[edge]
                r_str = str(rnum) if rnum < 10 else f"%{rnum}"
                bsym = bond_symbol(sub_adj[u][v]) if rnum not in opened else ""
                opened.add(rnum)
                ring_part += f"{bsym}{r_str}"
        tok = atom_token(u) + ring_part
        ch = tree_children[u]
        if not ch:
            return tok
        rendered_branches = [
            f"{bond_symbol(sub_adj[u][v])}{emit(v)}"
            for v in ch
        ]
        for branch in rendered_branches[:-1]:
            tok += f"({branch})"
        tok += rendered_branches[-1]
        return tok

    return emit(root)


# ---------------------------------------------------------------------------
# Exact Electronic Graph Automorphism Solver (Symmetry Equivalence Engine)
# ---------------------------------------------------------------------------

def electronic_automorphisms(
    lab_state: LabState,
    *,
    subset_atoms: Optional[Iterable[str]] = None,
    max_automorphisms: int = 128,
) -> List[Dict[str, str]]:
    """Compute exact graph automorphisms of the electronic molecular graph.

    Crucially, isotopic tags (`13C`, `18O`, `2H`) and fragment provenance (`A`/`B`)
    are EXCLUDED from the electronic invariant because chemically symmetric sites
    (e.g. C14 and C15 in a symmetric spiro/phenonium intermediate, or two carboxylate
    oxygens) react with identical probabilities regardless of hidden isotopic labels.

    To avoid combinatorial explosion on spectator methyl/tert-butyl groups (`3!^k`),
    identical leaf atoms attached to the same heavy parent are factored out during
    core skeleton backtracking and expanded only if non-trivially queried.
    """
    atoms, bonds, _ = extract_graph(lab_state)
    allowed = set(subset_atoms) if subset_atoms is not None else set(atoms.keys())
    nodes = sorted(u for u in atoms if u in allowed)
    if len(nodes) <= 1:
        return [{u: u for u in nodes}]

    adj: Dict[str, Dict[str, float]] = {u: {} for u in nodes}
    for (a, b), bond in bonds.items():
        if a in adj and b in adj:
            order = round(float(bond.order or sc._kind_order(bond.kind) or 1.0), 2)
            adj[a][b] = order
            adj[b][a] = order

    # Separate leaf hydrogens/halogens attached to a single heavy parent to prevent 3!^k leaf blowup
    leaf_Signature: Dict[str, Tuple[Any, ...]] = {}
    for u in nodes:
        at = atoms[u]
        mt = lab_state.atom_meta.get(u, AtomMeta())
        leaf_Signature[u] = (
            base_element(at.element),
            int(at.charge or 0),
            int(at.Hcount or 0),
            int(mt.radicals),
            bool(mt.excited),
        )

    leaf_groups: Dict[Tuple[str, Tuple[Any, ...], float], List[str]] = collections.defaultdict(list)
    leaf_nodes: Set[str] = set()
    for u in nodes:
        if len(adj[u]) == 1 and base_element(atoms[u].element) == "H":
            parent_id, b_ord = next(iter(adj[u].items()))
            if len(adj[parent_id]) > 1:
                leaf_groups[(parent_id, leaf_Signature[u], b_ord)].append(u)
                leaf_nodes.add(u)

    core_nodes = [u for u in nodes if u not in leaf_nodes]
    if not core_nodes:
        core_nodes = list(nodes)
        leaf_nodes.clear()
        leaf_groups.clear()

    core_set = set(core_nodes)
    core_bonds = {k: b for k, b in bonds.items() if k[0] in core_set and k[1] in core_set}
    # Augment core colors with counts of attached leaf types
    leaf_counts_on_parent: Dict[str, Tuple[Tuple[Any, ...], ...]] = {}
    for u in core_nodes:
        attached = sorted(
            (sig, b_ord, len(members))
            for (p, sig, b_ord), members in leaf_groups.items()
            if p == u
        )
        leaf_counts_on_parent[u] = tuple(attached)

    base_colors = _wl_colors(core_nodes, atoms, lab_state.atom_meta, core_bonds, include_isotopes=False)
    combined_sig = {u: (base_colors[u], leaf_counts_on_parent[u]) for u in core_nodes}
    pal = {s: i for i, s in enumerate(sorted(set(combined_sig.values())))}
    colors = {u: pal[combined_sig[u]] for u in core_nodes}

    buckets: Dict[int, List[str]] = collections.defaultdict(list)
    for u in core_nodes:
        buckets[colors[u]].append(u)

    # Order core vertices: non-singleton color classes with highest degree first
    ordered_core = sorted(
        core_nodes,
        key=lambda u: (len(buckets[colors[u]]) == 1, len(buckets[colors[u]]), -len(adj[u]), u),
    )

    core_adj = {
        u: {v: o for v, o in adj[u].items() if v in core_set}
        for u in core_nodes
    }

    mapping: Dict[str, str] = {}
    used: Set[str] = set()
    results: List[Dict[str, str]] = []

    def backtrack(idx: int) -> None:
        if len(results) >= max_automorphisms:
            return
        if idx == len(ordered_core):
            full_map = dict(mapping)
            # Map leaf groups canonically from parent -> image_parent
            # If a parent has 2 labeled/unlabeled leaves (e.g. CH2D or CHD), include their permutations
            # up to budget if any leaf carries an isotope or origin tag!
            labeled_leaf_groups: List[Tuple[List[str], List[str]]] = []
            for (parent_u, sig, b_ord), src_leaves in sorted(leaf_groups.items()):
                img_parent = full_map[parent_u]
                dst_leaves = sorted(leaf_groups.get((img_parent, sig, b_ord), []))
                src_sorted = sorted(src_leaves)
                if len(src_sorted) != len(dst_leaves):
                    return
                has_tag = any(
                    lab_state.atom_meta.get(x, AtomMeta()).isotope is not None
                    or lab_state.atom_meta.get(x, AtomMeta()).fragment_origin is not None
                    for x in src_sorted + dst_leaves
                )
                if has_tag and len(src_sorted) <= 3:
                    labeled_leaf_groups.append((src_sorted, dst_leaves))
                else:
                    for s, d in zip(src_sorted, dst_leaves):
                        full_map[s] = d

            if not labeled_leaf_groups:
                results.append(full_map)
                return

            # Expand permutations only for tagged leaf groups (e.g. -CHD- intramolecular KIE!)
            import itertools

            def expand_leaves(g_idx: int, cur_map: Dict[str, str]) -> None:
                if len(results) >= max_automorphisms:
                    return
                if g_idx == len(labeled_leaf_groups):
                    results.append(dict(cur_map))
                    return
                src_l, dst_l = labeled_leaf_groups[g_idx]
                for perm in itertools.permutations(dst_l):
                    for s, d in zip(src_l, perm):
                        cur_map[s] = d
                    expand_leaves(g_idx + 1, cur_map)

            expand_leaves(0, full_map)
            return

        u = ordered_core[idx]
        for cand in buckets[colors[u]]:
            if cand in used:
                continue
            # Check consistency with all already-mapped neighbors of u
            ok = True
            for nb, order in core_adj[u].items():
                if nb in mapping:
                    mapped_nb = mapping[nb]
                    if core_adj[cand].get(mapped_nb) != order:
                        ok = False
                        break
            if not ok:
                continue
            # Also verify non-edges to already-mapped vertices
            for prev_u, prev_img in mapping.items():
                if prev_u not in core_adj[u] and prev_img in core_adj[cand]:
                    ok = False
                    break
            if not ok:
                continue
            mapping[u] = cand
            used.add(cand)
            backtrack(idx + 1)
            used.remove(cand)
            del mapping[u]

    backtrack(0)
    return results or [{u: u for u in nodes}]


def automorphism_orbits(lab_state: LabState) -> Dict[str, Tuple[str, ...]]:
    """Return for every atom ID the sorted tuple of electronically equivalent atom IDs."""
    autos = electronic_automorphisms(lab_state)
    orbits: Dict[str, Set[str]] = collections.defaultdict(set)
    for perm in autos:
        for src, dst in perm.items():
            orbits[src].add(dst)
    return {aid: tuple(sorted(members)) for aid, members in orbits.items()}
