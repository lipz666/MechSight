"""Deterministic RDKit cheminformatics, canonical species keys, and parsers.

Implements the exact molecular representations, heavy-atom budget filters,
minimal reactant-partner covering, and strict/equivalent keys described in
Supporting Information Sections S3.1, S7.1, and S10.1.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from itertools import combinations
from typing import Any, Iterable

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")


def canonicalize_smiles(smi: str | None) -> str | None:
    """Parse and return RDKit canonical SMILES, or ``None`` if invalid."""
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else None


def heavy_atom_counts(smi: str | None) -> Counter[str] | None:
    """Return non-hydrogen element counts for a valid SMILES, or ``None``."""
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Counter(atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetSymbol() != "H")


def molecular_formula(smi: str | None) -> str | None:
    """Return RDKit molecular formula string, or ``None`` if invalid."""
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    return rdMolDescriptors.CalcMolFormula(mol) if mol is not None else None


def _key_from_sigma_graph(
    heavy: dict[int, str],
    hcount: dict[int, int],
    bonds: list[tuple[int, int]],
    with_h: bool = True,
) -> str:
    """Canonical SMILES of heavy-atom sigma graph (single bonds, zero charges)."""
    rw = Chem.RWMol()
    idx_map: dict[int, int] = {}
    for nid in sorted(heavy):
        atom = Chem.Atom(heavy[nid])
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(int(hcount[nid]) if with_h else 0)
        idx_map[nid] = rw.AddAtom(atom)
    for a, b in bonds:
        if a in idx_map and b in idx_map and rw.GetBondBetweenAtoms(idx_map[a], idx_map[b]) is None:
            rw.AddBond(idx_map[a], idx_map[b], Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return Chem.MolToSmiles(mol)


def sigma_graph_keys(smi: str | None, *, include_skeleton: bool = False) -> set[str]:
    """Return per-component heavy-atom sigma-bond graph keys (``kappa_strict``).

    Matches Supporting Information Section S3.1: heavy-atom connectivity with
    explicit per-atom attached hydrogen counts, ignoring formal charges and
    pi-bond Kekule localization.
    """
    if not smi:
        return set()
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            return set()
        try:
            mol.UpdatePropertyCache(strict=False)
        except Exception:
            return set()
    keys: set[str] = set()
    for frag in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False):
        try:
            frag.UpdatePropertyCache(strict=False)
        except Exception:
            continue
        heavy: dict[int, str] = {}
        hcount: dict[int, int] = {}
        for atom in frag.GetAtoms():
            if atom.GetSymbol() == "H":
                continue
            heavy[atom.GetIdx()] = atom.GetSymbol()
            hcount[atom.GetIdx()] = atom.GetTotalNumHs(includeNeighbors=True)
        if not heavy:
            continue
        bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in frag.GetBonds()]
        keys.add(_key_from_sigma_graph(heavy, hcount, bonds, with_h=True))
        if include_skeleton:
            keys.add(_key_from_sigma_graph(heavy, hcount, bonds, with_h=False))
    return keys


def fixed_h_inchi_keys(smi: str | None) -> set[str]:
    """Return fixed-hydrogen standard InChI keys per component (oMe-Gold strict key)."""
    if not smi:
        return set()
    out: set[str] = set()
    for part in smi.split("."):
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            return set()
        key = None
        try:
            inchi = Chem.MolToInchi(mol, options="/FixedH")
            if inchi:
                key = inchi
        except Exception:
            key = None
        out.add(key or f"smi:{Chem.MolToSmiles(mol)}")
    return out


def equiv_connectivity_keys(smi: str | None, *, allow_unsanitized: bool = False) -> set[str]:
    """Return 14-char standard InChIKey connectivity blocks (``kappa_equiv``).

    Unifies protonation states, mobile-hydrogen tautomers, and stereochemical
    depictions while distinguishing oxidation states and constitutional connectivity.
    """
    if not smi:
        return set()
    out: set[str] = set()
    for part in smi.split("."):
        mol = Chem.MolFromSmiles(part)
        if mol is None and allow_unsanitized:
            mol = Chem.MolFromSmiles(part, sanitize=False)
            if mol is not None:
                try:
                    mol.UpdatePropertyCache(strict=False)
                except Exception:
                    mol = None
        if mol is None:
            continue
        try:
            ik = Chem.MolToInchiKey(mol)
        except Exception:
            ik = ""
        if ik:
            out.add(ik.split("-")[0])
    return out


def select_minimal_partners(
    deficit: Counter[str],
    partners: list[str],
    max_partners: int = 3,
) -> list[str]:
    """Pick the smallest subset (<= ``max_partners``) of reactant/reagent species
    supplying missing heavy atoms for an elementary step transition (SI S7.1).
    """
    if not deficit:
        return []
    best: list[str] = []
    best_key: tuple[int, int] | None = None
    for r in range(1, max_partners + 1):
        for combo in combinations(partners, r):
            have: Counter[str] = Counter()
            for partner in combo:
                have += heavy_atom_counts(partner) or Counter()
            key = (sum((deficit - have).values()), sum(have.values()))
            if best_key is None or key < best_key:
                best, best_key = list(combo), key
        if best_key is not None and best_key[0] == 0:
            break
    return best


def build_step_reactions(
    reactants: Iterable[str],
    steps: list[dict[str, Any]],
    max_len: int = 450,
) -> list[str | None]:
    """Build ``prev[.partner...]>>next`` reaction strings for each step in ``steps``.

    Returns ``None`` if a step's SMILES cannot be parsed, or ``""`` if a step
    cannot form a non-empty left side or exceeds ``max_len``.
    """
    canon_partners = [canonicalize_smiles(s) for s in reactants]
    partners = [p for p in dict.fromkeys(canon_partners) if p]
    prev: str | None = None
    out: list[str | None] = []
    for st in steps:
        nxt = canonicalize_smiles(st.get("intermediate_smiles"))
        if nxt is None:
            out.append(None)
            continue
        have = heavy_atom_counts(prev) if prev else Counter()
        deficit = (heavy_atom_counts(nxt) or Counter()) - have
        left = list(dict.fromkeys(([prev] if prev else []) + select_minimal_partners(deficit, partners)))
        if not left:
            left = [prev] if prev is not None else []
        if not left or any(x is None for x in left):
            out.append("")
            prev = nxt
            continue
        rxn = ".".join(left) + ">>" + nxt
        out.append(rxn if len(rxn) <= max_len else "")
        prev = nxt
    return out


def parse_tagged_lines(text: str | None, tag: str) -> list[str]:
    """Extract non-empty lines inside ``<tag>...</tag>`` (stripping backticks)."""
    match = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text or "", re.S)
    if not match:
        return []
    return [ln.strip().strip("`").strip() for ln in match.group(1).splitlines() if ln.strip()]


def filter_proposed_states(
    reactants: Iterable[str],
    products: Iterable[str],
    answer_text: str | None,
    *,
    tag: str = "states",
) -> list[tuple[str, str]]:
    """Deterministically validate proposed intermediate SMILES (SI S7.1 / S10.1).

    Filters candidate lines in ``<states>...</states>`` (supporting either
    ``location: SMILES`` or bare ``SMILES``) by:
      1. RDKit sanitization and valence check,
      2. non-duplication of starting reactants or final products,
      3. heavy-atom element budget compatibility with the reactant/reagent pool,
      4. canonical SMILES deduplication (preserving order).
    """
    budget: Counter[str] = Counter()
    for smi in reactants:
        budget += heavy_atom_counts(smi) or Counter()
    endpoints = {canonicalize_smiles(s) for s in list(reactants) + list(products)}
    endpoints.discard(None)

    keep: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in parse_tagged_lines(answer_text, tag):
        loc, sep, rhs = line.rpartition(":")
        smi = rhs.strip() if sep else line.strip()
        canon = canonicalize_smiles(smi)
        heavy = heavy_atom_counts(smi) if canon else None
        if (
            canon
            and canon not in endpoints
            and heavy
            and not (heavy - budget)
            and canon not in seen
        ):
            seen.add(canon)
            keep.append((loc.strip() if sep and loc.strip() else "-", canon))
    return keep


def parse_mechanism_json(text: str | None) -> list[dict[str, Any]] | None:
    """Parse an ordered mechanism JSON list from an LLM response."""
    if not text:
        return None
    m = re.search(r"\[ANSWER\](.*?)\[/ANSWER\]", text, re.S)
    body = m.group(1) if m else text
    body = re.sub(r"```(?:json)?", "", body)
    lo, hi = body.find("["), body.rfind("]")
    if lo < 0 or hi <= lo:
        return None
    try:
        out = json.loads(body[lo : hi + 1])
    except Exception:
        return None
    if isinstance(out, dict):
        out = out.get("mechanism")
    if not isinstance(out, list):
        return None
    steps = [s for s in out if isinstance(s, dict) and isinstance(s.get("intermediate_smiles"), str)]
    return steps if steps else None


def parse_enumerated_paths(text: str | None, count: int = 3) -> list[list[dict[str, Any]] | None]:
    """Parse ``[PATH1]...[/PATH1]`` through ``[PATH{count}]...[/PATH{count}]``."""
    out: list[list[dict[str, Any]] | None] = []
    for idx in range(1, count + 1):
        m = re.search(rf"\[PATH{idx}\](.*?)\[/PATH{idx}\]", text or "", re.S)
        out.append(parse_mechanism_json(m.group(1)) if m else None)
    return out


def deduplicate_candidate_pathways(
    pathways: Iterable[list[dict[str, Any]] | None],
    max_candidates: int = 8,
) -> list[list[dict[str, Any]]]:
    """Deduplicate pathways by canonical intermediate-SMILES sequence (SI S7.1)."""
    out: list[list[dict[str, Any]]] = []
    seen: set[tuple[str | None, ...]] = set()
    for mech in pathways:
        if not mech:
            continue
        key = tuple(canonicalize_smiles(step.get("intermediate_smiles")) for step in mech)
        if key not in seen:
            seen.add(key)
            out.append(mech)
            if len(out) >= max_candidates:
                break
    return out
