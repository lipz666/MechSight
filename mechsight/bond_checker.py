"""Elementary bond-change checking (Delta_bond <= 2) via RXNMapper and RDKit.

Implements Eq. S8 from Supporting Information Section S7.1:
    Delta_bond(t) = |B_p(t) \\ B_r(t)| + |B_r(t) \\ B_p(t)| + |B_leave(t)|
Transitions with Delta_bond >= gap_threshold (default 3) or unparsable SMILES
are flagged as potential multi-bond shortcuts hiding transient intermediates.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem

from .chemistry import build_step_reactions
from .schema import GapFlag, ReactionCase

_SUBPROCESS_MAP_CODE = (
    "import json, sys\n"
    "from rxnmapper import RXNMapper\n"
    "rxns = json.load(sys.stdin)\n"
    "mapper = RXNMapper()\n"
    "out = {}\n"
    "for i in range(0, len(rxns), 16):\n"
    "    chunk = rxns[i:i+16]\n"
    "    try:\n"
    "        res = mapper.get_attention_guided_atom_maps(chunk, canonicalize_rxns=False)\n"
    "    except Exception:\n"
    "        res = []\n"
    "        for r in chunk:\n"
    "            try:\n"
    "                res.append(mapper.get_attention_guided_atom_maps([r], canonicalize_rxns=False)[0])\n"
    "            except Exception:\n"
    "                res.append(None)\n"
    "    for r, m in zip(chunk, res):\n"
    "        out[r] = m['mapped_rxn'] if m else None\n"
    "print(json.dumps(out))\n"
)


def count_heavy_bond_changes(mapped_rxn: str) -> int | None:
    """Compute heavy-atom bond changes Delta_bond(t) from an atom-mapped reaction SMILES.

    Counts bonds formed or broken between mapped heavy atoms preserved in the
    next state plus heavy-atom bonds broken upon departure of unmapped leaving
    atoms (Eq. S8). Hydrogen bonds are excluded.
    """
    if ">>" not in mapped_rxn:
        return None
    r_smi, p_smi = mapped_rxn.split(">>", 1)
    rm = Chem.MolFromSmiles(r_smi, sanitize=False)
    pm = Chem.MolFromSmiles(p_smi, sanitize=False)
    if rm is None or pm is None:
        return None

    in_product = {atom.GetAtomMapNum() for atom in pm.GetAtoms() if atom.GetAtomMapNum()}

    def _bond_set(mol: Chem.Mol, keep_fn: Any) -> set[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtom(), bond.GetEndAtom()
            if a.GetSymbol() == "H" or b.GetSymbol() == "H":
                continue
            x, y = a.GetAtomMapNum(), b.GetAtomMapNum()
            if keep_fn(x, y):
                out.add((x, y) if x <= y else (y, x))
        return out

    br = _bond_set(rm, lambda x, y: x in in_product and y in in_product)
    bp = _bond_set(pm, lambda x, y: bool(x and y))
    leaving = len(_bond_set(rm, lambda x, y: (x in in_product) != (y in in_product)))
    return len(bp - br) + len(br - bp) + leaving


class BondChangeChecker:
    """Cache-backed RXNMapper + RDKit elementary bond-change detector."""

    def __init__(
        self,
        *,
        gap_threshold: int = 3,
        max_rxn_len: int = 450,
        cache_path: str | Path | None = None,
        rxnmapper_python: str | Path | None = None,
        precomputed_cache: dict[str, int | None] | None = None,
    ) -> None:
        self.gap_threshold = gap_threshold
        self.max_rxn_len = max_rxn_len
        self.cache_path = Path(cache_path) if cache_path else None
        env_py = os.environ.get("MECHSIGHT_RXNMAPPER_PYTHON")
        self.rxnmapper_python = str(rxnmapper_python or env_py) if (rxnmapper_python or env_py) else None
        self._cache: dict[str, int | None] = {}
        if self.cache_path and self.cache_path.exists():
            self._cache.update(json.loads(self.cache_path.read_text()))
        if precomputed_cache:
            self._cache.update(precomputed_cache)
        self._inproc_mapper: Any = None

    def _save_cache(self) -> None:
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache))

    def _map_uncached(self, rxns: list[str]) -> None:
        todo = sorted({r for r in rxns if r and r not in self._cache})
        if not todo:
            return
        if self.rxnmapper_python:
            proc = subprocess.run(
                [self.rxnmapper_python, "-c", _SUBPROCESS_MAP_CODE],
                input=json.dumps(todo),
                text=True,
                capture_output=True,
                check=True,
            )
            mapped = json.loads(proc.stdout)
            for r in todo:
                mr = mapped.get(r)
                self._cache[r] = count_heavy_bond_changes(mr) if mr else None
            self._save_cache()
            return
        try:
            from rxnmapper import RXNMapper
        except ImportError as exc:
            raise RuntimeError(
                f"Cannot map {len(todo)} uncached step reaction(s): install `mechsight[rxnmapper]`, "
                "provide a precomputed cache_path, or set MECHSIGHT_RXNMAPPER_PYTHON to an env with rxnmapper."
            ) from exc

        if self._inproc_mapper is None:
            self._inproc_mapper = RXNMapper()
        for i in range(0, len(todo), 16):
            chunk = todo[i : i + 16]
            try:
                res = self._inproc_mapper.get_attention_guided_atom_maps(chunk, canonicalize_rxns=False)
            except Exception:
                res = []
                for r in chunk:
                    try:
                        res.append(self._inproc_mapper.get_attention_guided_atom_maps([r], canonicalize_rxns=False)[0])
                    except Exception:
                        res.append(None)
            for r, m in zip(chunk, res):
                self._cache[r] = count_heavy_bond_changes(m["mapped_rxn"]) if m else None
        self._save_cache()

    def prefetch(self, case: ReactionCase, pathways: Iterable[list[dict[str, Any]]]) -> None:
        """Atom-map and cache all unmapped transitions across a list of candidate pathways."""
        rxns: list[str] = []
        for mech in pathways:
            for rxn in build_step_reactions(case.reactants, mech, max_len=self.max_rxn_len):
                if rxn:
                    rxns.append(rxn)
        self._map_uncached(rxns)

    def delta_bond(self, rxn: str | None) -> int | str | None:
        """Return ``'unparsable'``, integer ``Delta_bond``, or ``None`` if unmapped."""
        if rxn is None:
            return "unparsable"
        if rxn == "":
            return None
        if rxn not in self._cache:
            self._map_uncached([rxn])
        return self._cache.get(rxn)

    def detect_gaps(self, case: ReactionCase, pathway: list[dict[str, Any]]) -> list[GapFlag]:
        """Identify suspected multi-bond shortcut steps and unparsable intermediates."""
        step_rxns = build_step_reactions(case.reactants, pathway, max_len=self.max_rxn_len)
        self._map_uncached([r for r in step_rxns if r])
        flags: list[GapFlag] = []
        for idx, rxn in enumerate(step_rxns):
            val = self.delta_bond(rxn)
            if val == "unparsable":
                flags.append(
                    GapFlag(step_index=idx, reason="species does not parse", delta_bond=None, step_rxn=rxn)
                )
            elif isinstance(val, int) and val >= self.gap_threshold:
                flags.append(
                    GapFlag(
                        step_index=idx,
                        reason="several heavy-atom bonds change in this one step",
                        delta_bond=val,
                        step_rxn=rxn,
                    )
                )
        return flags

    def render_annotated_candidates(
        self,
        case: ReactionCase,
        candidates: list[list[dict[str, Any]]],
    ) -> tuple[str, list[list[GapFlag]]]:
        """Format candidate pathways with ``[FLAGGED: ...]`` markers for Auto/Guided prompts."""
        self.prefetch(case, candidates)
        all_flags: list[list[GapFlag]] = []
        blocks: list[str] = []
        for c_idx, mech in enumerate(candidates, 1):
            flags = self.detect_gaps(case, mech)
            all_flags.append(flags)
            by_step = {flag.step_index: flag.reason for flag in flags}
            lines: list[str] = []
            for s_idx, step in enumerate(mech, 1):
                smi = step.get("intermediate_smiles", "")
                subtype = step.get("subtype", "")
                marker = f"   [FLAGGED: {by_step[s_idx - 1]}]" if (s_idx - 1) in by_step else ""
                lines.append(f"  step {s_idx}: {subtype} -> {smi}{marker}")
            blocks.append(f"Candidate {c_idx}:\n" + "\n".join(lines))
        return "\n\n".join(blocks), all_flags
