"""Score our rollouts with FukuyamaBench's own metric, for a like-for-like read.

Their evaluation is checkpoint-based and deliberately forgiving in two ways our
metrics are not: `ckpt.txt` collapses adjacent equivalent steps, and a
checkpoint is satisfied by scanning forward through the prediction, so extra
predicted steps cost nothing. Products are compared as canonical SMILES sets
with subset leniency, meaning a prediction that names the main product but not
the leaving group still matches. `exact_match` requires every checkpoint hit.

Reproducing it here means converting each state our environment reached back to
SMILES, which is the only way to say whether their reported number and ours
describe the same thing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from rdkit import Chem, RDLogger

from .standalone_chemistry import (
    apply_action,
    extract_action_payload,
    parse_pgl_literal,
)

RDLogger.DisableLog("rdApp.*")

_BOND = {
    "single": Chem.BondType.SINGLE,
    "double": Chem.BondType.DOUBLE,
    "triple": Chem.BondType.TRIPLE,
    "aromatic": Chem.BondType.AROMATIC,
}


def state_to_smiles(state: Any, only: Optional[Set[str]] = None) -> Set[str]:
    """Canonical SMILES of the species in a PGL state.

    With ``only``, keep just the connected species that contain one of those
    atoms. Our states hold the whole flask, while a workbook step lists only
    what that step produced; comparing the whole flask under their
    "prediction is a subset of the reference" leniency fails on every spectator
    and byproduct, which is a representation mismatch rather than a chemistry
    error.
    """

    from .standalone_chemistry import pgl

    atoms = dict(pgl.iter_atoms(state))
    bond_map = pgl.bond_map(state)
    if only is not None:
        neighbours: Dict[str, Set[str]] = {name: set() for name in atoms}
        for first, second in bond_map:
            neighbours.setdefault(first, set()).add(second)
            neighbours.setdefault(second, set()).add(first)
        keep: Set[str] = set()
        frontier = [name for name in only if name in atoms]
        while frontier:
            name = frontier.pop()
            if name in keep:
                continue
            keep.add(name)
            frontier.extend(neighbours.get(name, ()))
        atoms = {name: atom for name, atom in atoms.items() if name in keep}
    mol = Chem.RWMol()
    index_of: Dict[str, int] = {}
    for name, atom in sorted(atoms.items()):
        rd = Chem.Atom(str(atom.element))
        rd.SetFormalCharge(int(atom.charge or 0))
        rd.SetNoImplicit(True)
        rd.SetNumExplicitHs(int(atom.Hcount or 0))
        index_of[name] = mol.AddAtom(rd)
    for (first, second), bond in bond_map.items():
        if first not in index_of or second not in index_of:
            continue
        mol.AddBond(index_of[first], index_of[second], _BOND.get(str(bond.kind), Chem.BondType.SINGLE))
    try:
        finished = mol.GetMol()
        Chem.SanitizeMol(finished)
        # States now carry explicit hydrogen atoms for protons that move. The
        # workbook's reference SMILES keep hydrogens implicit, so fold them back
        # before canonicalizing or every protonated species fails to match.
        finished = Chem.RemoveHs(finished)
    except Exception:
        return set()
    return {
        Chem.MolToSmiles(fragment)
        for fragment in Chem.GetMolFrags(finished, asMols=True, sanitizeFrags=False)
    }


def canonical_set(smiles: Sequence[str]) -> Set[str]:
    out: Set[str] = set()
    for item in smiles:
        mol = Chem.MolFromSmiles(str(item))
        if mol is not None:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            out.add(Chem.MolToSmiles(mol))
    return out


def matches(predicted: Set[str], gold: Set[str], *, lenient: bool = True) -> bool:
    """Their comparison: equal sets, or predicted a subset when lenient."""

    if not predicted or not gold:
        return False
    if predicted == gold:
        return True
    return bool(lenient and predicted.issubset(gold))


def _touched(before: Dict[str, Any], after: Dict[str, Any]) -> Set[str]:
    """Atoms whose bonds or formal charge an action changed."""

    touched: Set[str] = set()
    before_bonds = before.get("bonds", {})
    after_bonds = after.get("bonds", {})
    for key in set(before_bonds) ^ set(after_bonds):
        touched.update(key.split("|"))
    for key in set(before_bonds) & set(after_bonds):
        if before_bonds[key].get("order") != after_bonds[key].get("order"):
            touched.update(key.split("|"))
    for name, atom in after.get("atoms", {}).items():
        if before.get("atoms", {}).get(name, {}).get("charge") != atom.get("charge"):
            touched.add(name)
    return touched


def predicted_states(rollout: Dict[str, Any], *, scope: str = "touched") -> List[Set[str]]:
    """Species after each accepted action.

    ``touched`` (default) keeps only species containing an atom that action
    changed, which is the analogue of a workbook step's product list.
    ``state`` keeps the whole flask, which is how this scorer first worked and
    is kept only so the two readings can be compared.
    """

    from .standalone_chemistry import state_signature

    state = parse_pgl_literal(rollout["reaction"]["initial_state_pgl"])
    out: List[Set[str]] = []
    for record in rollout.get("actions", []):
        if not record.get("accepted_into_history"):
            continue
        action, _ = extract_action_payload(f"<action>{record['action_pgl']}</action>")
        before = state_signature(state)
        state, _ = apply_action(state, action)
        if scope == "state":
            out.append(state_to_smiles(state))
        else:
            out.append(state_to_smiles(state, only=_touched(before, state_signature(state))))
    return out


def checkpoint_exact_match(
    predicted: Sequence[Set[str]], checkpoints: List[List[Set[str]]]
) -> Dict[str, Any]:
    """Their forward scan: each checkpoint may be met by any later predicted step."""

    cursor = 0
    hit = 0
    for options in checkpoints:
        found = False
        while cursor < len(predicted) and not found:
            for gold in options:
                if matches(predicted[cursor], gold):
                    found = True
                    break
            cursor += 1
        if found:
            hit += 1
        else:
            break
    return {
        "checkpoints_total": len(checkpoints),
        "checkpoints_correct": hit,
        "exact_match": hit == len(checkpoints),
    }


def load_checkpoints(case_dir: Path) -> Optional[List[List[Set[str]]]]:
    ckpt = case_dir / "ckpt.txt"
    rxn = case_dir / "rxn.json"
    if not ckpt.exists() or not rxn.exists():
        return None
    steps = json.loads(rxn.read_text())["mechanism_rxn"]
    by_id = {}
    for step in steps:
        try:
            products = step["rxn_smiles"].split(">>")[1]
        except Exception:
            continue
        by_id[str(step["step"])] = canonical_set(products.split("."))
    groups: List[List[Set[str]]] = []
    for line in ckpt.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        options = [by_id[i.strip()] for i in line.split(",") if i.strip() in by_id]
        if options:
            groups.append(options)
    return groups or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mechanisms_dir", type=Path)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--partition", default=None)
    parser.add_argument(
        "--scope",
        choices=["touched", "state"],
        default="touched",
        help="Compare the species an action touched (default) or the whole flask.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for line in args.rollouts.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rollout = json.loads(line)
        case_id = str(rollout["reaction"]["reaction_id"]).split("::")[-1]
        if args.partition and not case_id.startswith(args.partition):
            continue
        checkpoints = load_checkpoints(args.mechanisms_dir / case_id)
        if checkpoints is None:
            continue
        result = checkpoint_exact_match(predicted_states(rollout, scope=args.scope), checkpoints)
        result["case_id"] = case_id
        rows.append(result)

    n = len(rows)
    exact = sum(r["exact_match"] for r in rows)
    total_ckpt = sum(r["checkpoints_total"] for r in rows)
    hit_ckpt = sum(r["checkpoints_correct"] for r in rows)
    print(f"cases {n}" + (f" (partition {args.partition})" if args.partition else ""))
    print(f"  exact match (their metric)  {exact}/{n} = {exact / n:.1%}" if n else "  no cases")
    if total_ckpt:
        print(f"  checkpoints hit             {hit_ckpt}/{total_ckpt} = {hit_ckpt / total_ckpt:.1%}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
