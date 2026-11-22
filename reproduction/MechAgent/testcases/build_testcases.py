"""Pick a spread of mechanism test cases and write them out with SMILES inputs.

Selection is deliberate rather than random: a difficulty ladder from one step to
seven, every failure mode the deterministic metrics distinguish, and only cases
whose GT and predicted paths both replay cleanly, so each one can be scored,
drawn and diffed without a caveat.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
import os
from pathlib import Path

REPO = Path(os.environ.get("MECHSIGHT_ROOT", "."))
MECH = REPO / "MechAgent"
sys.path.insert(0, str(MECH))
sys.path.insert(0, str(MECH / "develop" / "viz"))

from develop import standalone_chemistry as sc          # noqa: E402
from develop import metrics as M                        # noqa: E402
from extract import replay, state_dict                  # noqa: E402

from rdkit import Chem, RDLogger                        # noqa: E402

RDLogger.DisableLog("rdApp.*")

RUN = MECH / "outputs" / "gemini37_oracle_steps"
CORE = REPO / "core" / "mechfinder_merged_pgl_paths.core.jsonl"

BOND = {
    "single": Chem.BondType.SINGLE,
    "double": Chem.BondType.DOUBLE,
    "triple": Chem.BondType.TRIPLE,
    "aromatic": Chem.BondType.AROMATIC,
    "transition": Chem.BondType.SINGLE,
    "dative": Chem.BondType.SINGLE,
}


def state_smiles(state) -> str:
    """A PGL ReactionState as SMILES, exactly the species the agent is given.

    PGL records a hydrogen as an atom only where the mechanism touches it, so
    implicit hydrogens are left to RDKit's valence model — forcing them off
    turns every ordinary carbon into a radical.
    """
    sig = state_dict(state)
    order = sorted(sig["atoms"])
    idx = {label: i for i, label in enumerate(order)}
    rw = Chem.RWMol()
    for label in order:
        info = sig["atoms"][label]
        atom = Chem.Atom(info["element"])
        atom.SetFormalCharge(int(info["charge"]))
        rw.AddAtom(atom)
    for b in sig["bonds"]:
        rw.AddBond(idx[b["a"]], idx[b["b"]], BOND.get(b["kind"], Chem.BondType.SINGLE))
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
        try:
            return Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False))
        except Exception:
            return Chem.MolToSmiles(mol)


def strip_maps(smiles: str) -> str:
    """Canonical SMILES without atom-map numbers, for feeding to other tools."""
    parts = []
    for fragment in smiles.split("."):
        mol = Chem.MolFromSmiles(fragment)
        if mol is None:
            return smiles
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        parts.append(Chem.MolToSmiles(mol))
    return ".".join(parts)


def net_charge_of(smiles: str) -> int:
    total = 0
    for fragment in smiles.split("."):
        mol = Chem.MolFromSmiles(fragment)
        if mol is None:
            return 0
        total += sum(a.GetFormalCharge() for a in mol.GetAtoms())
    return total


def core_smiles(row_indices):
    """USPTO reaction SMILES for the rows we need, streamed from the 840 MB core file."""
    wanted, out = set(row_indices), {}
    with CORE.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["source_row_index"] in wanted:
                out[row["source_row_index"]] = {
                    "reaction_smiles": row["source_reaction_smiles"],
                    "gt_steps": row["merged_step_count"],
                    "reaction_class": row["reaction_class"],
                }
                if len(out) == len(wanted):
                    break
    return out


def main() -> None:
    report = json.loads((RUN / "report.json").read_text(encoding="utf-8"))
    scores = {(p["reaction_id"], p["repeat_index"]): p for p in report["paths"]}

    rows = []
    for line in (RUN / "full.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["reaction"]["repeat_index"] != 1:
            continue
        rid = rec["reaction"]["reaction_id"]
        if not rid.startswith("mechfinder::"):
            continue
        pgl = rec["reaction"]["initial_state_pgl"]
        heavy = len(re.findall(r"'[A-Z][a-z]?\d+': Atom", pgl))
        if heavy > 34:
            continue
        try:
            initial = sc.parse_pgl_literal(pgl)
            target = sc.parse_pgl_literal(rec["reaction"]["target_state_pgl"])
        except Exception:
            continue

        gt_actions = rec["gt"]["actions_pgl"]
        pred_actions = [a["action_pgl"] for a in rec["actions"] if a.get("accepted_into_history")]
        gt_frames = replay(initial, gt_actions)
        pred_frames = replay(initial, pred_actions)
        if any(f["error"] for f in gt_frames) or any(f["error"] for f in pred_frames):
            continue                                     # keep only fully replayable cases

        score = scores.get((rid, 1))
        if score is None:
            continue
        agreement = sum(
            (
                Counter(
                    [k for step in M.path_steps([sc.extract_action_payload(a)[0] for a in pred_actions]) for k in step]
                )
                & Counter(
                    [k for step in M.path_steps([sc.extract_action_payload(a)[0] for a in gt_actions]) for k in step]
                )
            ).values()
        )
        rows.append(
            {
                "reaction_id": rid,
                "reaction_class": rid.split("::")[1],
                "row_index": int(rid.split("::")[2]),
                "atoms": heavy,
                "gt_steps": len(gt_actions),
                "predicted_steps": len(pred_actions),
                "L1_transfer_set": score["transfer_set_match"],
                "L2_transfer_sequence": score["transfer_sequence_match"],
                "L3_step_partition": score["step_partition_match"],
                "diagnosis": score["boundary_diagnosis"],
                "gt_transfers": score["gold_transfers"],
                "matched_transfers": agreement,
                "env_success": rec["reaction"]["success"],
                "stop_reason": rec["reaction"]["stop_reason"],
                "pgl_initial_smiles": state_smiles(initial),
                "pgl_target_smiles": state_smiles(target),
                "pgl_net_charge": sum(
                    a["charge"] for a in state_dict(initial)["atoms"].values()
                ),
                "initial_state_pgl": pgl,
                "target_state_pgl": rec["reaction"]["target_state_pgl"],
                "gt_actions_pgl": gt_actions,
            }
        )

    # a deliberate spread: every failure mode, and a ladder of step counts
    by_diagnosis = defaultdict(list)
    for r in rows:
        by_diagnosis[r["diagnosis"]].append(r)
    for bucket in by_diagnosis.values():
        bucket.sort(key=lambda r: (r["gt_steps"], r["atoms"]))

    picked, seen = [], set()

    def take(row):
        if row and row["reaction_id"] not in seen:
            seen.add(row["reaction_id"])
            picked.append(row)

    for bucket in by_diagnosis.values():                  # every diagnosis, easiest first
        take(bucket[0])
        if len(bucket) > 2:
            take(bucket[len(bucket) // 2])
    for steps in (1, 2, 3, 4, 5, 6, 7):                   # fill the difficulty ladder
        candidates = [r for r in rows if r["gt_steps"] == steps and r["reaction_id"] not in seen]
        if candidates:
            take(sorted(candidates, key=lambda r: r["atoms"])[0])

    picked.sort(key=lambda r: (r["gt_steps"], r["atoms"]))

    core = core_smiles([r["row_index"] for r in picked])
    for r in picked:
        mapped = core.get(r["row_index"], {}).get("reaction_smiles", "")
        r["uspto_reaction_smiles"] = mapped
        reactants, _, products = mapped.partition(">>")
        r["input_smiles"] = strip_maps(reactants)
        r["product_smiles"] = strip_maps(products)
        r["uspto_net_charge"] = net_charge_of(reactants)
        notes = []
        if Chem.MolFromSmiles(r["pgl_initial_smiles"]) is None:
            notes.append("pgl initial state is not a valid valence structure")
        if Chem.MolFromSmiles(r["pgl_target_smiles"]) is None:
            notes.append("pgl target state has a bridging hydrogen (H with two bonds)")
        if r["pgl_net_charge"] != r["uspto_net_charge"]:
            notes.append(
                f"annotation drops formal charge: PGL net {r['pgl_net_charge']:+d} "
                f"vs SMILES net {r['uspto_net_charge']:+d}"
            )
        r["data_notes"] = "; ".join(notes)

    out_dir = MECH / "testcases"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "testcases.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in picked) + "\n", encoding="utf-8"
    )
    columns = [
        "reaction_id", "reaction_class", "atoms", "gt_steps", "predicted_steps",
        "L1_transfer_set", "L2_transfer_sequence", "L3_step_partition", "diagnosis",
        "matched_transfers", "gt_transfers", "env_success", "stop_reason",
        "input_smiles", "product_smiles", "uspto_reaction_smiles",
        "pgl_initial_smiles", "pgl_target_smiles", "data_notes",
    ]
    with (out_dir / "testcases.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in picked:
            writer.writerow(r)

    print(f"{len(picked)} cases -> {out_dir}")
    for r in picked:
        flags = "".join(
            "1" if r[k] else "0"
            for k in ("L1_transfer_set", "L2_transfer_sequence", "L3_step_partition")
        )
        print(
            f"  {r['gt_steps']}步 {r['atoms']:>2}原子 L{flags} {r['diagnosis']:<17} "
            f"{r['reaction_class']}"
        )


main()
