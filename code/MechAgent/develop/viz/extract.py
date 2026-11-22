"""Extract one rollout into a coordinate-carrying JSON payload for the viz prototype."""
import json, sys, math, re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]   # .../MechAgent
sys.path.insert(0, str(REPO))
from develop import standalone_chemistry as sc

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

BONDT = {
    "single": Chem.BondType.SINGLE,
    "double": Chem.BondType.DOUBLE,
    "triple": Chem.BondType.TRIPLE,
    "aromatic": Chem.BondType.AROMATIC,
    "transition": Chem.BondType.SINGLE,
    "dative": Chem.BondType.SINGLE,
}


def state_dict(state):
    sig = sc.state_signature(state)
    atoms = {k: dict(v) for k, v in sig["atoms"].items()}
    bonds = []
    for key, b in sig["bonds"].items():
        a1, a2 = key.split("|")
        bonds.append({"a": a1, "b": a2, "kind": b["kind"], "order": b["order"]})
    return {"atoms": atoms, "bonds": bonds}


def element_of(label, atoms):
    return atoms[label]["element"]


def union_coords(states):
    """One shared 2D layout for every atom label seen anywhere on the path."""
    atoms = {}
    for st in states:
        for label, a in st["atoms"].items():
            atoms.setdefault(label, a)
    order = sorted(atoms)
    idx = {label: i for i, label in enumerate(order)}

    # union of bonds, preferring the highest order seen (keeps rings drawn as rings)
    ubonds = {}
    for st in states:
        for b in st["bonds"]:
            key = tuple(sorted((b["a"], b["b"])))
            prev = ubonds.get(key)
            if prev is None or b["order"] > prev["order"]:
                ubonds[key] = b

    mol = Chem.RWMol()
    for label in order:
        a = Chem.Atom(atoms[label]["element"])
        a.SetNoImplicit(True)
        a.SetFormalCharge(int(atoms[label]["charge"]))
        mol.AddAtom(a)
    # degree guard: skip union bonds that would over-coordinate an atom for layout
    deg = {label: 0 for label in order}
    maxdeg = {"H": 1, "C": 4, "N": 4, "O": 2, "S": 4, "P": 5, "F": 1, "Cl": 1, "Br": 1, "I": 1}
    kept = []
    for key, b in sorted(ubonds.items(), key=lambda kv: -kv[1]["order"]):
        a1, a2 = key
        e1, e2 = atoms[a1]["element"], atoms[a2]["element"]
        if deg[a1] >= maxdeg.get(e1, 6) or deg[a2] >= maxdeg.get(e2, 6):
            continue
        deg[a1] += 1
        deg[a2] += 1
        mol.AddBond(idx[a1], idx[a2], BONDT.get(b["kind"], Chem.BondType.SINGLE))
        kept.append(key)

    m = mol.GetMol()
    m.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(m)
    # CoordGen lays out fused and macro rings far better than the built-in
    # generator, which matters because this one layout serves every state
    try:
        from rdkit.Chem import rdCoordGen
        rdCoordGen.AddCoords(m)
    except Exception:
        AllChem.Compute2DCoords(m)
    conf = m.GetConformer()
    coords = {}
    for label in order:
        p = conf.GetAtomPosition(idx[label])
        coords[label] = [round(p.x, 4), round(-p.y, 4)]  # flip y for screen space
    return coords


ATOM_LABEL = re.compile(r"^[A-Z][a-z]?[0-9]+$")


def endpoint_json(ep):
    if isinstance(ep, str):
        return {"type": "atom", "atoms": [ep], "tag": None}
    vals = [str(v) for v in ep]
    atoms = [v for v in vals if ATOM_LABEL.match(v)]
    tags = [v for v in vals if not ATOM_LABEL.match(v)]
    kind = "bond" if len(atoms) >= 2 else "atom"
    return {"type": kind, "atoms": atoms, "tag": tags[0] if tags else None}


def action_json(action_dict):
    out = []
    for t in action_dict["electron_transfers"]:
        out.append({
            "electrons": t["electrons"],
            "kind": t["kind"],
            "source": endpoint_json(t["source"]),
            "target": endpoint_json(t["target"]),
        })
    return out


def replay(initial, actions):
    """Return list of (state_dict, transfers_leading_out_of_that_state, error)."""
    state = initial
    frames = [{"state": state_dict(state), "transfers": None, "error": None}]
    for act in actions:
        payload, ok = sc.extract_action_payload(act) if isinstance(act, str) else (act, {"valid": True})
        if not ok.get("valid"):
            frames[-1]["error"] = "unparseable action"
            break
        frames[-1]["transfers"] = action_json(payload)
        try:
            state, _ops = sc.apply_action(state, payload)
        except Exception as exc:  # keep the arrow, mark the failure
            frames[-1]["error"] = str(exc)
            break
        frames.append({"state": state_dict(state), "transfers": None, "error": None})
    return frames


def build(record):
    r = record["reaction"]
    initial = sc.parse_pgl_literal(r["initial_state_pgl"])
    target = sc.parse_pgl_literal(r["target_state_pgl"])

    gt_frames = replay(initial, record["gt"]["actions_pgl"])
    pred_actions = [a["action_pgl"] for a in record["actions"] if a.get("accepted_into_history")]
    pred_frames = replay(initial, pred_actions)

    all_states = [f["state"] for f in gt_frames] + [f["state"] for f in pred_frames] + [state_dict(target)]
    coords = union_coords(all_states)

    steps_meta = []
    plan = record.get("planner_plan") or {}
    for a in record["actions"]:
        if not a.get("accepted_into_history"):
            continue
        steps_meta.append({
            "summary": a.get("event_summary"),
            "decision": a.get("planner_decision"),
            "assessment": a.get("planner_assessment"),
            "review_round": a.get("review_round"),
            "directive_id": a.get("directive_id"),
        })

    return {
        "reaction_id": r["reaction_id"],
        "repeat": r["repeat_index"],
        "success": r["success"],
        "stop_reason": r["stop_reason"],
        "input_mode": r["task_input_mode"],
        "coords": coords,
        "target_state": state_dict(target),
        "gt": gt_frames,
        "pred": pred_frames,
        "pred_steps": steps_meta,
        "plan": {
            "reaction_family": plan.get("reaction_family"),
            "reaction_summary": plan.get("reaction_summary"),
            "global_mechanism": plan.get("global_mechanism"),
            "elementary_steps": plan.get("elementary_steps"),
        },
        "assessment": record["gt"]["evaluator_assessment"],
        "all_proposals": [
            {
                "proposal_id": a.get("proposal_id"),
                "review_round": a.get("review_round"),
                "decision": a.get("planner_decision"),
                "valid": (a.get("validation_result") or {}).get("valid"),
                "accepted": a.get("accepted_into_history"),
                "summary": a.get("event_summary"),
                "objection": a.get("plan_objection"),
            }
            for a in record["actions"]
        ],
    }


if __name__ == "__main__":
    src = Path(sys.argv[1])
    wanted = sys.argv[2]
    repeat = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    out = Path(sys.argv[4])
    for line in src.open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["reaction"]["reaction_id"] == wanted and rec["reaction"]["repeat_index"] == repeat:
            payload = build(rec)
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            print("wrote", out, len(json.dumps(payload)), "bytes")
            break
    else:
        print("not found")
