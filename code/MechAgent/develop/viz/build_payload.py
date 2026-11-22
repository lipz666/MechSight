"""Build the multi-case JSON payload consumed by the visualization prototype."""
import json, sys, re
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                       # .../MechAgent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from develop import standalone_chemistry as sc
from develop import metrics as M
from develop import state_diff as SD
from extract import replay, state_dict, union_coords, action_json
import render_rdkit as R

RUN = REPO / "outputs" / "gemini37_oracle_steps"
SRC = RUN / "full.jsonl"
CASES = [
    ("mechfinder::intramolecular_lactonization::4401", 1),
    ("mechfinder::imine_formation::136", 1),
    ("mechfinder::SN1::15", 1),
]


def keys_of(action_pgl):
    payload, ok = sc.extract_action_payload(action_pgl)
    if not ok.get("valid"):
        return []
    return [M.transfer_key(t) for t in payload["electron_transfers"]]


def annotate(frames, actions_pgl, other_pool):
    """Tag each transfer with whether the other path contains it (multiset aware)."""
    pool = Counter(other_pool)
    for frame, act in zip(frames, actions_pgl):
        if frame["transfers"] is None:
            continue
        ks = keys_of(act)
        for tr, k in zip(frame["transfers"], ks):
            tr["key"] = ["+".join(k[0]), "+".join(k[1])]
            if pool[k] > 0:
                pool[k] -= 1
                tr["match"] = "shared"
            else:
                tr["match"] = "unique"


def outstanding(state_sig, target_sig):
    d = SD.diff_signatures(state_sig, target_sig)
    return {
        "form": len(d.bonds_to_form),
        "break": len(d.bonds_to_break),
        "order": len(d.bond_order_changes),
        "charge": len(d.charge_changes),
        "total": d.shared_atom_changes,
        "text": SD.render_diff(d),
    }


def build(rec):
    r = rec["reaction"]
    initial = sc.parse_pgl_literal(r["initial_state_pgl"])
    target = sc.parse_pgl_literal(r["target_state_pgl"])
    target_sig = sc.state_signature(target)

    gt_actions = rec["gt"]["actions_pgl"]
    pred_actions = [a["action_pgl"] for a in rec["actions"] if a.get("accepted_into_history")]

    gt_frames = replay(initial, gt_actions)
    pred_frames = replay(initial, pred_actions)

    gt_keys = [k for a in gt_actions for k in keys_of(a)]
    pred_keys = [k for a in pred_actions for k in keys_of(a)]
    annotate(gt_frames, gt_actions, pred_keys)
    annotate(pred_frames, pred_actions, gt_keys)

    for frames in (gt_frames, pred_frames):
        for f in frames:
            f["outstanding"] = outstanding(
                {"atoms": {k: v for k, v in f["state"]["atoms"].items()},
                 "bonds": {"|".join(sorted((b["a"], b["b"]))): {"kind": b["kind"], "order": b["order"]}
                           for b in f["state"]["bonds"]}},
                target_sig,
            )

    cmp = M.compare_path(
        [sc.extract_action_payload(a)[0] for a in pred_actions],
        [sc.extract_action_payload(a)[0] for a in gt_actions],
        reaction_id=r["reaction_id"], repeat_index=r["repeat_index"],
    )

    all_states = [f["state"] for f in gt_frames] + [f["state"] for f in pred_frames] + [state_dict(target)]
    coords = union_coords(all_states)

    # RDKit draws the structures; every frame is normalized onto one world frame
    canvas = R.canvas_for(coords)
    world = R.global_transform(coords)
    for frames in (gt_frames, pred_frames):
        for i, f in enumerate(frames):
            nxt = frames[i + 1]["state"] if i + 1 < len(frames) else None
            drawn = R.render_frame(f, coords, canvas, world, next_state=nxt)
            f["svg"] = drawn["svg"]
            f["focus"] = drawn["focus"]

    plan = rec.get("planner_plan") or {}
    return {
        "reaction_id": r["reaction_id"],
        "label": r["reaction_id"].split("::")[1].replace("_", " "),
        "repeat": r["repeat_index"],
        "success": r["success"],
        "stop_reason": r["stop_reason"],
        "canvas": canvas,
        "coords": coords,
        "target_state": state_dict(target),
        "gt": gt_frames,
        "pred": pred_frames,
        "plan": {
            "reaction_family": plan.get("reaction_family"),
            "reaction_summary": plan.get("reaction_summary"),
            "global_mechanism": plan.get("global_mechanism"),
            "elementary_steps": plan.get("elementary_steps") or [],
        },
        "proposals": [
            {
                "proposal_id": a.get("proposal_id"),
                "directive_id": a.get("directive_id"),
                "review_round": a.get("review_round"),
                "decision": a.get("planner_decision"),
                "valid": (a.get("validation_result") or {}).get("valid"),
                "accepted": a.get("accepted_into_history"),
                "summary": a.get("event_summary"),
                "objection": a.get("plan_objection"),
                "assessment": a.get("planner_assessment"),
            }
            for a in rec["actions"]
        ],
        "assessment": rec["gt"]["evaluator_assessment"],
        "comparison": cmp.to_dict() if hasattr(cmp, "to_dict") else cmp.__dict__,
    }


def main():
    wanted = {(cid, rep) for cid, rep in CASES}
    found = {}
    for line in SRC.open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        key = (rec["reaction"]["reaction_id"], rec["reaction"]["repeat_index"])
        if key in wanted and key not in found:
            found[key] = build(rec)
    cases = [found[k] for k in CASES if k in found]
    report = json.load(open(RUN / "report.json", encoding="utf-8"))
    payload = {
        "cases": cases,
        "benchmark": {
            "num_paths": report["num_paths"],
            "num_reactions": report["num_reactions"],
            "levels": report["levels"],
            "transfer_scores": report["transfer_scores"],
            "step_scores": report["step_scores"],
            "diagnosis": report["diagnosis"],
            "step_count_relation": report["step_count_relation"],
            "h_sensitivity": report["h_sensitivity"],
        },
    }
    out = HERE / "payload.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("cases:", [c["reaction_id"] for c in cases], "size", out.stat().st_size)

    template = (HERE / "page.html").read_text(encoding="utf-8")
    page = HERE / "index.html"
    page.write_text(template.replace("__PAYLOAD__", out.read_text(encoding="utf-8")), encoding="utf-8")
    print("wrote", page)


main()
