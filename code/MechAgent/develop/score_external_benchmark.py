"""Score develop-2 runs with the MechFinder layered metrics, on bond changes.

MechFinder is scored on curved arrows (``metrics.compare_path``). The converted
RMR datasets record only the states between steps, so their arrows are unknown.
Both sides are therefore reduced to what consecutive states determine: each
step's bond formations, cleavages, bond-order changes and formal-charge changes,
with explicit hydrogens anonymized. The changes go through the unchanged
``metrics.compare_path`` and ``metrics.aggregate``, so L1/L2/L3, both F1 scores,
the failure diagnosis and the reaction-clustered CI keep their definitions.

Two readings are reported. ``raw`` compares every change. ``normalized`` (the
headline) first removes, on both sides, what a state diff records but no arrow
decides:

* a Kekule redraw: bond-order changes within one step whose connected bonds
  leave every atom's total bond order unchanged;
* a formal charge on an explicit hydrogen: the runtime neutralizes a proton once
  it bonds, while annotations keep either convention;
* an implicit-hydrogen count, which the environment never edits.

The reading is validated on MechFinder, where arrows exist: a path whose arrows
match the annotation should also match on normalized changes. Gold changes there
come from the annotated intermediate states, not from replaying the annotated
arrows, whose shorthand endpoints (``target='H301'`` for a new C-H bond) a
literal replay misreads.

Usage (from the MechAgent root, with a python that has RDKit):
  python -m develop.score_external_benchmark OUT.json \\
    --clean NAME CLEAN_JSONL ROLLOUTS_JSONL [MECHANISMS_DIR] \\
    --mechfinder NAME CORE_DIR ROLLOUTS_JSONL
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import metrics
from . import score_fukuyama as sf
from .standalone_chemistry import (
    ReactionState,
    apply_action,
    extract_action_payload,
    load_core_mechanism_tasks,
    parse_pgl_literal,
    state_signature,
)

# Stop reasons meaning the gateway failed before the agent could act.
INFRA_STOPS = {"planner_create_plan_error", "planner_directive_error", "planner_directive_after_replan_error"}
JUDGMENTS = (
    "planner_global_mechanism_judgment",
    "planner_elementary_steps_judgment",
    "predicted_events_judgment",
)
KIND_ORDER = {"single": 1.0, "double": 2.0, "triple": 3.0, "aromatic": 1.5, "transition": 0.5}


def _is_h(label: str) -> bool:
    return len(label) > 1 and label[0] == "H" and label[1:].isdigit()


def _step_from_signatures(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, list]:
    """One step's changes with the bond orders a normalization needs."""

    b_bonds, a_bonds = before.get("bonds", {}), after.get("bonds", {})
    b_atoms, a_atoms = before.get("atoms", {}), after.get("atoms", {})
    return {
        "form": [tuple(key.split("|")) for key in set(a_bonds) - set(b_bonds)],
        "break": [tuple(key.split("|")) for key in set(b_bonds) - set(a_bonds)],
        "order": [
            (*key.split("|"), float(b_bonds[key].get("order") or 0), float(a_bonds[key].get("order") or 0))
            for key in set(a_bonds) & set(b_bonds)
            if b_bonds[key].get("order") != a_bonds[key].get("order")
        ],
        "charge": [
            name for name in set(a_atoms) & set(b_atoms)
            if b_atoms[name].get("charge") != a_atoms[name].get("charge")
        ],
        "hcount": [
            name for name in set(a_atoms) & set(b_atoms)
            if b_atoms[name].get("Hcount") != a_atoms[name].get("Hcount")
        ],
    }


def _step_from_clean(step: Dict[str, list]) -> Dict[str, list]:
    return {
        "form": [(a, b) for a, b, _ in step["bonds_formed"]],
        "break": [(a, b) for a, b, _ in step["bonds_broken"]],
        "order": [(a, b, KIND_ORDER.get(x, 0.0), KIND_ORDER.get(y, 0.0)) for a, b, x, y in step["bond_order_changes"]],
        "charge": [name for name, _, _ in step["charge_changes"]],
        "hcount": [],
    }


def _kekule_bonds(orders: List[Tuple[str, str, float, float]]) -> set:
    """Bonds in order-change components where no atom's total bond order moves."""

    adjacency, delta = defaultdict(set), defaultdict(float)
    for a, b, x, y in orders:
        adjacency[a].add(b)
        adjacency[b].add(a)
        delta[a] += y - x
        delta[b] += y - x
    redraw, seen = set(), set()
    for start in adjacency:
        if start in seen:
            continue
        component, stack = set(), [start]
        while stack:
            node = stack.pop()
            if node not in component:
                component.add(node)
                stack.extend(adjacency[node])
        seen |= component
        if all(abs(delta[node]) < 1e-9 for node in component):
            redraw |= component
    return {(a, b) for a, b, *_ in orders if a in redraw and b in redraw}


def to_counter(step: Dict[str, list], *, normalized: bool) -> Counter:
    changes: Counter = Counter()
    for a, b in step["form"]:
        changes[sf._bond_key("form", a, b)] += 1
    for a, b in step["break"]:
        changes[sf._bond_key("break", a, b)] += 1
    redraw = _kekule_bonds(step["order"]) if normalized else set()
    for a, b, *_ in step["order"]:
        if (a, b) not in redraw:
            changes[sf._bond_key("order", a, b)] += 1
    for name in step["charge"]:
        if not (normalized and _is_h(name)):
            changes[("charge", sf._anon(name))] += 1
    if not normalized:
        for name in step["hcount"]:
            changes[("hcount", sf._anon(name))] += 1
    return changes


def predicted_step_records(rollout: Dict[str, Any]) -> List[Dict[str, list]]:
    state = parse_pgl_literal(rollout["reaction"]["initial_state_pgl"])
    steps = []
    for record in rollout.get("actions", []):
        if not record.get("accepted_into_history"):
            continue
        action, _ = extract_action_payload(f"<action>{record['action_pgl']}</action>")
        before = state_signature(state)
        state, _ = apply_action(state, action)
        steps.append(_step_from_signatures(before, state_signature(state)))
    return steps


def as_actions(steps: List[Counter], *, bonds_only: bool = False) -> List[Dict[str, Any]]:
    """Encode each change as a pseudo transfer so metrics.compare_path can key it."""

    actions = []
    for step in steps:
        transfers = []
        for key, count in sorted(step.items()):
            if bonds_only and key[0] == "charge":
                continue
            transfers.extend({"source": key[0], "target": tuple(key[1:])} for _ in range(count))
        actions.append({"electron_transfers": transfers})
    return actions


def _judgments(rollout: Dict[str, Any]) -> Optional[Dict[str, str]]:
    assessment = (rollout.get("gt") or {}).get("evaluator_assessment") or {}
    if not all(key in assessment for key in JUDGMENTS):
        return None
    return {key: assessment[key] for key in JUDGMENTS}


def score_paths(
    rollouts: Path,
    gold_by_case: Dict[str, List[Dict[str, list]]],
    mechanisms: Optional[Path],
    partitions: Dict[str, str],
    max_repeat: Optional[int] = None,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for line in rollouts.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rollout = json.loads(line)
        if max_repeat is not None and int(rollout["reaction"].get("repeat_index") or 0) > max_repeat:
            continue
        reaction_id = str(rollout["reaction"]["reaction_id"])
        case_id = reaction_id.split("::")[-1] if reaction_id.startswith("fukuyama::") else reaction_id
        gold = gold_by_case.get(case_id)
        if gold is None:
            continue
        predicted = predicted_step_records(rollout)
        repeat = int(rollout["reaction"].get("repeat_index") or 0)
        row: Dict[str, Any] = {
            "case_id": case_id,
            "partition": partitions.get(case_id),
            "stop": rollout["reaction"].get("stop_reason"),
            "judgments": _judgments(rollout),
            "skill": bool(((rollout.get("planner_plan") or {}).get("planner_skill_selection") or {}).get("selected_skills")),
            "paper": None,
            "arrow_l1": None,
        }
        row["infra"] = row["stop"] in INFRA_STOPS
        for reading in ("raw", "normalized"):
            p = [to_counter(step, normalized=reading == "normalized") for step in predicted]
            g = [to_counter(step, normalized=reading == "normalized") for step in gold]
            row[reading] = metrics.compare_path(as_actions(p), as_actions(g), reaction_id=case_id, repeat_index=repeat)
            row[f"{reading}_bonds_only"] = metrics.compare_path(
                as_actions(p, bonds_only=True), as_actions(g, bonds_only=True), reaction_id=case_id, repeat_index=repeat
            ).transfer_set_match
        if mechanisms is not None:
            # RDKit is needed only for the checkpoint metric.
            from . import score_fukuyama_paper as sp

            checkpoints = sp.load_checkpoints(mechanisms / case_id)
            if checkpoints:
                row["paper"] = sp.checkpoint_exact_match(sp.predicted_states(rollout, scope="touched"), checkpoints)
        gold_arrows = (rollout.get("gt") or {}).get("actions_pgl") or []
        if gold_arrows:
            row["arrow_l1"] = metrics.compare_path(
                [parse_pgl_literal(a["action_pgl"]).to_dict() for a in rollout["actions"] if a.get("accepted_into_history")],
                [parse_pgl_literal(s).to_dict() for s in gold_arrows],
            ).transfer_set_match
        rows.append(row)

    def rate(values: List[float], clusters: List[str]) -> Dict[str, Any]:
        low, high = metrics.clustered_bootstrap_ci(values, clusters)
        return {"count": int(sum(values)), "rate": sum(values) / len(values) if values else 0.0, "ci95": [low, high]}

    def summarize(selected: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid = [row for row in selected if not row["infra"]]
        if not valid:
            return {"num_paths": 0}
        clusters = [row["case_id"] for row in valid]
        out: Dict[str, Any] = {"runs_excluded_gateway": len(selected) - len(valid)}
        for reading in ("normalized", "raw"):
            report = metrics.aggregate([row[reading] for row in valid])
            report["levels"]["L1_bonds_only"] = rate([1.0 if row[f"{reading}_bonds_only"] else 0.0 for row in valid], clusters)
            report["L1_consistency_over_repeats"] = dict(Counter(
                "all" if all(x) else "none" if not any(x) else "some"
                for x in ([row[reading].transfer_set_match for row in valid if row["case_id"] == case] for case in sorted(set(clusters)))
            ))
            out[reading] = report
        out["skill_injected_paths"] = sum(row["skill"] for row in valid)
        out["stop_reasons"] = dict(Counter(row["stop"] for row in selected).most_common())
        with_paper = [row for row in valid if row["paper"] is not None]
        if with_paper:
            out["paper_checkpoint_exact"] = rate(
                [1.0 if row["paper"]["exact_match"] else 0.0 for row in with_paper], [row["case_id"] for row in with_paper]
            )
        judged = [row for row in valid if row["judgments"] is not None]
        if judged:
            out["llm_judgments"] = {"judged_paths": len(judged)}
            for key in JUDGMENTS:
                values = [1.0 if row["judgments"][key] == "correct" else 0.0 for row in judged]
                out["llm_judgments"][key] = {
                    **rate(values, [row["case_id"] for row in judged]),
                    "distribution": dict(Counter(row["judgments"][key] for row in judged)),
                }
        with_arrows = [row for row in valid if row["arrow_l1"] is not None]
        if with_arrows:
            out["validation_against_arrows"] = {
                reading: dict(Counter(
                    f"arrow_{'match' if row['arrow_l1'] else 'miss'}__change_{'match' if row[reading].transfer_set_match else 'miss'}"
                    for row in with_arrows
                ))
                for reading in ("raw", "normalized")
            }
        return out

    result = {"overall": summarize(rows)}
    names = sorted({row["partition"] for row in rows if row["partition"]})
    if len(names) > 1:
        result["by_partition"] = {part: summarize([row for row in rows if row["partition"] == part]) for part in names}
    result["gateway_excluded_runs"] = sorted(f"{row['case_id']}#r{row['normalized'].repeat_index}" for row in rows if row["infra"])
    return result


def clean_gold(path: Path) -> Tuple[Dict[str, List[Dict[str, list]]], Dict[str, str]]:
    gold, partitions = {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            case = json.loads(line)
            gold[str(case["case_id"])] = [_step_from_clean(step) for step in case["gt_step_changes"]]
            partitions[str(case["case_id"])] = str(case.get("partition") or "")
    return gold, partitions


def mechfinder_gold(core_dir: Path, rollouts: Path) -> Tuple[Dict[str, List[Dict[str, list]]], Dict[str, Any]]:
    wanted = {
        str(json.loads(line)["reaction"]["reaction_id"])
        for line in rollouts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    tasks = load_core_mechanism_tasks(
        core_dir=str(core_dir), subset="mechfinder", split="all", max_tasks=None,
        mechfinder_num_classes=len({item.split("::")[1] for item in wanted}), mechfinder_per_class=1,
    )
    gold, audit = {}, {"states_vs_transformations_mismatch": []}
    for task in tasks:
        if task.mechanism_id not in wanted:
            continue
        states = [item for item in task.path.sequence if isinstance(item, ReactionState)]
        if len(states) != len(task.path.sequence) - len(states) + 1:
            audit["states_vs_transformations_mismatch"].append(task.mechanism_id)
        gold[task.mechanism_id] = [
            _step_from_signatures(state_signature(a), state_signature(b)) for a, b in zip(states, states[1:])
        ]
    audit.update(cases_with_gold=len(gold), cases_requested=len(wanted))
    return gold, audit


def render(name: str, report: Dict[str, Any]) -> str:
    overall = report["overall"]
    if overall.get("num_paths") == 0:
        return f"== {name}: no scorable paths"
    lines = [f"== {name}  (normalized bond changes)", metrics.render_report(overall["normalized"])]
    b = overall["normalized"]["levels"]["L1_bonds_only"]
    lines.append(f"  L1 bonds only          {b['rate']:.1%}  [{b['ci95'][0]:.1%}, {b['ci95'][1]:.1%}]")
    raw = overall["raw"]["levels"]["L1_transfer_set"]
    lines.append(f"  L1 raw (no normalization) {raw['rate']:.1%}  [{raw['ci95'][0]:.1%}, {raw['ci95'][1]:.1%}]")
    if "paper_checkpoint_exact" in overall:
        p = overall["paper_checkpoint_exact"]
        lines.append(f"  paper checkpoint exact {p['rate']:.1%}  [{p['ci95'][0]:.1%}, {p['ci95'][1]:.1%}]")
    for key, value in (overall.get("llm_judgments") or {}).items():
        if key != "judged_paths":
            lines.append(f"  {key} {value['rate']:.1%}  [{value['ci95'][0]:.1%}, {value['ci95'][1]:.1%}]  {value['distribution']}")
    if "validation_against_arrows" in overall:
        lines.append(f"  validation against arrows {overall['validation_against_arrows']}")
    lines.append(f"  stop reasons {overall['stop_reasons']}  | gateway-excluded {overall['runs_excluded_gateway']}")
    return "\n".join(lines)


def main() -> None:
    out = Path(sys.argv[1])
    groups: List[List[str]] = []
    max_repeat: Optional[int] = None
    tokens = iter(sys.argv[2:])
    for token in tokens:
        if token == "--max-repeat":
            # Score only the first N repeats of a run launched with more.
            max_repeat = int(next(tokens))
        elif token in {"--clean", "--mechfinder"}:
            groups.append([token])
        else:
            groups[-1].append(token)
    report: Dict[str, Any] = {}
    for group in groups:
        kind, name, source, rollouts = group[0], group[1], Path(group[2]), Path(group[3])
        if kind == "--clean":
            gold, partitions = clean_gold(source)
            report[name] = score_paths(
                rollouts, gold, Path(group[4]) if len(group) > 4 else None, partitions, max_repeat
            )
        else:
            gold, audit = mechfinder_gold(source, rollouts)
            report[name] = score_paths(rollouts, gold, None, {}, max_repeat)
            report[name]["gold_audit"] = audit
        print(render(name, report[name]), flush=True)

    def plain(value: Any) -> Any:
        if isinstance(value, metrics.PathComparison):
            return value.to_dict()
        raise TypeError(type(value))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=plain), encoding="utf-8")


if __name__ == "__main__":
    main()
