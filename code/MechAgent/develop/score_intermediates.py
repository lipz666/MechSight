"""Programmatic intermediate-by-intermediate judgment of an executed mechanism.

The LLM columns judge a text plan and an arrow list against the annotation.
This module judges the executed path structurally: it replays the accepted
actions to get the model's intermediate states, gets the annotated
intermediate states, and matches the two sequences in order.

Two states are the same intermediate when every heavy atom present in both has
the same element, formal charge, total hydrogen count (recorded Hcount plus
bonded explicit H), set of shared heavy-atom neighbours, and bond-order sum
over those neighbours. Exactly three differences are therefore invisible:

* which explicit H label sits on an atom (protons are indistinguishable);
* a Kekule redraw, which moves individual bond orders but no atom's sum;
* a species that one side's annotation boundary omits.

A proton, a formal charge or a pi bond on a different atom, or different
connectivity, is a different intermediate. So is a different resonance
contributor: the program does not decide resonance equivalence.

Annotated intermediates. For the converted RMR datasets, each step's recorded
changes are applied cumulatively to the initial state, so both sides share one
atom-label space; the derivation is checked by comparing its last state with
the target. For MechFinder, the annotated path's own states are used; they are
known to disagree with MechFinder's arrows in places (see
doc/rmr_benchmark_report.md, section 3.4), which the validation table exposes.

Usage (from the MechAgent root):
  python -m develop.score_intermediates OUT.json [--max-repeat N] \\
    --clean NAME CLEAN_JSONL ROLLOUTS_JSONL \\
    --mechfinder NAME CORE_DIR ROLLOUTS_JSONL
"""

from __future__ import annotations

import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import metrics
from .standalone_chemistry import (
    ReactionState,
    apply_action,
    extract_action_payload,
    load_core_mechanism_tasks,
    parse_pgl_literal,
    state_signature,
)

Signature = Dict[str, Any]
KIND_ORDER = {"single": 1.0, "double": 2.0, "triple": 3.0, "aromatic": 1.5, "transition": 0.5, "dative": 1.0}
INFRA_STOPS = {"planner_create_plan_error", "planner_directive_error", "planner_directive_after_replan_error"}
JUDGMENTS = (
    ("planner_elementary_steps_judgment", "LLM strict steps"),
    ("predicted_events_judgment", "LLM executed events"),
)


def _heavy(signature: Signature) -> set:
    return {name for name, atom in signature["atoms"].items() if atom["element"] != "H"}


def atom_views(signature: Signature, atoms: set) -> Dict[str, Tuple]:
    """Per heavy atom: (element, charge, total H, order sum, neighbours, has a fractional bond)."""

    elements = {name: atom["element"] for name, atom in signature["atoms"].items()}
    neighbours: Dict[str, set] = defaultdict(set)
    order_sum: Dict[str, float] = defaultdict(float)
    fractional: set = set()
    bonded_h: Counter = Counter()
    for key, bond in signature["bonds"].items():
        first, second = key.split("|")
        order = float(bond.get("order") or 0.0)
        if order <= 0:
            continue
        for atom, other in ((first, second), (second, first)):
            if atom not in atoms:
                continue
            if elements.get(other) == "H":
                bonded_h[atom] += 1
            elif other in atoms:
                neighbours[atom].add(other)
                order_sum[atom] += order
                if abs(order - round(order)) > 1e-9:
                    fractional.add(atom)
    return {
        name: (
            elements[name],
            int(signature["atoms"][name].get("charge") or 0),
            int(signature["atoms"][name].get("Hcount") or 0) + bonded_h[name],
            round(order_sum[name], 2),
            frozenset(neighbours[name]),
            name in fractional,
        )
        for name in atoms
    }


def same_intermediate(first: Signature, second: Signature) -> bool:
    """Compare shared heavy atoms; see the module docstring for what is ignored.

    An atom with a fractional bond order (an ``aromatic`` 1.5 bond) on either
    side is compared without its order sum. PGL keeps a delocalized ring at
    1.5, and an arrow cannot turn the untouched 1.5 bonds of a dearomatized
    ring into the single/double pattern the annotation writes, so the order
    sum there records the representation, not the chemistry. Its charge,
    hydrogens and neighbours are still compared.
    """

    shared = _heavy(first) & _heavy(second)
    if not shared:
        return False
    left, right = atom_views(first, shared), atom_views(second, shared)
    for name in shared:
        a, b = left[name], right[name]
        if a[:3] != b[:3] or a[4] != b[4]:
            return False
        if not (a[5] or b[5]) and a[3] != b[3]:
            return False
    return True


def _bond_slot(bonds: Dict[str, Any], a: str, b: str) -> str:
    forward, backward = f"{a}|{b}", f"{b}|{a}"
    if forward in bonds:
        return forward
    if backward in bonds:
        return backward
    return "|".join(sorted((a, b)))


def gold_states_from_clean(case: Dict[str, Any]) -> List[Signature]:
    state = copy.deepcopy(state_signature(parse_pgl_literal(case["initial_state_pgl"])))
    states: List[Signature] = []
    for step in case["gt_step_changes"]:
        for a, b, kind in step["bonds_formed"]:
            state["bonds"][_bond_slot(state["bonds"], a, b)] = {"kind": kind, "order": KIND_ORDER.get(kind, 1.0)}
        for a, b, _ in step["bonds_broken"]:
            state["bonds"].pop(_bond_slot(state["bonds"], a, b), None)
        for a, b, _, after in step["bond_order_changes"]:
            state["bonds"][_bond_slot(state["bonds"], a, b)] = {"kind": after, "order": KIND_ORDER.get(after, 1.0)}
        for name, _, after in step["charge_changes"]:
            state["atoms"][name]["charge"] = int(after)
        states.append(copy.deepcopy(state))
    return states


def predicted_states(rollout: Dict[str, Any]) -> List[Signature]:
    state = parse_pgl_literal(rollout["reaction"]["initial_state_pgl"])
    states: List[Signature] = []
    for record in rollout.get("actions", []):
        if not record.get("accepted_into_history"):
            continue
        action, _ = extract_action_payload(f"<action>{record['action_pgl']}</action>")
        state, _ = apply_action(state, action)
        states.append(state_signature(state))
    return states


def compare_sequences(predicted: List[Signature], gold: List[Signature]) -> Dict[str, Any]:
    """Match intermediates in order. Gold includes the final state as its last element."""

    m, n = len(predicted), len(gold)
    equal = [[same_intermediate(predicted[i], gold[j]) for j in range(n)] for i in range(m)]
    table = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            table[i][j] = table[i - 1][j - 1] + 1 if equal[i - 1][j - 1] else max(table[i - 1][j], table[i][j - 1])
    common = table[m][n]
    return {
        "predicted_intermediates": m,
        "gold_intermediates": n,
        # Same count, and the k-th predicted state equals the k-th annotated one.
        "strict_sequence": n > 0 and m == n and all(equal[k][k] for k in range(n)),
        # Every annotated intermediate appears, in order; extra predicted ones allowed.
        "ordered_all": n > 0 and common == n,
        "matched_in_order": common,
        "final_match": bool(m and n and equal[m - 1][n - 1]),
    }


def _judgments(rollout: Dict[str, Any]) -> Optional[Dict[str, str]]:
    assessment = (rollout.get("gt") or {}).get("evaluator_assessment") or {}
    if not all(key in assessment for key, _ in JUDGMENTS):
        return None
    return {key: assessment[key] for key, _ in JUDGMENTS}


def score(
    rollouts: Path,
    gold_by_case: Dict[str, List[Signature]],
    partitions: Dict[str, str],
    max_repeat: Optional[int],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for line in rollouts.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rollout = json.loads(line)
        repeat = int(rollout["reaction"].get("repeat_index") or 0)
        if max_repeat is not None and repeat > max_repeat:
            continue
        if rollout["reaction"].get("stop_reason") in INFRA_STOPS:
            continue
        reaction_id = str(rollout["reaction"]["reaction_id"])
        case_id = reaction_id.split("::")[-1] if reaction_id.startswith("fukuyama::") else reaction_id
        gold = gold_by_case.get(case_id)
        if gold is None:
            continue
        row = {"case_id": case_id, "repeat": repeat, "partition": partitions.get(case_id), **compare_sequences(predicted_states(rollout), gold)}
        row["judgments"] = _judgments(rollout)
        gold_arrows = (rollout.get("gt") or {}).get("actions_pgl") or []
        if gold_arrows:
            arrows = metrics.compare_path(
                [parse_pgl_literal(a["action_pgl"]).to_dict() for a in rollout["actions"] if a.get("accepted_into_history")],
                [parse_pgl_literal(s).to_dict() for s in gold_arrows],
            )
            row["arrow_l1"], row["arrow_l3"] = arrows.transfer_set_match, arrows.step_partition_match
        rows.append(row)

    def summarize(selected: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not selected:
            return {"paths": 0}
        clusters = [row["case_id"] for row in selected]

        def rate(values: List[float]) -> Dict[str, Any]:
            low, high = metrics.clustered_bootstrap_ci(values, clusters)
            return {"rate": sum(values) / len(values), "ci95": [low, high], "count": int(sum(values))}

        gold_total = sum(row["gold_intermediates"] for row in selected)
        predicted_total = sum(row["predicted_intermediates"] for row in selected)
        matched_total = sum(row["matched_in_order"] for row in selected)
        out: Dict[str, Any] = {
            "paths": len(selected),
            "cases": len(set(clusters)),
            "strict_sequence": rate([float(row["strict_sequence"]) for row in selected]),
            "ordered_all": rate([float(row["ordered_all"]) for row in selected]),
            "final_match": rate([float(row["final_match"]) for row in selected]),
            "intermediate_recall": matched_total / gold_total if gold_total else 0.0,
            "intermediate_precision": matched_total / predicted_total if predicted_total else 0.0,
        }
        judged = [row for row in selected if row["judgments"]]
        if judged:
            for key, label in JUDGMENTS:
                out[label] = sum(row["judgments"][key] == "correct" for row in judged) / len(judged)
            out["crosstab_strict_sequence_vs_llm_events"] = dict(Counter(
                f"program_{'pass' if row['strict_sequence'] else 'fail'}__llm_{row['judgments']['predicted_events_judgment']}"
                for row in judged
            ))
            out["crosstab_ordered_all_vs_llm_steps"] = dict(Counter(
                f"program_{'pass' if row['ordered_all'] else 'fail'}__llm_{row['judgments']['planner_elementary_steps_judgment']}"
                for row in judged
            ))
        with_arrows = [row for row in selected if "arrow_l3" in row]
        if with_arrows:
            out["validation_against_arrow_L3"] = dict(Counter(
                f"arrow_L3_{'pass' if row['arrow_l3'] else 'fail'}__program_strict_{'pass' if row['strict_sequence'] else 'fail'}"
                for row in with_arrows
            ))
        return out

    report = {"overall": summarize(rows)}
    names = sorted({row["partition"] for row in rows if row["partition"]})
    if len(names) > 1:
        report["by_partition"] = {part: summarize([row for row in rows if row["partition"] == part]) for part in names}
    report["rows"] = [{k: v for k, v in row.items() if k != "judgments"} for row in rows]
    return report


def clean_gold(path: Path) -> Tuple[Dict[str, List[Signature]], Dict[str, str], Dict[str, Any]]:
    gold, partitions, derived_final_matches, cases = {}, {}, 0, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        states = gold_states_from_clean(case)
        gold[str(case["case_id"])] = states
        partitions[str(case["case_id"])] = str(case.get("partition") or "")
        target = state_signature(parse_pgl_literal(case["target_state_pgl"]))
        derived_final_matches += bool(states and same_intermediate(states[-1], target))
        cases += 1
    return gold, partitions, {"cases": cases, "derived_last_state_equals_target": derived_final_matches}


def mechfinder_gold(core_dir: Path, rollouts: Path) -> Tuple[Dict[str, List[Signature]], Dict[str, Any]]:
    records = [json.loads(line) for line in rollouts.read_text(encoding="utf-8").splitlines() if line.strip()]
    wanted = {str(r["reaction"]["reaction_id"]): r["reaction"]["initial_state_pgl"] for r in records}
    tasks = load_core_mechanism_tasks(
        core_dir=str(core_dir), subset="mechfinder", split="all", max_tasks=None,
        mechfinder_num_classes=len({item.split("::")[1] for item in wanted}), mechfinder_per_class=1,
    )
    gold, initial_agrees = {}, 0
    for task in tasks:
        if task.mechanism_id not in wanted:
            continue
        signatures = [state_signature(item) for item in task.path.sequence if isinstance(item, ReactionState)]
        gold[task.mechanism_id] = signatures[1:]
        initial_agrees += same_intermediate(signatures[0], state_signature(parse_pgl_literal(wanted[task.mechanism_id])))
    return gold, {"cases": len(gold), "annotated_first_state_equals_task_initial": initial_agrees}


def main() -> None:
    out = Path(sys.argv[1])
    max_repeat: Optional[int] = None
    groups: List[List[str]] = []
    tokens = iter(sys.argv[2:])
    for token in tokens:
        if token == "--max-repeat":
            max_repeat = int(next(tokens))
        elif token in {"--clean", "--mechfinder"}:
            groups.append([token])
        else:
            groups[-1].append(token)
    report: Dict[str, Any] = {}
    for kind, name, source, rollouts in groups:
        if kind == "--clean":
            gold, partitions, audit = clean_gold(Path(source))
        else:
            gold, audit = mechfinder_gold(Path(source), Path(rollouts))
            partitions = {}
        report[name] = score(Path(rollouts), gold, partitions, max_repeat)
        report[name]["gold_audit"] = audit
        o = report[name]["overall"]
        print(f"== {name}  paths={o['paths']} cases={o.get('cases')}  gold audit {audit}")
        for key in ("strict_sequence", "ordered_all", "final_match"):
            print(f"  {key:16s} {o[key]['rate']:.1%}  [{o[key]['ci95'][0]:.1%}, {o[key]['ci95'][1]:.1%}]")
        print(f"  intermediate recall {o['intermediate_recall']:.1%}  precision {o['intermediate_precision']:.1%}")
        for key in ("LLM strict steps", "LLM executed events"):
            if key in o:
                print(f"  {key} {o[key]:.1%}")
        for key in ("crosstab_strict_sequence_vs_llm_events", "crosstab_ordered_all_vs_llm_steps", "validation_against_arrow_L3"):
            if key in o:
                print(f"  {key} {o[key]}")
        for part, summary in (report[name].get("by_partition") or {}).items():
            print(f"  [{part}] strict {summary['strict_sequence']['rate']:.1%}  ordered {summary['ordered_all']['rate']:.1%}  final {summary['final_match']['rate']:.1%}  recall {summary['intermediate_recall']:.1%}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
