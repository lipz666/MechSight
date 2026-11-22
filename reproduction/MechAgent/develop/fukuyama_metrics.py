"""Granularity- and resonance-tolerant metrics for state-only mechanism datasets.

FukuyamaBench records the states between elementary steps, not arrows. The
original scorer compares per-step bond, bond-order and charge differences, so
two kinds of disagreement that are not chemical errors fail a path:

* Granularity. Splitting one annotated step into two adds the charges of the
  extra intermediate to the change multiset; the same events then fail L1.
* Resonance. Where the charge or the pi bond of a delocalized intermediate is
  drawn is a representation choice; the state difference records it.

Primary metrics here:

``sigma_L1``
    The multiset of sigma-bond formations and cleavages over the whole path
    (hydrogen labels anonymized) equals the annotation's. Bond orders and
    charges are ignored, so it is resonance-invariant, and a multiset over
    the path is granularity-invariant.
``sigma_L2``
    ``sigma_L1`` plus causal order: every predicted step's sigma events can
    be assigned to annotated steps whose index never decreases along the
    predicted path. A finer path passes; a reordered one does not.
``intermediates_in_order``
    Every annotated intermediate, and the final state, appears among the
    predicted states in order (a finer path may visit extra states). States
    are compared per molecule of the annotated state, up to resonance: same
    atoms, same sigma connectivity, same hydrogen count on every heavy atom,
    same net charge. Two Lewis structures that agree on all of that differ
    only in pi/lone-pair placement, which is what resonance is.

``intermediates_in_order_strict`` additionally requires identical bond orders
and heavy-atom formal charges; it is reported only to size the resonance
effect. None of these replaces expert review: they decide agreement with one
annotated path, not chemical validity.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import standalone_chemistry as chem

Step = Dict[str, list]


def is_h(label: str) -> bool:
    return len(label) > 1 and label[0] == "H" and label[1:].isdigit()


def _anon(label: str) -> str:
    return "H*" if is_h(label) else label


def sigma_key(kind: str, a: str, b: str) -> Tuple[str, str, str]:
    return (kind,) + tuple(sorted((_anon(a), _anon(b))))


def sigma_events(step: Step) -> Counter:
    events: Counter = Counter()
    for a, b, *_ in step["form"]:
        events[sigma_key("form", a, b)] += 1
    for a, b, *_ in step["break"]:
        events[sigma_key("break", a, b)] += 1
    return events


def sigma_scores(predicted: Sequence[Step], gold: Sequence[Step]) -> Dict[str, Any]:
    pred = [sigma_events(s) for s in predicted]
    gt = [sigma_events(s) for s in gold]
    flat_pred, flat_gt = sum(pred, Counter()), sum(gt, Counter())
    l1 = flat_pred == flat_gt
    # Earliest-available assignment keeps the bound as low as possible, so it
    # finds a non-decreasing assignment whenever one exists for a path whose
    # multiset already matches.
    l2 = l1
    if l1:
        pool = defaultdict(list)
        for index, step in enumerate(gt):
            for key, count in step.items():
                pool[key].extend([index] * count)
        bound = 0
        for step in pred:
            assigned = []
            for key, count in step.items():
                for _ in range(count):
                    choices = [i for i in pool[key] if i >= bound]
                    if not choices:
                        l2 = False
                        break
                    pool[key].remove(min(choices))
                    assigned.append(min(choices))
                if not l2:
                    break
            if not l2:
                break
            bound = max(assigned, default=bound)
    matched = sum((flat_pred & flat_gt).values())
    return {"sigma_L1": l1, "sigma_L2": l2, "sigma_matched": matched,
            "sigma_predicted": sum(flat_pred.values()), "sigma_gold": sum(flat_gt.values()),
            "gold_transient": sorted(_transient(gt)), "predicted_transient": sorted(_transient(pred))}


def _transient(steps: Iterable[Counter]) -> set:
    seen = defaultdict(set)
    for step in steps:
        for kind, a, b in step:
            seen[(a, b)].add(kind)
    return {pair for pair, kinds in seen.items() if kinds == {"form", "break"}}


# ---------------------------------------------------------------------------
# State comparison
# ---------------------------------------------------------------------------

def state_view(state: Any) -> Dict[str, Any]:
    """Heavy-atom graph with total H per heavy atom and molecules as components."""
    atoms = dict(chem.pgl.iter_atoms(state))
    bonds = {}
    for (a, b), bond in chem.pgl.bond_map(state).items():
        bonds[frozenset((a, b))] = float(bond.order or chem._kind_order(bond.kind) or 0.0)
    heavy = {n for n, a in atoms.items() if a.element != "H"}
    hcount = {n: int(atoms[n].Hcount or 0) for n in heavy}
    hcharge = defaultdict(int)
    adjacency = defaultdict(set)
    for pair in bonds:
        a, b = tuple(pair)
        if a in heavy and b in heavy:
            adjacency[a].add(b)
            adjacency[b].add(a)
        elif a in heavy or b in heavy:
            owner, proton = (a, b) if a in heavy else (b, a)
            hcount[owner] += 1
            hcharge[owner] += int(atoms[proton].charge or 0)
    component, seen = {}, set()
    for start in sorted(heavy):
        if start in seen:
            continue
        members, stack = set(), [start]
        while stack:
            n = stack.pop()
            if n not in members:
                members.add(n)
                stack.extend(adjacency[n] - members)
        seen |= members
        frozen = frozenset(members)
        for n in members:
            component[n] = frozen
    return {"atoms": {n: (atoms[n].element, int(atoms[n].charge or 0)) for n in heavy},
            "hcount": hcount, "hcharge": dict(hcharge), "adjacency": adjacency,
            "orders": {p: o for p, o in bonds.items() if p <= heavy}, "component": component}


def _components(view: Dict[str, Any]) -> List[frozenset]:
    return sorted(set(view["component"].values()), key=lambda c: sorted(c))


def same_state(predicted: Dict[str, Any], gold: Dict[str, Any], *, strict: bool = False) -> bool:
    """Does every molecule of the annotated state appear unchanged in the prediction?

    Molecules present only in the prediction (unreacted reagents the annotation
    introduces later, byproducts it drops) are ignored.
    """
    for molecule in _components(gold):
        if not molecule <= set(predicted["atoms"]) or predicted["component"][next(iter(molecule))] != molecule:
            return False
        for n in molecule:
            if predicted["atoms"][n][0] != gold["atoms"][n][0] or predicted["hcount"][n] != gold["hcount"][n]:
                return False
            if predicted["adjacency"][n] != gold["adjacency"][n]:
                return False
        if strict:
            if any(predicted["atoms"][n][1] != gold["atoms"][n][1] for n in molecule):
                return False
            if any(predicted["orders"].get(p) != o for p, o in gold["orders"].items() if p <= molecule):
                return False
        else:
            def net(view):
                return sum(view["atoms"][n][1] + view["hcharge"].get(n, 0) for n in molecule)
            if net(predicted) != net(gold):
                return False
    return True


def intermediates_in_order(predicted_states: Sequence[Any], gold_states: Sequence[Any],
                           *, strict: bool = False) -> Tuple[bool, int]:
    """(all annotated states after the initial one hit in order, how many were hit)."""
    visited = [state_view(s) for s in predicted_states]
    hit, cursor = 0, 0
    for gold in gold_states[1:]:
        target = state_view(gold)
        found = next((i for i in range(cursor, len(visited)) if same_state(visited[i], target, strict=strict)), None)
        if found is None:
            break
        hit, cursor = hit + 1, found + 1
    return hit == len(gold_states) - 1, hit


def step_from_states(before: Any, after: Any) -> Step:
    """Sigma-bond formations and cleavages between two states."""
    b = set(chem.pgl.bond_map(before))
    a = set(chem.pgl.bond_map(after))
    return {"form": [tuple(k) for k in a - b], "break": [tuple(k) for k in b - a]}


def gold_steps_from_clean(case: Dict[str, Any]) -> List[Step]:
    return [{"form": [(x, y) for x, y, _ in s["bonds_formed"]], "break": [(x, y) for x, y, _ in s["bonds_broken"]]}
            for s in case["gt_step_changes"]]


def replay(initial_pgl: str, actions_pgl: Sequence[str]) -> Tuple[List[Any], int]:
    """States after each executable action; stops at the first invalid one."""
    state = chem.parse_pgl_literal(initial_pgl)
    states = []
    for text in actions_pgl:
        try:
            action, parsed = chem.extract_action_payload(f"<action>{text}</action>")
            if not parsed.get("valid"):
                break
            state, _ = chem.apply_action(state, action)
        except Exception:
            break
        states.append(state)
    return states, len(states)


def score_case(case: Dict[str, Any], actions_pgl: Sequence[str]) -> Dict[str, Any]:
    initial = chem.parse_pgl_literal(case["initial_state_pgl"])
    states, executed = replay(case["initial_state_pgl"], actions_pgl)
    predicted = [step_from_states(b, a) for b, a in zip([initial] + states[:-1], states)]
    gold = gold_steps_from_clean(case)
    gold_states = [chem.parse_pgl_literal(s) for s in case["states_pgl"]]
    ordered, hits = intermediates_in_order(states, gold_states)
    ordered_strict, hits_strict = intermediates_in_order(states, gold_states, strict=True)
    final = bool(states) and same_state(state_view(states[-1]), state_view(gold_states[-1]))
    return {**sigma_scores(predicted, gold), "intermediates_in_order": ordered, "intermediates_hit": hits,
            "intermediates_in_order_strict": ordered_strict, "intermediates_hit_strict": hits_strict,
            "final_state_match": final, "executed_steps": executed, "proposed_steps": len(actions_pgl),
            "gold_steps": len(gold)}
