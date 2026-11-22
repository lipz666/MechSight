"""P1 pre-experiment: can MechFinder be solved by class + template instantiation?

No LLM is called. A "template" is the annotated arrow sequence of another
MechFinder record (an exemplar). It is instantiated on a test case by mapping
the exemplar's arrow atoms onto the test initial state (element, charge and
pairwise bond order must agree) and replaying the mapped arrows. A mapping is
accepted only if the replay reaches the test target on heavy atoms, the same
endpoint information every other arm receives.

Arms, all scored on the frozen exp1 cases with the exp1 arrow metrics:
  oracle_class  exemplars from the GT reaction class (diagnostic upper bound)
  retrieval     no class given; exemplars from every class whose net
                bond-making/breaking signature equals the test endpoints'
                (LLM-free, deployable baseline)
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments' / 'exp1_core_comparison'))
from develop import metrics
from develop import standalone_chemistry as chem
from develop.score_intermediates import atom_views, same_intermediate
from prepare import source_key
from summarize import heavy_signature, translate

EXP1 = ROOT / 'outputs' / 'publication_exp1_20261018'
CORPUS = ROOT.parent / 'core' / 'mechfinder_merged_pgl_paths.core.jsonl'
SEED = 20261024


def atoms_bonds(state):
    atoms = dict(chem.pgl.iter_atoms(state))
    orders = {}
    for (a, b), bond in chem.pgl.bond_map(state).items():
        orders[frozenset((a, b))] = float(bond.order or chem._kind_order(bond.kind) or 0)
    return atoms, orders


def net_signature(initial, final):
    """Bonds made/broken and charge changes on heavy atoms, by element.

    Order changes of bonds that persist are ignored, so a Kekule redrawing
    of a ring cannot hide a matching exemplar.
    """
    ia, ib = atoms_bonds(initial)
    fa, fb = atoms_bonds(final)
    heavy = {n for n, a in ia.items() if a.element != 'H'}
    kept = heavy & set(fa)
    events = collections.Counter()
    for pair in set(ib) | set(fb):
        a, b = sorted(pair)
        if a not in heavy or b not in heavy or not ({a, b} & kept):
            continue
        before = ib.get(pair, 0) > 0
        after = a in kept and b in kept and fb.get(pair, 0) > 0
        if before != after:
            events[('form' if after else 'break',) + tuple(sorted((ia[a].element, ia[b].element)))] += 1
    for n in kept:
        if int(ia[n].charge or 0) != int(fa[n].charge or 0):
            events[('charge', ia[n].element, int(ia[n].charge or 0), int(fa[n].charge or 0))] += 1
    return tuple(sorted(events.items()))


def endpoint_labels(endpoint):
    if isinstance(endpoint, str):
        return [endpoint]
    return [x for x in endpoint[:2]]


def remap_action(action, mapping):
    action = copy.deepcopy(action)
    for t in action.electron_transfers:
        for side in ('source', 'target'):
            v = getattr(t, side)
            if isinstance(v, str):
                setattr(t, side, mapping[v])
            else:
                v = list(v)
                v[0] = mapping[v[0]]
                if len(v) >= 2 and v[1] in mapping:
                    v[1] = mapping[v[1]]
                setattr(t, side, tuple(v))
    return action


def load_exemplar(row):
    """Heavy arrow atoms keep their labels along a MechFinder path; protons do
    not (the annotation renames H301 to H900003 between states), so hydrogens
    are resolved per step against the state that precedes that step."""
    path = chem.parse_pgl_literal(row['mechanism_graph'])
    states = [s for s in path.sequence if isinstance(s, chem.ReactionState)]
    actions = [a for a in path.sequence if isinstance(a, chem.ReactionTransformation)]
    initial = states[0]
    atoms, orders = atoms_bonds(initial)
    core, hydrogens = [], []
    for k, a in enumerate(actions):
        step_atoms, step_orders = atoms_bonds(states[k])
        roles = {}
        # Merged MechFinder states can draw a proton on two atoms at once; the
        # bond this step breaks names the atom it is actually leaving.
        broken = {frozenset(endpoint_labels(t.source)) for t in a.electron_transfers if not isinstance(t.source, str)}
        for t in a.electron_transfers:
            for label in endpoint_labels(t.source) + endpoint_labels(t.target):
                if label not in step_atoms:
                    raise ValueError('arrow atom absent from preceding state')
                if step_atoms[label].element == 'H':
                    partners = sorted(m for p in step_orders if label in p for m in p
                                      if m != label and step_atoms[m].element != 'H')
                    leaving = [m for m in partners if frozenset((label, m)) in broken]
                    partners = leaving or partners
                    roles[label] = (partners[0] if partners else None, int(step_atoms[label].charge or 0))
                    label = partners[0] if partners else None
                if label is None:
                    continue
                if label not in atoms:
                    raise ValueError('heavy arrow atom absent from initial state')
                if label not in core:
                    core.append(label)
        hydrogens.append(roles)
    return {'id': f"mechfinder::{row['reaction_class']}::{row['source_row_index']}",
            'reaction_class': row['reaction_class'], 'actions': actions, 'core': core, 'hydrogens': hydrogens,
            'atoms': {n: (atoms[n].element, int(atoms[n].charge or 0)) for n in core},
            'orders': {p: o for p, o in orders.items() if p <= set(core)},
            'neighbours': {n: {m for p in orders for m in p if n in p and m != n} for n in core},
            'signature': net_signature(initial, states[-1])}


def search_order(ex):
    """Heavy atoms, connected ones first (BFS inside the core)."""
    order, seen = [], set()
    for root in ex['core']:
        if root in seen:
            continue
        queue = [root]
        seen.add(root)
        while queue:
            n = queue.pop(0)
            order.append(n)
            for m in sorted(ex['neighbours'][n] & set(ex['core'])):
                if m not in seen and frozenset((n, m)) in ex['orders']:
                    seen.add(m)
                    queue.append(m)
    return order


def mappings(ex, test, limit):
    atoms, orders, neighbours = test['atoms'], test['orders'], test['neighbours']
    order = search_order(ex)
    by_type = collections.defaultdict(list)
    for n, a in atoms.items():
        by_type[(a.element, int(a.charge or 0))].append(n)
    found = 0
    assign, used = {}, set()

    def candidates(n):
        mapped_neighbours = [m for m in ex['neighbours'][n] if m in assign and frozenset((n, m)) in ex['orders']]
        if mapped_neighbours:
            pool = neighbours[assign[mapped_neighbours[0]]]
        else:
            pool = by_type[ex['atoms'][n]]
        return sorted(c for c in pool if c not in used and (atoms[c].element, int(atoms[c].charge or 0)) == ex['atoms'][n])

    def consistent(n, c):
        for m, d in assign.items():
            if ex['orders'].get(frozenset((n, m)), 0) != orders.get(frozenset((c, d)), 0):
                return False
        return True

    def rec(i):
        nonlocal found
        if found >= limit:
            return
        if i == len(order):
            found += 1
            yield dict(assign)
            return
        n = order[i]
        for c in candidates(n):
            if consistent(n, c):
                assign[n] = c
                used.add(c)
                yield from rec(i + 1)
                del assign[n]
                used.discard(c)

    yield from rec(0)


def resolve_step(ex, k, mapping, state):
    """Map step k's hydrogens onto equivalent hydrogens of the current test state."""
    atoms, orders = atoms_bonds(state)
    local, used = dict(mapping), set()
    for h, (partner, charge) in sorted(ex['hydrogens'][k].items()):
        if partner is None:
            pool = [n for n, a in atoms.items() if a.element == 'H' and int(a.charge or 0) == charge
                    and not any(n in p for p in orders)]
        else:
            anchor = mapping[partner]
            pool = [m for p in orders if anchor in p for m in p if m != anchor and atoms[m].element == 'H']
        pool = sorted(n for n in pool if n not in used)
        if not pool:
            raise ValueError('no equivalent hydrogen in test state')
        local[h] = pool[0]
        used.add(pool[0])
    return remap_action(ex['actions'][k], local)


def instantiate_path(ex, mapping, state):
    mapped = []
    for k in range(len(ex['actions'])):
        action = resolve_step(ex, k, mapping, state)
        state, _ = chem.apply_action(state, action.to_dict())
        mapped.append(action)
    return mapped, state


def canonical(actions):
    key = []
    for a in actions:
        for t in a.electron_transfers:
            key.append(repr((t.source, t.target)))
    return hashlib.sha256('|'.join(sorted(key)).encode()).hexdigest()[:16]


def target_distance(final, target_sig):
    """Heavy atoms whose element/charge/neighbour set differ from the target."""
    signature = heavy_signature(final)
    shared = set(signature['atoms']) & set(target_sig['atoms'])
    left, right = atom_views(signature, shared), atom_views(target_sig, shared)
    return sum(left[n][:3] != right[n][:3] or left[n][4] != right[n][4] for n in shared)


def instantiate(ex, test, limit):
    """Return (mapped path, mappings tried, verified, distance).

    A mapping whose replay reaches the target is returned at once. Otherwise
    the executable mapping closest to the target is kept: about a quarter of
    MechFinder paths do not reproduce their own target state in this runtime
    (H renumbering, charges on bonded protons), so exact verification alone
    would reject the annotation's own arrows.
    """
    target_sig = test['target_heavy']
    tried, best = 0, None
    for mapping in mappings(ex, test, limit):
        tried += 1
        try:
            mapped, final = instantiate_path(ex, mapping, test['initial'])
        except Exception:
            continue
        if same_intermediate(heavy_signature(final), target_sig):
            return mapped, tried, True, 0
        distance = target_distance(final, target_sig)
        if best is None or distance < best[1]:
            best = (mapped, distance)
    if best is None:
        return None, tried, False, None
    return best[0], tried, False, best[1]


def prepare_test(row):
    initial = chem.parse_pgl_literal(row['initial_state_pgl'])
    target = chem.parse_pgl_literal(row['target_state_pgl'])
    atoms, orders = atoms_bonds(initial)
    neighbours = collections.defaultdict(set)
    for p in orders:
        a, b = tuple(p)
        neighbours[a].add(b)
        neighbours[b].add(a)
    return {'initial': initial, 'atoms': atoms, 'orders': orders, 'neighbours': neighbours,
            'target_heavy': heavy_signature(target), 'signature': net_signature(initial, target)}


def solve(test, pool, limit):
    wins, near = [], []
    attempts = 0
    for ex in pool:
        mapped, tried, verified, distance = instantiate(ex, test, limit)
        attempts += tried
        if verified:
            wins.append((ex, mapped))
        elif mapped:
            near.append((distance, ex, mapped))
    verified = bool(wins)
    if not wins and near:
        # Unverified fallback: vote among the executable instantiations that
        # end closest to the target.
        closest = min(d for d, _, _ in near)
        wins = [(ex, m) for d, ex, m in near if d == closest]
    if not wins:
        return None, {'verified': False, 'verified_exemplars': 0, 'candidate_exemplars': len(pool), 'mappings_tried': attempts}
    votes = collections.Counter(canonical(m) for _, m in wins)
    best = votes.most_common(1)[0][0]
    chosen = next((ex, m) for ex, m in wins if canonical(m) == best)
    classes = collections.Counter(ex['reaction_class'] for ex, _ in wins)
    return chosen, {'verified': verified, 'voting_exemplars': len(wins), 'candidate_exemplars': len(pool), 'mappings_tried': attempts,
                    'distinct_templates': len(votes), 'winning_votes': votes[best],
                    'chosen_exemplar': chosen[0]['id'], 'chosen_class': chosen[0]['reaction_class'],
                    'class_votes': dict(classes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'p1_template_20261024')
    ap.add_argument('--per-class', type=int, default=40)
    ap.add_argument('--mapping-limit', type=int, default=400)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    inputs = [json.loads(l) for l in (EXP1 / 'inputs.jsonl').read_text().splitlines()]
    gold = {r['case_id']: r for r in map(json.loads, (EXP1 / 'gold.jsonl').read_text().splitlines())}
    test_ids = {g['source_reaction_id'] for g in gold.values()}
    test_keys = {source_key(g['source_reaction_smiles']) for g in gold.values()}

    # Exemplar pool: held out from every test record and test reaction structure.
    buckets = collections.defaultdict(list)
    with CORPUS.open() as f:
        for line in f:
            row = json.loads(line)
            rid = f"mechfinder::{row['reaction_class']}::{row['source_row_index']}"
            if rid in test_ids:
                continue
            buckets[row['reaction_class']].append((hashlib.sha256(f'{SEED}:{rid}'.encode()).hexdigest(), row))
    pool, skipped = [], collections.Counter()
    t0 = time.time()
    for cls, rows in sorted(buckets.items()):
        taken = 0
        for _, row in sorted(rows, key=lambda x: x[0]):
            if taken >= args.per_class:
                break
            if source_key(row['source_reaction_smiles']) in test_keys:
                skipped['same_structure_as_test'] += 1
                continue
            try:
                pool.append(load_exemplar(row))
                taken += 1
            except Exception as exc:
                skipped['exemplar:' + str(exc)[:60]] += 1
    print(json.dumps({'event': 'pool', 'exemplars': len(pool), 'classes': len(buckets),
                      'skipped': dict(skipped), 'seconds': round(time.time() - t0, 1)}), flush=True)
    by_class = collections.defaultdict(list)
    by_sig = collections.defaultdict(list)
    for ex in pool:
        by_class[ex['reaction_class']].append(ex)
        by_sig[ex['signature']].append(ex)

    rows = {'oracle_class': [], 'retrieval': []}
    comparisons = {'oracle_class': [], 'retrieval': []}
    for row in inputs:
        cid = row['case_id']
        g = gold[cid]
        test = prepare_test(row)
        inv = {v: k for k, v in g['label_map'].items()}
        expected = [chem.parse_pgl_literal(t) for t in g['gt_actions_pgl']]
        for arm, candidates in [('oracle_class', by_class[g['reaction_class']]),
                                ('retrieval', by_sig.get(test['signature'], []))]:
            t1 = time.time()
            chosen, info = solve(test, candidates, args.mapping_limit)
            actions = chosen[1] if chosen else []
            cmp = metrics.compare_path([translate(a, inv) for a in actions], expected, reaction_id=cid, repeat_index=1)
            comparisons[arm].append(cmp)
            rows[arm].append({'case_id': cid, 'arm': arm, 'reaction_class': g['reaction_class'],
                              'covered': chosen is not None, **info,
                              'predicted_class_correct': (info.get('chosen_class') == g['reaction_class']) if chosen else None,
                              'transfer_set_match': cmp.transfer_set_match,
                              'transfer_sequence_match': cmp.transfer_sequence_match,
                              'step_partition_match': cmp.step_partition_match,
                              'boundary_diagnosis': cmp.boundary_diagnosis,
                              'predicted_actions_pgl': [repr(a) for a in actions],
                              'seconds': round(time.time() - t1, 2)})
        print(json.dumps({'case': cid, 'class': g['reaction_class'],
                          **{arm: (rows[arm][-1]['covered'], rows[arm][-1]['transfer_set_match']) for arm in rows}}), flush=True)

    summary = {}
    for arm in rows:
        agg = metrics.aggregate(comparisons[arm], seed=SEED)
        agg.pop('paths', None)
        covered = [r for r in rows[arm] if r['covered']]
        agg['coverage'] = len(covered) / len(rows[arm])
        agg['verified_rate'] = sum(bool(r.get('verified')) for r in rows[arm]) / len(rows[arm])
        verified = [r for r in rows[arm] if r.get('verified')]
        agg['L1_when_verified'] = (sum(r['transfer_set_match'] for r in verified) / len(verified)) if verified else None
        agg['class_accuracy_when_covered'] = (sum(bool(r['predicted_class_correct']) for r in covered) / len(covered)) if covered else None
        agg['L1_when_covered'] = (sum(r['transfer_set_match'] for r in covered) / len(covered)) if covered else None
        summary[arm] = agg
    (out / 'rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for arm in rows for r in rows[arm]))
    (out / 'summary.json').write_text(json.dumps({'config': {k: str(v) for k, v in vars(args).items()},
                                                  'pool_size': len(pool), 'arms': summary}, ensure_ascii=False, indent=2))
    print(json.dumps({arm: {'L1': s['levels']['L1_transfer_set']['rate'], 'L3': s['levels']['L3_step_partition']['rate'],
                            'coverage': s['coverage']} for arm, s in summary.items()}))


if __name__ == '__main__':
    main()
