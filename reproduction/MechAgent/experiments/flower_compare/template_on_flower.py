"""Stage 1b: a learning-free template baseline on FlowER's own test set.

FlowER's USPTO mechanisms were imputed with 1,220 expert templates for 252
reaction classes. If its benchmark mostly measures template recovery, then
retrieving a training sequence with the same net change and replaying its
elementary steps on the test molecules should already reproduce most test
pathways.

Per test sequence (reactant state and final state are the only inputs):
  1. candidates = training sequences with the same net signature (bonds made
     and broken, by element pair, plus charge changes, over the whole path);
  2. map the candidate's core atoms (every atom whose bonds or charge change
     anywhere in its path, hydrogens included) onto the test reactants:
     element, charge and every pairwise bond order must agree;
  3. replay the candidate's per-step changes; keep instantiations whose final
     state equals the test final state;
  4. vote among the kept pathways (canonical state sequences).
Score: the predicted sequence of states equals the annotated one exactly
(canonical, map-free, stereo-free), i.e. FlowER's top-1 pathway criterion.
"""
from __future__ import annotations

import argparse
import os
import signal
import collections
import json
import random
import time
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / 'external' / 'FlowER' / 'data' / 'flower_new_dataset'
PS = Chem.SmilesParserParams()
PS.removeHs = False
PS.sanitize = False


def read_sequences(path, wanted=None, limit_lines=None):
    seqs = collections.defaultdict(list)
    with open(path) as f:
        for i, line in enumerate(f):
            if limit_lines and i >= limit_lines:
                break
            rxn, idx = line.rstrip('\n').rsplit('|', 1)
            if wanted is not None and idx not in wanted:
                continue
            seqs[idx].append(rxn)
    return seqs


def graph(smiles):
    """Map number -> (element, charge); frozenset(pair) -> bond order (x2, integer)."""
    mol = Chem.MolFromSmiles(smiles, PS)
    atoms = {a.GetAtomMapNum(): (a.GetSymbol(), a.GetFormalCharge()) for a in mol.GetAtoms()}
    bonds = {}
    for b in mol.GetBonds():
        bonds[frozenset((b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()))] = int(round(b.GetBondTypeAsDouble() * 2))
    return atoms, bonds


def states_of(steps):
    """Ordered states of a sequence; FlowER marks the end with a self-loop step."""
    states = [steps[0].split('>>')[0]]
    for s in steps:
        src, tgt = s.split('>>')
        if tgt != states[-1]:
            states.append(tgt)
    return states


def canonical(atoms, bonds):
    rw = Chem.RWMol()
    index = {}
    for m, (el, ch) in sorted(atoms.items()):
        a = Chem.Atom(el)
        a.SetFormalCharge(ch)
        a.SetNoImplicit(True)
        index[m] = rw.AddAtom(a)
    kinds = {2: Chem.BondType.SINGLE, 4: Chem.BondType.DOUBLE, 6: Chem.BondType.TRIPLE, 3: Chem.BondType.AROMATIC}
    for pair, order in bonds.items():
        x, y = tuple(pair)
        rw.AddBond(index[x], index[y], kinds.get(order, Chem.BondType.SINGLE))
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)


def signature(first, last):
    (a0, b0), (a1, b1) = first, last
    ev = collections.Counter()
    for pair in set(b0) | set(b1):
        x, y = sorted(pair)
        before, after = pair in b0, pair in b1
        if before != after:
            ev[('form' if after else 'break',) + tuple(sorted((a0[x][0], a0[y][0])))] += 1
    for m in a0:
        if a0[m][1] != a1[m][1]:
            ev[('charge', a0[m][0], a0[m][1], a1[m][1])] += 1
    return tuple(sorted(ev.items()))


def template_of(state_graphs):
    """Core atoms and per-step changes restricted to them."""
    core = set()
    for (a0, b0), (a1, b1) in zip(state_graphs, state_graphs[1:]):
        for pair in set(b0) | set(b1):
            if b0.get(pair) != b1.get(pair):
                core |= pair
        core |= {m for m in a0 if a0[m][1] != a1[m][1]}
    a_init, b_init = state_graphs[0]
    steps = []
    for (a0, b0), (a1, b1) in zip(state_graphs, state_graphs[1:]):
        changes = {pair: b1.get(pair, 0) for pair in set(b0) | set(b1) if pair <= core and b0.get(pair) != b1.get(pair)}
        charges = {m: a1[m][1] for m in core if a0[m][1] != a1[m][1]}
        steps.append((changes, charges))
    neighbours = collections.defaultdict(collections.Counter)
    for pair in b_init:
        x, y = tuple(pair)
        if x in core:
            neighbours[x][a_init[y][0]] += 1
        if y in core:
            neighbours[y][a_init[x][0]] += 1
    return {'core': sorted(core), 'atoms': {m: a_init[m] for m in core},
            'bonds': {p: o for p, o in b_init.items() if p <= core}, 'steps': steps,
            'context': {m: neighbours[m] for m in core}}


def context_score(tpl, mapping, atoms, bonds):
    """How well the template's first-shell environment matches the test molecule.

    Template classes are defined by the reaction centre and its surroundings,
    so among templates sharing a net change the one whose core atoms see the
    same neighbouring elements is the better analogue. Uses reactants only.
    """
    neighbours = collections.defaultdict(collections.Counter)
    targets = set(mapping.values())
    for pair in bonds:
        x, y = tuple(pair)
        if x in targets:
            neighbours[x][atoms[y][0]] += 1
        if y in targets:
            neighbours[y][atoms[x][0]] += 1
    shared = total = 0
    for m, c in mapping.items():
        a, b = tpl['context'][m], neighbours[c]
        shared += sum((a & b).values())
        total += max(sum(a.values()), sum(b.values()))
    return shared / total if total else 0.0


def mappings(tpl, atoms, bonds, limit):
    core = tpl['core']
    adj = collections.defaultdict(set)
    for pair in bonds:
        x, y = tuple(pair)
        adj[x].add(y)
        adj[y].add(x)
    tadj = collections.defaultdict(set)
    for pair in tpl['bonds']:
        x, y = tuple(pair)
        tadj[x].add(y)
        tadj[y].add(x)
    # order: connected-first BFS over the template core
    order, seen = [], set()
    for root in sorted(core, key=lambda m: tpl['atoms'][m][0] == 'H'):
        if root in seen:
            continue
        queue = [root]
        seen.add(root)
        while queue:
            n = queue.pop(0)
            order.append(n)
            for m in sorted(tadj[n]):
                if m not in seen:
                    seen.add(m)
                    queue.append(m)
    by_type = collections.defaultdict(list)
    for m, t in atoms.items():
        by_type[t].append(m)
    assign, used, found = {}, set(), [0]

    def rec(i):
        if found[0] >= limit:
            return
        if i == len(order):
            found[0] += 1
            yield dict(assign)
            return
        n = order[i]
        anchors = [assign[m] for m in tadj[n] if m in assign]
        pool = adj[anchors[0]] if anchors else by_type[tpl['atoms'][n]]
        for c in sorted(pool):
            if c in used or atoms[c] != tpl['atoms'][n]:
                continue
            if any(tpl['bonds'].get(frozenset((n, m)), 0) != bonds.get(frozenset((c, d)), 0) for m, d in assign.items()):
                continue
            assign[n] = c
            used.add(c)
            yield from rec(i + 1)
            del assign[n]
            used.discard(c)

    yield from rec(0)


def instantiate(tpl, atoms, bonds, mapping):
    a, b = dict(atoms), dict(bonds)
    states = []
    for changes, charges in tpl['steps']:
        for pair, order in changes.items():
            x, y = tuple(pair)
            key = frozenset((mapping[x], mapping[y]))
            if order:
                b[key] = order
            else:
                b.pop(key, None)
        for m, ch in charges.items():
            a[mapping[m]] = (a[mapping[m]][0], ch)
        states.append((dict(a), dict(b)))
    return states


_G = {}


def with_timeout(fn):
    """Per-case wall-clock cap; a timed-out case counts as a template failure."""
    def run(idx):
        def alarm(signum, frame):
            raise TimeoutError
        signal.signal(signal.SIGALRM, alarm)
        signal.alarm(_G.get('timeout', 300))
        try:
            return fn(idx)
        except TimeoutError:
            return {'seq': idx, 'timeout': True, 'covered': False, 'pathway_exact': False,
                    'tf_pathway_verified': False, 'tf_pathway_unverified': False, 'gold_among_verified': False,
                    'candidates': -1, 'gold_steps': -1, 'distinct_verified_paths': 0}
        finally:
            signal.alarm(0)
    return run


def _solve_capped(idx):
    return with_timeout(solve)(idx)


def _solve_tf_capped(idx):
    return with_timeout(solve_teacher_forced)(idx)


def solve(idx):
    pool, tests, sigs, limit = _G['pool'], _G['tests'], _G['sigs'], _G['limit']
    st = tests[idx]
    atoms, bonds = st[0]
    gold = [canonical(*x) for x in st]
    final = gold[-1]
    votes = collections.Counter()
    scores = {}
    for tpl in pool.get(sigs[idx], []):
        for mapping in mappings(tpl, atoms, bonds, limit):
            try:
                pred = [canonical(*x) for x in instantiate(tpl, atoms, bonds, mapping)]
            except Exception:
                continue
            # drop no-change steps, as in the gold state list
            path = [gold[0]] + [x for k, x in enumerate(pred) if x != ([gold[0]] + pred)[k]]
            if path[-1] == final:
                key = tuple(path)
                votes[key] += 1
                scores[key] = max(scores.get(key, 0.0), context_score(tpl, mapping, atoms, bonds))
    # Best environment match first; votes break ties.
    best = max(votes, key=lambda k: (scores[k], votes[k])) if votes else None
    return {'seq': idx, 'gold_steps': len(gold) - 1, 'candidates': len(pool.get(sigs[idx], [])),
            'covered': best is not None, 'pathway_exact': best == tuple(gold),
            'distinct_verified_paths': len(votes), 'gold_among_verified': tuple(gold) in votes,
            'gold_states': gold, 'best_states': list(best) if best else None}


def solve_teacher_forced(idx):
    """FlowER's top-1 pathway metric is teacher-forced: each step is predicted
    from the GT previous state. Same here: at step k only template paths whose
    first k states equal the GT prefix may vote for state k+1. Two variants:
    'verified' keeps only instantiations reaching the final state (uses the
    product, as FlowER's beam check does); 'unverified' keeps every executable
    instantiation (no product information at all)."""
    pool, tests, sigs, limit = _G['pool'], _G['tests'], _G['sigs'], _G['limit']
    st = tests[idx]
    atoms, bonds = st[0]
    gold = [canonical(*x) for x in st]
    paths = {}
    for tpl in pool.get(sigs[idx], []):
        for mapping in mappings(tpl, atoms, bonds, limit):
            try:
                pred = [canonical(*x) for x in instantiate(tpl, atoms, bonds, mapping)]
            except Exception:
                continue
            path = tuple([gold[0]] + [x for k, x in enumerate(pred) if x != ([gold[0]] + pred)[k]])
            score = context_score(tpl, mapping, atoms, bonds)
            votes, best = paths.get(path, (0, 0.0))
            paths[path] = (votes + 1, max(best, score))
    out = {'seq': idx, 'gold_steps': len(gold) - 1}
    for variant in ('verified', 'unverified'):
        pool_paths = {p: v for p, v in paths.items() if variant == 'unverified' or p[-1] == gold[-1]}
        ok = True
        for k in range(len(gold) - 1):
            options = collections.defaultdict(lambda: (0, 0.0))
            for p, (votes, score) in pool_paths.items():
                if len(p) > k + 1 and p[:k + 1] == tuple(gold[:k + 1]):
                    v, sc = options[p[k + 1]]
                    options[p[k + 1]] = (v + votes, max(sc, score))
            pick = max(options, key=lambda x: (options[x][1], options[x][0])) if options else None
            if pick != gold[k + 1]:
                ok = False
                break
        out[f'tf_pathway_{variant}'] = ok
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=1000)
    ap.add_argument('--train-lines', type=int, default=0, help='0 = whole training file')
    ap.add_argument('--per-signature', type=int, default=30)
    ap.add_argument('--mapping-limit', type=int, default=200)
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'flower_compare_20261025' / 'template_on_flower')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--case-timeout', type=int, default=300)
    ap.add_argument('--teacher-forced', action='store_true', help='score the teacher-forced pathway metric instead')
    cli = ap.parse_args()
    cli.output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # Two passes so only the sampled sequences are held in memory (the whole
    # test file as strings exhausted RAM next to FlowER). Same sample as before.
    with open(DATA / 'test.txt') as f:
        all_ids = sorted({line.rstrip('\n').rsplit('|', 1)[1] for line in f})
    rng = random.Random(20261025)
    rng.shuffle(all_ids)
    ids = all_ids[:cli.sample]
    del all_ids
    test = read_sequences(DATA / 'test.txt', wanted=set(ids))
    tests = {}
    for idx in ids:
        st = [graph(s) for s in states_of(test[idx])]
        tests[idx] = st
    sigs = {idx: signature(st[0], st[-1]) for idx, st in tests.items()}
    wanted_sigs = set(sigs.values())
    print(json.dumps({'event': 'test_loaded', 'sequences': len(tests), 'distinct_signatures': len(wanted_sigs),
                      'seconds': round(time.time() - t0)}), flush=True)

    cache = cli.output / f'pool_cache_s{cli.sample}_k{cli.per_signature}.pkl'
    pool = collections.defaultdict(list)
    if cache.exists():
        import pickle
        pool.update(pickle.loads(cache.read_bytes()))
        scanned = -1
        print(json.dumps({'event': 'pool_loaded_from_cache', 'signatures': len(pool)}), flush=True)
    current, current_idx, scanned = [], None, 0

    def flush():
        nonlocal scanned
        if not current:
            return
        scanned += 1
        try:
            states = states_of(current)
            first, last = graph(states[0]), graph(states[-1])
            sig = signature(first, last)
            if sig in wanted_sigs and len(pool[sig]) < cli.per_signature:
                pool[sig].append(template_of([graph(s) for s in states]))
        except Exception:
            pass

    with open(DATA / 'train.txt') if not cache.exists() else open(os.devnull) as f:
        for i, line in enumerate(f):
            if cli.train_lines and i >= cli.train_lines:
                break
            rxn, idx = line.rstrip('\n').rsplit('|', 1)
            if idx != current_idx:
                flush()
                current, current_idx = [], idx
            current.append(rxn)
        flush()
    if not cache.exists():
        import pickle
        cache.write_bytes(pickle.dumps(dict(pool)))
    print(json.dumps({'event': 'train_indexed', 'train_sequences_scanned': scanned,
                      'signatures_with_exemplars': sum(1 for s in wanted_sigs if pool.get(s)),
                      'seconds': round(time.time() - t0)}), flush=True)

    _G.update(pool=dict(pool), tests=tests, sigs=sigs, limit=cli.mapping_limit, timeout=cli.case_timeout)
    rows_path = cli.output / ('rows_tf.jsonl' if cli.teacher_forced else 'rows.jsonl')
    solver = _solve_tf_capped if cli.teacher_forced else _solve_capped
    rows = [json.loads(l) for l in rows_path.read_text().splitlines()] if rows_path.exists() else []
    done = {r['seq'] for r in rows}
    todo = [idx for idx in tests if idx not in done]
    import multiprocessing as mp
    with mp.get_context('fork').Pool(cli.workers) as workers, rows_path.open('a') as out:
        for k, row in enumerate(workers.imap_unordered(solver, todo, chunksize=2), 1):
            rows.append(row)
            out.write(json.dumps(row) + '\n')
            out.flush()
            if k % 50 == 0:
                print(json.dumps({'event': 'progress', 'done': len(rows), 'of': len(tests),
                                  'exact_so_far': round(sum(r.get('pathway_exact', r.get('tf_pathway_verified', False)) for r in rows) / len(rows), 3),
                                  'seconds': round(time.time() - t0)}), flush=True)
    n = len(rows)
    if cli.teacher_forced:
        summary = {'sample': n, 'tf_pathway_top1_verified': sum(r['tf_pathway_verified'] for r in rows) / n,
                   'tf_pathway_top1_unverified': sum(r['tf_pathway_unverified'] for r in rows) / n,
                   'reference': 'FlowER reports 88.8% top-1 pathway accuracy (teacher-forced, Nature 2025 Fig. 2c).',
                   'seconds': round(time.time() - t0)}
        (cli.output / 'summary_tf.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=1))
        return
    summary = {'sample': n, 'train_sequences_scanned': scanned,
               'pathway_top1_exact': sum(r['pathway_exact'] for r in rows) / n,
               'coverage_verified': sum(r['covered'] for r in rows) / n,
               'exact_given_covered': (sum(r['pathway_exact'] for r in rows) / max(1, sum(r['covered'] for r in rows))),
               'no_candidate': sum(r['candidates'] == 0 for r in rows) / n,
               'oracle_any_verified_exact': sum(r['gold_among_verified'] for r in rows) / n,
               'reference': 'FlowER reports 88.8% top-1 pathway accuracy (Graph2SMILES 92.0%) on its test set (Nature 2025, Fig. 2c).',
               'seconds': round(time.time() - t0)}
    (cli.output / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (cli.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
