"""P11: Mech-Agent v5 = gap detection -> state completion -> path revision (dev: T3_dev, J3).

Pool (as P10): base_default + samp1..4 + the three hidden-event-directed P2 paths.
  1 gap detection   every step of every candidate is atom-mapped (stepmap.py, RXNMapper);
                    a step with >= GAP (3; calibrate phase) heavy-atom bond changes, or an unparsable species, is a gap
  2 completion      one call: for the flagged steps, propose the intermediate species that split
                    them into elementary steps; proposals are kept only if RDKit parses them, they
                    are not a starting/final species and their heavy atoms fit the reactants
  3 revision        one call: candidates with their gaps + the checked states -> eliminate, then
                    keep or repair; the revised path is mapped again and, if gaps remain, one more
                    completion + revision round on that path alone (round 2)
Arms: V5_r1 (one round), V5 (up to two rounds). M_ms_none (P10) is the comparison.
Phases: requests -> [env-map stepmap.py] -> r1 -> requests -> [stepmap] -> r2 -> analyze.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations

from rdkit import Chem, RDLogger

import common4 as K
import p3_selector as P3
import p10_memory as P10
import stepmap as SMAP

RDLogger.DisableLog('rdApp.*')
GAP = 3
MAXLEN = 450

COMPLETE = ('Reactants and reagents (SMILES): {reactants}\nProducts (SMILES): {products}\nReaction conditions: {conditions}\n\n'
            'Candidate mechanisms for this reaction (intermediate SMILES after each step). An automatic atom-mapping check '
            'marks steps where several heavy-atom bonds change at once; such a step usually hides one or more short-lived '
            'intermediates (a tetrahedral adduct, an onium or ylide, a zwitterion, a rearranging cation, a catalyst- or '
            'reagent-bound species, ...), unless it is a genuinely concerted (e.g. pericyclic) step.\n\n{cands}\n\n'
            'For each flagged step, decide whether it hides intermediates. If it does, write the missing intermediate '
            'species that would split it into elementary steps. Also add any other intermediate you think every candidate '
            'misses. Give one species per line between <states> and </states>, as `C<candidate>.S<step>: SMILES` '
            '(the step it belongs before). Write nothing for steps you judge concerted.')
REVISE = ('Reactants and reagents (SMILES): {reactants}\nProducts (SMILES): {products}\nReaction conditions: {conditions}\n\n'
          'Candidate mechanisms proposed for this reaction (intermediate SMILES after each step; steps flagged by an '
          'atom-mapping check change several bonds at once):\n\n{cands}\n\nIntermediates proposed for the flagged steps '
          '(valid structures; not all of them need to be right):\n{states}\n\nFirst judge each candidate: discard those '
          'with unsound elementary steps or missing intermediates. Then write the final mechanism: keep the most plausible '
          'candidate or repair it, inserting the proposed intermediates where they belong and only where they are '
          'chemically justified. Return the final mechanism between [ANSWER] and [/ANSWER] as a JSON list of steps, each '
          '{{"step": n, "type": ..., "subtype": ..., "intermediate_smiles": ...}}.')


def sets(names=('T3_dev', 'J3')):
    return {n: K.load_gold() if n == 'gold' else K.load_fukuyama(n) for n in names}


def job_none(ds, case):
    """M_ms_none (P10 control) for sets P10 never ran: the same pool and eliminate/repair call."""
    cid = case['id']
    cands = cand_list(ds, cid)
    if len(cands) < 2:
        return 'skip'
    p = P10.pool(ds, cid)
    text = P3.render([p[n] for n in P3.order(cid, p)])
    return 'ok' if K.call(f'{ds}/M_ms_none', cid, P10.MS.format(memory='', cands=text, instruction=P10.NO_MEM,
                                                                  **fields(case))) else 'error'


# ------------------------------------------------------------------ gap detection
def canon(smi):
    m = Chem.MolFromSmiles(smi or '')
    return Chem.MolToSmiles(m) if m is not None else None


def cover(deficit, partners):
    """Smallest set (<= 3 species, fewest heavy atoms) of partners supplying the missing heavy atoms.

    Extra partners mislead RXNMapper (e.g. a product phenyl mapped onto benzoic acid), so each
    step only gets the partners its heavy-atom balance needs; best partial cover otherwise."""
    if not deficit:
        return []
    best, best_key = [], None
    for r in (1, 2, 3):
        for combo in combinations(partners, r):
            have = Counter()
            for p in combo:
                have += heavy(p) or Counter()
            key = (sum((deficit - have).values()), sum(have.values()))
            if best_key is None or key < best_key:
                best, best_key = list(combo), key
        if best_key and best_key[0] == 0:
            break
    return best


def step_rxns(case, mech):
    """One mapping request per step (None where a species does not parse)."""
    partners = [canon(s) for s in case['reactants']]
    partners = [p for p in dict.fromkeys(partners) if p]
    prev, out = None, []
    for st in mech:
        nxt = canon(st.get('intermediate_smiles'))
        if nxt is None:
            out.append(None)
            continue
        have = heavy(prev) if prev else Counter()
        left = list(dict.fromkeys(([prev] if prev else []) + cover((heavy(nxt) or Counter()) - have, partners)))
        if not left:
            left = [prev]
        if None in left:  # first step with nothing to cover (lit150 only; never reached on the frozen sets): no map
            out.append('')
            prev = nxt
            continue
        rxn = '.'.join(left) + '>>' + nxt
        out.append(rxn if len(rxn) <= MAXLEN else '')
        prev = nxt
    return out


_cache = None


def changes(rxn):
    global _cache
    if _cache is None:
        _cache = json.loads(SMAP.CACHE.read_text()) if SMAP.CACHE.exists() else {}
    if rxn is None:
        return 'unparsable'
    if rxn == '':
        return None
    return _cache.get(rxn)


def gaps(case, mech):
    """[(step index, reason)] for one candidate."""
    out = []
    for i, rxn in enumerate(step_rxns(case, mech)):
        n = changes(rxn)
        if n == 'unparsable':
            out.append((i, 'species does not parse'))
        elif isinstance(n, int) and n >= GAP:
            out.append((i, 'several heavy-atom bonds change in this one step'))
    return out


def render(case, cands):
    blocks = []
    for i, m in enumerate(cands, 1):
        g = dict(gaps(case, m))
        lines = [f"  step {j}: {s.get('subtype', '')} -> {s['intermediate_smiles']}"
                 + (f'   [FLAGGED: {g[j - 1]}]' if j - 1 in g else '') for j, s in enumerate(m, 1)]
        blocks.append(f'Candidate {i}:\n' + '\n'.join(lines))
    return '\n\n'.join(blocks)


def heavy(smi):
    m = Chem.MolFromSmiles(smi)
    return Counter(a.GetSymbol() for a in m.GetAtoms() if a.GetSymbol() != 'H') if m is not None else None


def checked_states(case, answer):
    """Proposed species that parse, are new and fit within the reactants' heavy atoms."""
    budget = Counter()
    for s in case['reactants']:
        budget += heavy(s) or Counter()
    ends = {canon(s) for s in case['reactants'] + case['products']}
    keep = []
    for line in K.tagged(answer, 'states'):
        loc, _, smi = line.rpartition(':')
        smi = smi.strip()
        c = canon(smi)
        h = heavy(smi) if c else None
        if c and c not in ends and h and not (h - budget) and c not in {k for _, k in keep}:
            keep.append((loc.strip() or '-', c))
    return keep


def fields(case):
    return dict(reactants='.'.join(case['reactants']), products='.'.join(case['products']), conditions=case['conditions'])


def cand_list(ds, cid):
    """The P10 pool in P3 order, identical candidates (same canonical species sequence) merged."""
    p = P10.pool(ds, cid)
    out, seen = [], set()
    for n in P3.order(cid, p):
        key = tuple(canon(s.get('intermediate_smiles')) for s in p[n])
        if key not in seen:
            seen.add(key)
            out.append(p[n])
    return out


# ------------------------------------------------------------------ phases
def requests(all_sets):
    reqs = set()
    for ds, cases in all_sets.items():
        for cid, case in cases.items():
            mechs = cand_list(ds, cid)
            for tag in ('V5_rev1',):
                m = K.parse_mech(K.cached(f'{ds}/{tag}', cid))
                if m:
                    mechs.append(m)
            for m in mechs:
                reqs.update(r for r in step_rxns(case, m) if r)
    SMAP.REQ.write_text('\n'.join(json.dumps({'rxn': r}) for r in sorted(reqs)) + '\n')
    print(len(reqs), 'step reactions requested')


def job_r1(ds, case):
    cid = case['id']
    cands = cand_list(ds, cid)
    if len(cands) < 2:
        return 'skip'
    text = render(case, cands)
    comp = K.call(f'{ds}/V5_comp1', cid, COMPLETE.format(cands=text, **fields(case)))
    if comp is None:
        return 'error'
    st = checked_states(case, comp)
    states = '\n'.join(f'- ({loc}) {s}' for loc, s in st) or '- none'
    return 'ok' if K.call(f'{ds}/V5_rev1', cid, REVISE.format(cands=text, states=states, **fields(case))) else 'error'


def job_r2(ds, case):
    cid = case['id']
    m = K.parse_mech(K.cached(f'{ds}/V5_rev1', cid))
    if not m or not gaps(case, m):
        return 'nogap'
    text = render(case, [m])
    comp = K.call(f'{ds}/V5_comp2', cid, COMPLETE.format(cands=text, **fields(case)))
    if comp is None:
        return 'error'
    st = checked_states(case, comp)
    if not st:
        return 'nostates'
    states = '\n'.join(f'- ({loc}) {s}' for loc, s in st)
    return 'ok' if K.call(f'{ds}/V5_rev2', cid, REVISE.format(cands=text, states=states, **fields(case))) else 'error'


def final(ds, cid, rounds):
    if rounds == 2:
        a = K.cached(f'{ds}/V5_rev2', cid)
        if a is not None and K.parse_mech(a):
            return a
    return K.cached(f'{ds}/V5_rev1', cid)


def analyze(all_sets):
    rep = {}
    pooled = {}
    for ds, cases in all_sets.items():
        rows = {a: [] for a in ('base_default', 'judge_pick', 'M_ms_none', 'V5_r1', 'V5')}
        strata = {}
        for cid, case in cases.items():
            if K.cached(f'{ds}/V5_rev1', cid) is None or K.cached(f'{ds}/M_ms_none', cid) is None:
                continue
            recs = {'base_default': K.cached(f'{ds}/base_default', cid), 'M_ms_none': K.cached(f'{ds}/M_ms_none', cid),
                    'V5_r1': final(ds, cid, 1), 'V5': final(ds, cid, 2)}
            p = P3.pool(ds, cid)
            nm = P3.order(cid, p)
            judge = K.cached(f'{ds}/judge', cid)
            sc = {}
            if judge is not None:
                jm = __import__('re').search(r'<best>\s*(\d+)', judge)
                pick = nm[int(jm.group(1)) - 1] if jm and 0 < int(jm.group(1)) <= len(nm) else 'base_default'
                sc['judge_pick'] = K.score(case, p.get(pick))
            for a, ans in recs.items():
                sc[a] = K.score(case, K.parse_mech(ans))
            for a, s in sc.items():
                rows[a].append((cid, s))
                pooled.setdefault(a, []).append((f'{ds}:{cid}', s))
            h = sc['base_default'].get('n_hidden_events')
            if h is not None:
                b = '0' if h == 0 else '1-2' if h <= 2 else '>=3'
                for a in ('base_default', 'M_ms_none', 'V5'):
                    strata.setdefault(b, {}).setdefault(a, []).append(sc[a]['all_present'])
        keys = ['all_present', 'in_order_frac'] + (['S_partial', 'in_order_full'] if ds == 'gold' else [])
        block = {a: K.summarize(r, keys) for a, r in rows.items() if r}
        for a in ('M_ms_none', 'V5_r1', 'V5', 'judge_pick'):
            if rows[a]:
                block[f'{a}_minus_A0'] = K.paired(rows[a], rows['base_default'], 'all_present')
        if ds == 'gold':
            block['V5_minus_A0_S_partial'] = K.paired(rows['V5'], rows['base_default'], 'S_partial')
            block['M_ms_none_minus_A0_S_partial'] = K.paired(rows['M_ms_none'], rows['base_default'], 'S_partial')
        block['V5_minus_M_ms_none'] = K.paired(rows['V5'], rows['M_ms_none'], 'all_present')
        block['strata_all_present'] = {b: {a: [len(v), round(sum(v) / len(v), 3)] for a, v in d.items()}
                                       for b, d in sorted(strata.items())}
        rep[ds] = block
    rep['pooled'] = {a: K.summarize(r, ['all_present']) for a, r in pooled.items()}
    for a in ('M_ms_none', 'V5_r1', 'V5', 'judge_pick'):
        if len(pooled.get(a, [])) == len(pooled['base_default']):
            rep['pooled'][f'{a}_minus_A0'] = K.paired(pooled[a], pooled['base_default'], 'all_present')
    rep['pooled']['V5_minus_M_ms_none'] = K.paired(pooled['V5'], pooled['M_ms_none'], 'all_present')
    name = 'SUMMARY_P11.json' if set(all_sets) == {'T3_dev', 'J3'} else f'SUMMARY_P11_{"_".join(all_sets)}.json'
    (K.OUT / name).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def calibrate(all_sets):
    """Offline: all_present of base_default with vs without a detected gap, per threshold."""
    global GAP
    for t in (2, 3, 4, 5):
        GAP = t
        cnt = {True: [], False: []}
        for ds, cases in all_sets.items():
            for cid, case in cases.items():
                m = K.parse_mech(K.cached(f'{ds}/base_default', cid))
                if m:
                    cnt[bool(gaps(case, m))].append(K.score(case, m)['all_present'])
        print(f'GAP>={t}: with gap n={len(cnt[True])} all_present={sum(cnt[True]) / max(1, len(cnt[True])):.3f} | '
              f'no gap n={len(cnt[False])} all_present={sum(cnt[False]) / max(1, len(cnt[False])):.3f}')
    GAP = 3


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['requests', 'calibrate', 'r1', 'r2', 'none', 'analyze'], required=True)
    ap.add_argument('--workers', type=int, default=200)
    ap.add_argument('--sets', default='T3_dev,J3')
    cli = ap.parse_args()
    s = sets(tuple(cli.sets.split(',')))
    if cli.phase == 'requests':
        requests(s)
    elif cli.phase == 'calibrate':
        calibrate(s)
    elif cli.phase in ('r1', 'r2', 'none'):
        fn = {'r1': job_r1, 'r2': job_r2, 'none': job_none}[cli.phase]
        K.run_pool(fn, [(ds, c) for ds, cs in s.items() for c in cs.values()], cli.workers)
    else:
        analyze(s)
