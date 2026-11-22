"""P26: Auto v6 candidate = species ledger (dev: T3_dev + J3).

Headroom (diag_auto2): adding the GT hidden species that some pool candidate already had but the v5 final
dropped would lift dev all_present 56.7 -> 63.9. v5 shows whole candidates, so the revision anchors on one
path and loses species that only other candidates contain. The ledger makes this an explicit species-level
decision: every species that occurs in a pool candidate or in the checked completion states but not in the
v5 final is listed once, with its support (number of candidates) and its context (species before -> it ->
species after in its first candidate). One call: decide per species (insert / reject), then the final path.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import common4 as K
import p10_memory as P10
import p11_mechagent as M

LEDGER = ('Reactants and reagents (SMILES): {reactants}\nProducts (SMILES): {products}\nReaction conditions: {conditions}\n\n'
          'Current mechanism (intermediate after each step):\n{draft}\n\nOther intermediates that independent attempts at '
          'this mechanism passed through but the current mechanism does not (support = number of the {n} attempts; '
          'context = the species before and after it in that attempt):\n{ledger}\n\nGo through the list one by one. '
          'For each species decide whether the real mechanism passes through it: insert it if the current mechanism '
          'skips it (two or more bond changes merged into one step, a missing adduct, ion pair, catalyst- or '
          'reagent-bound species, ...), reject it if it belongs to a different, less plausible pathway or is a '
          'spectator. Write one line per species, `k: insert` or `k: reject`, with a short reason. Then return the final '
          'mechanism between [ANSWER] and [/ANSWER] as a JSON list of steps, each '
          '{{"step": n, "type": ..., "subtype": ..., "intermediate_smiles": ...}}.')
MAXL = 14


def ledger(ds, case):
    cid = case['id']
    fin = K.parse_mech(M.final(ds, cid, 2)) or []
    fk = set().union(*[K.species_keys(s.get('intermediate_smiles') or '') for s in fin] or [set()])
    ends = {M.canon(s) for s in case['reactants'] + case['products']}
    sup, ctx = Counter(), {}
    pool = [m for m in P10.pool(ds, cid).values() if m]
    for m in pool:
        smis = [M.canon(s.get('intermediate_smiles')) for s in m]
        seen = set()
        for i, s in enumerate(smis):
            if not s or s in ends or s in seen or (K.species_keys(s) & fk):
                continue
            seen.add(s)
            sup[s] += 1
            ctx.setdefault(s, ((smis[i - 1] if i else 'reactants') or '?', (smis[i + 1] if i + 1 < len(smis) else 'products') or '?'))
    for t in ('V5_comp1', 'V5_comp2'):
        for _, s in M.checked_states(case, K.cached(f'{ds}/{t}', cid) or ''):
            if s not in sup and not (K.species_keys(s) & fk) and s not in ends:
                sup[s] += 0
                ctx.setdefault(s, ('proposed for a flagged step', ''))
    items = sorted(sup, key=lambda s: -sup[s])[:MAXL]
    return fin, items, sup, ctx, len(pool)


def job(ds, case):
    fin, items, sup, ctx, n = ledger(ds, case)
    if not fin:
        return 'nofinal'
    if not items:
        return 'empty'
    draft = '\n'.join(f"  step {j}: {s.get('subtype', '')} -> {s.get('intermediate_smiles')}" for j, s in enumerate(fin, 1))
    led = '\n'.join(f'{k}. {s}   support {sup[s]}/{n}; context: {ctx[s][0]} -> [this] -> {ctx[s][1]}'
                    if ctx[s][1] else f'{k}. {s}   ({ctx[s][0]})' for k, s in enumerate(items, 1))
    return 'ok' if K.call(f'{ds}/A6_ledger', case['id'], LEDGER.format(draft=draft, ledger=led, n=n, **M.fields(case))) else 'error'


def final(ds, cid):
    a = K.cached(f'{ds}/A6_ledger', cid)
    return a if a is not None and K.parse_mech(a) else M.final(ds, cid, 2)


def precision(case, m):
    import p12_controls as C
    hid, vis = C.gt_intermediates(case)
    gk = set().union(*[K.species_keys(s) for s in hid + vis] or [set()])
    ks = [K.species_keys(s.get('intermediate_smiles') or '') for s in (m or [])]
    ks = [k for k in ks if k]
    return sum(bool(k & gk) for k in ks) / len(ks) if ks else 0.0


def analyze(s):
    rows = {a: [] for a in ('Direct', 'M_ms_none', 'V5', 'A6')}
    for ds, cases in s.items():
        for cid, case in cases.items():
            if K.cached(f'{ds}/V5_rev1', cid) is None:
                continue
            ans = {'Direct': K.cached(f'{ds}/base_default', cid), 'M_ms_none': K.cached(f'{ds}/M_ms_none', cid),
                   'V5': M.final(ds, cid, 2), 'A6': final(ds, cid)}
            for a, t in ans.items():
                m = K.parse_mech(t)
                sc = K.score(case, m)
                sc['precision'] = precision(case, m)
                sc['n_steps'] = len(m or [])
                rows[a].append((f'{ds}:{cid}', sc))
    rep = {a: K.summarize(r, ['all_present', 'in_order_frac', 'precision', 'n_steps']) for a, r in rows.items()}
    for hi, lo in (('A6', 'V5'), ('A6', 'Direct'), ('A6', 'M_ms_none'), ('V5', 'Direct')):
        rep[f'{hi}_minus_{lo}'] = {k: K.paired(rows[hi], rows[lo], k) for k in ('all_present', 'precision')}
    (K.OUT / 'SUMMARY_P26_A6_DEV.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['run', 'analyze', 'show'], required=True)
    ap.add_argument('--sets', default='T3_dev,J3')
    ap.add_argument('--workers', type=int, default=120)
    cli = ap.parse_args()
    s = M.sets(tuple(cli.sets.split(',')))
    if cli.phase == 'run':
        K.run_pool(job, [(ds, c) for ds, cs in s.items() for c in cs.values()], cli.workers)
        analyze(s)
    elif cli.phase == 'show':
        ds = next(iter(s)); case = next(iter(s[ds].values()))
        fin, items, sup, ctx, n = ledger(ds, case)
        print(n, len(items), [(i, sup[i], ctx[i]) for i in items[:5]])
    else:
        analyze(s)
