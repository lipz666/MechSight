"""P12: the Fig. 1d / Fig. 2b controls on the dev sets (T3_dev, J3).

  O_hid      official prompt + every GT hidden intermediate (structures, order not given)
  O_vis      official prompt + every GT visible intermediate (control: true intermediates of the
             other kind; 84/125 dev cases have fewer visible than hidden ones, so not count-matched)
  (O_hid1    one hidden intermediate = P8 O_llm_struct, reused)
  SR1, SR2   self-refine on base_default: the model checks its own mechanism for skipped
             intermediates and steps that change several bonds at once, then revises (2 rounds)
Cases without a hidden intermediate keep base_default in O_hid/O_vis (nothing to give).
"""
from __future__ import annotations

import argparse
import json

import common4 as K
import p8_observations as P8

ORACLE = ('\n\n## Known intermediates\n\nThe following species are known intermediates of this reaction (order not '
          'given):\n{species}\n\nThe mechanism must pass through them.\n\n## Input')
REFINE = ('Reactants and reagents (SMILES): {reactants}\nProducts (SMILES): {products}\nReaction conditions: {conditions}\n\n'
          'A proposed stepwise mechanism (intermediate SMILES after each step):\n{mech}\n\nCritically check it step by '
          'step: does any step skip a short-lived intermediate or change several bonds at once? Is every step a sound '
          'elementary step? Then write the improved mechanism (or the same one if it is already right) between [ANSWER] '
          'and [/ANSWER] as a JSON list of steps, each {{"step": n, "type": ..., "subtype": ..., "intermediate_smiles": ...}}.')


def gt_intermediates(case):
    """(hidden SMILES list, visible SMILES list) of the GT path, one entry per distinct species."""
    pgl = case['pgl']
    events = K.P.hidden_events(pgl)
    hidden_keys = set().union(*(i['keys'] for i in events.values() if i['keys'])) if events else set()
    initial = K.chem.parse_pgl_literal(pgl['initial_state_pgl'])
    states = K.C.gt_states(pgl)
    endpoint = {K.P.state_component_keys(s, c)[0] for s in (initial, states[-1]) for c in K.C._components(s)}
    hid, vis, seen = [], [], set()
    for st in states[:-1]:
        for comp in K.C._components(st):
            key = K.P.state_component_keys(st, comp)[0]
            if not key or key in endpoint or key in seen:
                continue
            seen.add(key)
            (hid if key in hidden_keys else vis).append(K.C._molecule_smiles(st, comp))
    return hid, vis


def fields(case):
    return dict(reactants='.'.join(case['reactants']), products='.'.join(case['products']), conditions=case['conditions'])


def job_oracle(ds, case):
    if 'pgl' not in case:
        return 'skip'
    hid, vis = gt_intermediates(case)
    if not hid:
        return 'nohidden'
    for tag, sp in (('O_hid', hid), ('O_vis', vis)):
        if sp:
            p = K.official_prompt(case).replace('\n\n## Input', ORACLE.format(species='\n'.join(f'- {s}' for s in sp)), 1)
            K.call(f'{ds}/{tag}', case['id'], p)
    return 'ok'


def show(m):
    return '\n'.join(f"  step {j}: {s.get('subtype', '')} -> {s['intermediate_smiles']}" for j, s in enumerate(m, 1))


def job_refine(ds, case):
    prev = K.parse_mech(K.cached(f'{ds}/base_default', case['id']))
    for tag in ('SR1', 'SR2'):
        if not prev:
            return 'noparse'
        ans = K.call(f'{ds}/{tag}', case['id'], REFINE.format(mech=show(prev), **fields(case)))
        prev = K.parse_mech(ans) or prev
    return 'ok'


def analyze(all_sets):
    rep = {}
    for ds, cases in all_sets.items():
        rows = {a: [] for a in ('base_default', 'O_hid', 'O_vis', 'O_hid1', 'SR1', 'SR2')}
        strata = {}
        for cid, case in cases.items():
            base = K.cached(f'{ds}/base_default', cid)
            if base is None or K.cached(f'{ds}/SR2', cid) is None:
                continue
            hid, vis = gt_intermediates(case) if 'pgl' in case else ([], [])
            ans = {'base_default': base, 'SR1': K.cached(f'{ds}/SR1', cid), 'SR2': K.cached(f'{ds}/SR2', cid),
                   'O_hid': K.cached(f'{ds}/O_hid', cid) if hid else base,
                   'O_vis': (K.cached(f'{ds}/O_vis', cid) if vis else base) if hid else base,
                   'O_hid1': (K.cached(f'{ds}/O_llm_struct', cid) or base) if hid else base}
            if any(v is None for v in ans.values()):
                continue
            for a, t in ans.items():
                s = K.score(case, K.parse_mech(t))
                rows[a].append((cid, s))
                if 'pgl' not in case:
                    continue
                b = '0' if not hid else '1-2' if s['n_hidden_events'] <= 2 else '>=3'
                strata.setdefault(b, {}).setdefault(a, []).append(s['all_present'])
        keys = ['all_present', 'in_order_frac'] + (['S_partial'] if ds == 'gold' else [])
        block = {a: K.summarize(r, keys) for a, r in rows.items() if r}
        for a in ('O_hid', 'O_vis', 'O_hid1', 'SR1', 'SR2'):
            block[f'{a}_minus_A0'] = K.paired(rows[a], rows['base_default'], 'all_present')
        block['strata_all_present'] = {b: {a: [len(v), round(sum(v) / len(v), 3)] for a, v in d.items()}
                                       for b, d in sorted(strata.items())}
        rep[ds] = block
    name = 'SUMMARY_P12.json' if set(all_sets) == {'T3_dev', 'J3'} else f'SUMMARY_P12_{"_".join(all_sets)}.json'
    (K.OUT / name).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['oracle', 'refine', 'analyze'], required=True)
    ap.add_argument('--workers', type=int, default=200)
    ap.add_argument('--sets', default='T3_dev,J3')
    cli = ap.parse_args()
    s = {n: K.load_gold() if n == 'gold' else K.load_fukuyama(n) for n in cli.sets.split(',')}
    if cli.phase in ('oracle', 'refine'):
        K.run_pool(job_oracle if cli.phase == 'oracle' else job_refine,
                   [(ds, c) for ds, cs in s.items() for c in cs.values()], cli.workers)
    else:
        analyze(s)
