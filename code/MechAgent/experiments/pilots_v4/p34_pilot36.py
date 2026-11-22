"""P34: dev-set (lit172g, 165 papers) pilot of MechSight-Guided variants on gemini-3.6-flash (2026-11-06).

Upstream (Auto pool, hypotheses, clue samples) = the existing gemini-3.8-flash caches in lit172g; only the
changed final step runs on 3.6, next to 3.6 re-runs of the two references, so variant - reference compares
like with like. The confirmation set (litnew_g) is not touched.

Arms (all 3.6 unless noted)
  Dclues38 / G71_38  cached 3.8 references
  Dclues   Direct + clues (frozen prompt)          G71   v7.1 revision (frozen prompt)
  A0       G71 + every consensus species (support >= 3/7) it left out, appended (no call)
  A1       A0, but the left-out species are placed into the mechanism by one insertion call
  A2       v7.1 prompt with the exclusion option removed (every corroborated species must be included)
  B        A2 + catalytic-cycle states (one call per paper whose reactants contain a transition metal)
  A2u/Bu   A2 / B + consensus union as in A0
Split: tune = md5(id) % 3 != 0, check = the rest (choose on tune, report check once).

python p34_pilot36.py [--workers 300] [--analyze-only] [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

from rdkit import Chem

import common4 as K
import failover as F
import p11_mechagent as M
import p25_guided7 as G7
import p28_lit172_v7 as G
import p29_gold_rebuild as GOLD
import p30_lit172_gold as P
import p31_guided71 as T

MODEL = os.environ.get('PILOT_MODEL', 'gemini-3.6-flash')
for ep in F.ENDPOINTS:
    ep['model'] = MODEL
PDS = 'lit172g_p36'
RUN = K.OUT / 'p34_pilot36_20261106'
TM = {'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Zr', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'W', 'Re', 'Os',
      'Ir', 'Pt', 'Au'}

OLD_RULE = ('Include every corroborated intermediate in its proper place unless an elementary-step or observation '
            'argument rules it out; for each one you exclude, state the reason in one sentence before the answer.')
assert OLD_RULE in T.REVISE71
REVISE_A2 = T.REVISE71.replace(OLD_RULE, 'Include every corroborated intermediate in its proper place; none may be '
                                         'omitted.')

PLACE = (
    'Reactants and reagents (SMILES): {reactants}\nProducts (SMILES): {products}\n'
    'Reaction conditions: {conditions}\n\nExperimental observations:\n{obs}\n\n'
    'Current mechanism (JSON list of steps):\n{mech}\n\n'
    'Intermediates proposed by at least 3 of 7 independent mechanism proposals but missing from the current '
    'mechanism:\n{missing}\n\n'
    'Insert each missing intermediate at its chemically proper place, adjusting neighbouring steps so that every '
    'step stays elementary. Do not delete or change the existing intermediates. Return the final mechanism between '
    '[ANSWER] and [/ANSWER] as a JSON list of steps, each {{"step": n, "type": ..., "subtype": ..., '
    '"intermediate_smiles": ...}}.'
)

CAT = (
    'Reactants and reagents (SMILES): {reactants}\nProducts (SMILES): {products}\n'
    'Reaction conditions: {conditions}\n\nExperimental observations:\n{obs}\n\n'
    'The reaction involves the metal-containing reagents {metals}. Write out the metal-centred cycle species by '
    'species: the active catalyst, every substrate- or reagent-bound complex, the products of oxidative addition, '
    'transmetalation, ligand exchange, migratory insertion, single-electron transfer or radical capture, and the '
    'complex before reductive elimination or product release. Draw each species with its actual ligands taken '
    'from the reactant list (dative bonds as ->, charges and radical centres explicit, no abbreviations). '
    'Every SMILES must parse in RDKit: never put a dative bond on a ring-closure digit; for a chelating ligand '
    'draw one dative bond as a branch, e.g. c1ccc(-c2cccc[n]2->[Pd](Br)c2ccccc2)[n]c1. '
    'Return one species per line between <states> and </states>, as "short label: SMILES".'
)


def metals(case):
    out = []
    for s in case['reactants']:
        m = Chem.MolFromSmiles(s)
        if m and {a.GetSymbol() for a in m.GetAtoms()} & TM:
            out.append(s)
    return out


def tune(cid):
    return int(hashlib.md5(cid.encode()).hexdigest(), 16) % 3 != 0


def g71_prompt(case, rule_prompt, extra_states=()):
    cid = case['id']
    auto = K.parse_mech(M.final(P.DS, cid, 2))
    cands = ([auto] if auto else []) + M.cand_list(P.DS, cid)[:4] + G.clue_paths(cid)
    states = [s for tag in ('V5_comp1', 'V5_comp2') for _, s in M.checked_states(case, K.cached(f'{P.DS}/{tag}', cid) or '')]
    states = list(dict.fromkeys(states)) + list(extra_states)  # frozen v7.1 format: one SMILES per line
    return rule_prompt.format(obs=G.obs(case), nsrc=G7.NHYP + 3, consensus=T.consensus_lines(G.proposals(case), T.TH71),
                              cand_note=' (candidate 1 is the Auto final path)', cands=M.render(case, cands),
                              states='\n'.join(states) or '- none', **M.fields(case))


def cat_states(case):
    out = K.cached(f'{PDS}/B_cat', case['id'])
    return [s for _, s in M.checked_states(case, out or '')]


def consensus(case):
    return [s for s, n in G.proposals(case).most_common() if n >= T.TH71]


def missing(case, mech):
    have = [k for s in mech or [] if (k := K.species_keys(s.get('intermediate_smiles') or ''))]
    return [s for s in consensus(case) if (k := K.species_keys(s)) and not any(k & h for h in have)]


def job(kind, case):
    cid = case['id']
    if kind == 'Dclues':
        p = K.official_prompt(case).replace('\n\n## Input', G.P8.OBS.format(obs=G.obs(case)), 1)
        return 'ok' if K.call(f'{PDS}/O_llm', cid, p, temperature=0.2) else 'error'
    if kind == 'G71':
        return 'ok' if K.call(f'{PDS}/G71_rev', cid, g71_prompt(case, T.REVISE71)) else 'error'
    if kind == 'A2':
        return 'ok' if K.call(f'{PDS}/A2_rev', cid, g71_prompt(case, REVISE_A2)) else 'error'
    if kind == 'cat':
        p = CAT.format(obs=G.obs(case), metals=', '.join(metals(case)), **M.fields(case))
        return 'ok' if K.call(f'{PDS}/B_cat', cid, p, temperature=0.7) else 'error'
    if kind == 'A1':
        mech = K.parse_mech(K.cached(f'{PDS}/G71_rev', cid))
        miss = missing(case, mech)
        if not mech or not miss:
            return 'ok'
        p = PLACE.format(obs=G.obs(case), mech=json.dumps(mech, ensure_ascii=False),
                         missing='\n'.join(f'- {s}' for s in miss), **M.fields(case))
        return 'ok' if K.call(f'{PDS}/A1_place', cid, p) else 'error'
    if kind == 'B':
        extra = cat_states(case)
        if not extra:
            return 'ok'
        return 'ok' if K.call(f'{PDS}/B_rev', cid, g71_prompt(case, REVISE_A2, extra)) else 'error'
    raise ValueError(kind)


def union(case, text):
    mech = K.parse_mech(text) or []
    return mech + [{'step': None, 'intermediate_smiles': s} for s in missing(case, mech)]


def arms(case):
    cid = case['id']
    g = K.cached(f'{PDS}/G71_rev', cid)
    a1 = K.cached(f'{PDS}/A1_place', cid)
    a2 = K.cached(f'{PDS}/A2_rev', cid)
    b = K.cached(f'{PDS}/B_rev', cid) or a2  # no metal states -> B = A2
    a1m = K.parse_mech(a1) if a1 else K.parse_mech(g)
    return {'Dclues38': K.parse_mech(K.cached(f'{P.DS}/O_llm', cid)),
            'G71_38': K.parse_mech(K.cached(f'{P.DS}/G71_rev', cid)),
            'A0_38': union(case, K.cached(f'{P.DS}/G71_rev', cid)),
            'Dclues': K.parse_mech(K.cached(f'{PDS}/O_llm', cid)), 'G71': K.parse_mech(g), 'A0': union(case, g),
            'A1': a1m, 'A1u': union(case, json.dumps(a1m) if a1m else None),
            'A2': K.parse_mech(a2), 'A2u': union(case, a2), 'B': K.parse_mech(b), 'Bu': union(case, b)}


COMP = [('G71', 'Dclues'), ('A0', 'Dclues'), ('A1', 'Dclues'), ('A1u', 'Dclues'), ('A2', 'Dclues'),
        ('A2u', 'Dclues'), ('B', 'Dclues'), ('Bu', 'Dclues'), ('A0', 'G71'), ('A1', 'G71'), ('A2', 'G71'),
        ('B', 'A2'), ('Bu', 'A2u'), ('G71_38', 'Dclues38'), ('A0_38', 'Dclues38')]


def analyze(lst, split):
    rec = P.records()
    rows = {}
    for c in lst:
        if split != 'all' and tune(c['id']) != (split == 'tune'):
            continue
        refs = GOLD.substrate_refs(rec[c['id']]['record'])
        for a, mech in arms(c).items():
            rows.setdefault(a, []).append((c['id'], T.score(mech, c, refs)))
    if not rows:
        return {'split': split, 'n': 0, 'arms': {}, 'diffs': {}}
    keys = ['recall', 'precision', 'f1', 'n_pred']
    rep = {'split': split, 'n': len(rows.get('G71', [])),
           'arms': {a: K.summarize(r, keys + ['parsed']) for a, r in rows.items()},
           'diffs': {f'{h}-{l}': {k: K.paired(rows[h], rows[l], k) for k in ('recall', 'precision')} for h, l in COMP}}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=300)
    ap.add_argument('--analyze-only', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    cli = ap.parse_args()
    T.configure('dev')
    G.RUN, G.STATE = RUN, RUN / 'status.json'
    RUN.mkdir(parents=True, exist_ok=True)
    cs = P.cases()
    P.CS = cs
    lst = [c for c in cs.values() if c['clues'] and any(K.species_keys(s) for s in c['author'])]
    if cli.limit:
        lst = lst[:cli.limit]
    print(json.dumps({'model': MODEL, 'papers': len(lst), 'metal_papers': sum(bool(metals(c)) for c in lst)}), flush=True)
    if not cli.analyze_only:
        assert os.environ.get('FALLBACK_API_KEY'), 'FALLBACK_API_KEY required'
        G.phase('p36_round1', job, [(k, c) for c in lst for k in ('Dclues', 'G71', 'A2')]
                + [('cat', c) for c in lst if metals(c)], cli.workers)
        G.phase('p36_round2', job, [(k, c) for c in lst for k in ('A1', 'B')], cli.workers)
    out = {s: analyze(lst, s) for s in ('tune', 'check', 'all')}
    name = 'SUMMARY_P34.json' if not cli.limit else 'SUMMARY_P34_smoke.json'
    (RUN / name).write_text(json.dumps(out, indent=1))
    for s in ('tune', 'all'):
        r = out[s]
        print(f"== {s} n={r['n']}")
        for a, v in r['arms'].items():
            print(f"  {a:9s} R {v['recall'][0]:.3f}  P {v['precision'][0]:.3f}  n {v['n_pred'][0]:.2f}  parsed {v['parsed'][0]:.2f}")
        for d, v in r['diffs'].items():
            print(f"  {d:16s} dR {v['recall']}  dP {v['precision']}")


if __name__ == '__main__':
    main()
