"""P8: application pilot - an experiment reveals a hidden clue; can MechSight use it?

Observations simulated from the exact Fukuyama ground truth (T3_dev, J3), weakest first:
  formula  HRMS: molecular formula and m/z of one hidden GT intermediate (heavy hidden event
           preferred; the species of visibility.shown_species), structure not given
  label    isotope labelling: one atom of a transient (hidden-event) bond carries 13C/18O/15N in
           the reactant; the labelled product species is reported
  struct   the hidden intermediate's structure itself (upper bound, = 'giving the answer')
Arms:
  O_llm_<obs>  official prompt + the observation (plain LLM given the same clue)
  O_ms_<obs>   MechSight-obs: candidate pathways (P1/P2 pool) + the observation; the model
               eliminates candidates the clue rules out, then writes the final mechanism
Baseline without a clue: p6 base_default. Metrics: all GT intermediates present, heavy hidden
event recall, and recall of the event behind the clue.
"""
from __future__ import annotations

import argparse
import json
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

import common4 as K
import p3_selector as P3

sys.path.insert(0, str(K.HERE.parent / 'inert_knowledge'))
import visibility as V  # noqa: E402

RDLogger.DisableLog('rdApp.*')
OBS = ('\n\n## Experimental observation made while running this reaction\n\n{obs}\n\nThe mechanism must be '
       'consistent with this observation.\n\n## Input')
MS = ('Reactants and reagents (SMILES): {reactants}\nProducts (SMILES): {products}\nReaction conditions: {conditions}\n\n'
      'Experimental observation made while running this reaction:\n{obs}\n\nCandidate mechanisms proposed before the '
      'observation (intermediate SMILES after each step):\n\n{cands}\n\nFirst decide, candidate by candidate, whether it is '
      'consistent with the observation (which species would give this signal / where the label would end up). Discard the '
      'inconsistent ones. Then write the final mechanism: keep a consistent candidate, or repair the best one so that it '
      'explains the observation. Return the final mechanism in the JSON format below between [ANSWER] and [/ANSWER].\n\n'
      'Format: a JSON list of steps, each {{"step": n, "type": ..., "subtype": ..., "intermediate_smiles": ...}}.')
ISO = {'C': 13, 'O': 18, 'N': 15, 'S': 34}
ORDERS = {1.0: Chem.BondType.SINGLE, 2.0: Chem.BondType.DOUBLE, 3.0: Chem.BondType.TRIPLE, 1.5: Chem.BondType.AROMATIC}


def state_smiles(state, label, iso):
    """Dot-joined SMILES of the species of a PGL state that contains atom `label`, with that atom labelled."""
    atoms = dict(K.chem.pgl.iter_atoms(state))
    rw, idx = Chem.RWMol(), {}
    for n, a in atoms.items():
        at = Chem.Atom(a.element)
        at.SetFormalCharge(int(a.charge or 0))
        at.SetNumExplicitHs(int(a.Hcount or 0))
        at.SetNoImplicit(True)
        if n == label:
            at.SetIsotope(iso)
        idx[n] = rw.AddAtom(at)
    for (x, y), b in K.chem.pgl.bond_map(state).items():
        rw.AddBond(idx[x], idx[y], ORDERS.get(float(b.order or 1.0), Chem.BondType.SINGLE))
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    frags = Chem.GetMolFrags(mol, asMols=True)
    return next(Chem.MolToSmiles(f) for f in frags if any(a.GetIsotope() for a in f.GetAtoms()))


def observations(case):
    pgl = case['pgl']
    shown = V.shown_species(pgl)
    if not shown:
        return None
    mol = Chem.MolFromSmiles(shown[0])
    out = {'struct': f'An intermediate was isolated and characterised: {shown[0]}', 'shown_key': shown[1]}
    if mol is not None:
        out['formula'] = (f'HRMS of the reaction mixture shows a transient species of molecular formula '
                          f'{rdMolDescriptors.CalcMolFormula(mol)} (exact mass {Descriptors.ExactMolWt(mol):.4f}). '
                          'Its structure was not determined.')
    events = {t: i for t, i in K.P.hidden_events(pgl).items() if i['keys']}
    initial = K.chem.parse_pgl_literal(pgl['initial_state_pgl'])
    final = K.chem.parse_pgl_literal(pgl['target_state_pgl'])
    elems = dict(K.chem.pgl.iter_atoms(initial))
    for t, info in sorted(events.items(), key=lambda kv: not kv[1]['heavy']):
        lab = next((x for x in t if elems.get(x) is not None and elems[x].element in ISO), None)
        if lab is None:
            continue
        try:
            iso = ISO[elems[lab].element]
            r, p = state_smiles(initial, lab, iso), state_smiles(final, lab, iso)
        except Exception:
            continue
        out['label'] = (f'Isotope labelling: with the reactant labelled as {r}, the label is found in the product '
                        f'species {p}.')
        out['label_event'] = '-'.join(t)
        break
    return out


def sets():
    return {'T3_dev': K.load_fukuyama('T3_dev'), 'J3': K.load_fukuyama('J3')}


def job_llm(ds, case, kind):
    o = observations(case)
    if not o or kind not in o:
        return 'skip'
    p = K.official_prompt(case).replace('\n\n## Input', OBS.format(obs=o[kind]), 1)
    return 'ok' if K.call(f'{ds}/O_llm_{kind}', case['id'], p) else 'error'


def job_ms(ds, case, kind):
    o = observations(case)
    pool = P3.pool(ds, case['id'])
    if not o or kind not in o or len(pool) < 2:
        return 'skip'
    names = P3.order(case['id'], pool)
    p = MS.format(reactants='.'.join(case['reactants']), products='.'.join(case['products']),
                  conditions=case['conditions'], obs=o[kind], cands=P3.render([pool[n] for n in names]))
    return 'ok' if K.call(f'{ds}/O_ms_{kind}', case['id'], p) else 'error'


def analyze(all_sets):
    rep = {}
    for ds, cases in all_sets.items():
        for kind in ('formula', 'label', 'struct'):
            arms = {}
            for cid, case in cases.items():
                o = observations(case)
                if not o or kind not in o:
                    continue
                recs = {'O0': K.cached(f'{ds}/base_default', cid), 'O_llm': K.cached(f'{ds}/O_llm_{kind}', cid),
                        'O_ms': K.cached(f'{ds}/O_ms_{kind}', cid)}
                if recs['O0'] is None or recs['O_llm'] is None:
                    continue
                for arm, ans in recs.items():
                    if ans is None:
                        continue
                    m = K.parse_mech(ans)
                    s = K.score(case, m)
                    keys = {k for st in (m or []) for k in K.species_keys(st.get('intermediate_smiles') or '')}
                    s['clue_species_present'] = float(o['shown_key'] in keys)
                    arms.setdefault(arm, []).append((cid, s))
            if not arms:
                continue
            block = {a: K.summarize(r, ['all_present', 'clue_species_present', 'in_order_frac']) for a, r in arms.items()}
            for a in ('O_llm', 'O_ms'):
                if a in arms:
                    block[f'{a}_minus_O0'] = K.paired(arms[a], arms['O0'], 'all_present')
            if 'O_ms' in arms:
                block['O_ms_minus_O_llm'] = K.paired(arms['O_ms'], arms['O_llm'], 'all_present')
            rep[f'{ds}_{kind}'] = block
    (K.OUT / 'SUMMARY_P8.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['llm', 'ms', 'analyze'], default='llm')
    ap.add_argument('--workers', type=int, default=60)
    cli = ap.parse_args()
    s = sets()
    if cli.phase != 'analyze':
        fn = job_llm if cli.phase == 'llm' else job_ms
        K.run_pool(fn, [(ds, c, k) for ds, cs in s.items() for c in cs.values() for k in ('formula', 'label', 'struct')],
                   cli.workers)
    analyze(s)
