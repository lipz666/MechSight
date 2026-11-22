"""V2: SMILES-only stepwise mechanisms (no PGL), gemini via the two-gateway client.

Per J3 case the model sees reactants/reagents, products and conditions as SMILES (the same
inputs as the free-text probe S) and writes the full stepwise mechanism: one line per
elementary step with the dot-joined SMILES of every species present after that step.
Two samples (M1, M2) give the noise floor.

Scoring is species-level with sigma-graph keys (heavy-atom connectivity + H counts; charges
and bond orders ignored), so no atom mapping is needed:
  * recall of GT visible vs hidden intermediates (hidden = contains a transient bond)
  * heavy hidden events recalled, compared with the PGL path E (same species-level scoring)
  * in-order GT intermediates, all GT intermediates present
  * validity: SMILES parse rate and per-step element / charge balance
  * inertness in SMILES: heavy hidden events omitted in M1 -> re-appear in M2 vs. known in
    the other free-text narratives (S, G and N1-N3 of the activation pilot)
Registration: outputs/smiles_validation_20261103/REGISTRATION.json.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'mechsight'))
import pilot as PL  # noqa: E402
from pilot import A, C, P, chem, fm, phase2  # noqa: E402
import failover as F  # noqa: E402

OUT = P.ROOT / 'outputs' / 'smiles_validation_20261103'
ACT = P.ROOT / 'outputs' / 'activation_pilot_20261103'
TASK = ('Reactants and reagents (SMILES): {reactants}\n'
        'Products (SMILES): {products}\n'
        'Reaction conditions: {conditions}\n\n'
        'Write the complete stepwise arrow-pushing mechanism from the reactants to the products as a sequence of '
        'elementary steps (one bond-forming/breaking event or one concerted event per step; proton transfers count '
        'as steps). Reason first. Then, inside <mechanism> and </mechanism>, write one line per elementary step: the '
        'dot-separated SMILES of ALL species present after that step (keep spectator ions, reagents and byproducts so '
        'that every step is atom- and charge-balanced), with explicit formal charges. The last line must contain the '
        'products.')


def job(tag, case):
    dest = OUT / tag / f"{case['case_id']}.json"
    if dest.exists():
        return 'cached'
    initial = chem.parse_pgl_literal(case['initial_state_pgl'])
    target = chem.parse_pgl_literal(case['target_state_pgl'])
    prompt = TASK.format(reactants=P.endpoint_smiles(initial), products=P.endpoint_smiles(target),
                         conditions=A.conditions_text(case))
    m = F.FailoverModel(OUT / tag / 'traces' / f"{case['case_id']}.jsonl", max_tokens=P.MAX_TOKENS)
    try:
        ans = m.complete([{'role': 'system', 'content': P.FREE_SYSTEM}, {'role': 'user', 'content': prompt}], tag)
    except Exception as exc:
        return f'error:{type(exc).__name__}'
    found = re.search(r'<mechanism>(.*?)</mechanism>', ans, re.S)
    lines = [ln.strip().strip('`').strip() for ln in (found.group(1) if found else '').splitlines()]
    lines = [re.sub(r'^(step\s*\d+\s*[:.)-]\s*|\d+\s*[:.)]\s*)', '', ln, flags=re.I) for ln in lines if ln]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({'case_id': case['case_id'], 'states': lines, 'tag_found': bool(found),
                                'endpoints': m.endpoints_used, 'prompt': prompt}, indent=1))
    return 'ok'


# ------------------------------------------------------------------ scoring
def formula(smi):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    els = Counter(a.GetSymbol() for a in mol.GetAtoms())
    return els, sum(a.GetFormalCharge() for a in mol.GetAtoms())


def gt_species(case):
    init = chem.parse_pgl_literal(case['initial_state_pgl'])
    states = C.gt_states(case)
    endpoint = {P.state_component_keys(s, c)[0] for s in (init, states[-1]) for c in C._components(s)}
    seq = []
    for st in states[:-1]:
        keys = {P.state_component_keys(st, c)[0] for c in C._components(st)} - endpoint - {''}
        if keys:
            seq.append(keys)
    return endpoint, seq


def score(case, states_smiles):
    endpoint, gt_seq = gt_species(case)
    pred = [{k for k, _ in P.smiles_keys(s)} for s in states_smiles]
    allpred = set().union(*pred) if pred else set()
    events = P.hidden_events(case)
    hidden_keys = set().union(*(i['keys'] for i in events.values() if i['keys'])) if events else set()
    gt_all = set().union(*gt_seq) if gt_seq else set()
    vis, hid = gt_all - hidden_keys, gt_all & hidden_keys
    # in-order: each GT intermediate state's new species found in order
    cursor = hits = 0
    for keys in gt_seq:
        found = next((i for i in range(cursor, len(pred)) if keys & pred[i]), None)
        if found is not None:
            hits, cursor = hits + 1, found + 1
    parsed = [formula(s) for s in states_smiles]
    balance = [a is not None and b is not None and a == b for a, b in zip(parsed, parsed[1:])]
    return {'vis_n': len(vis), 'vis_hit': len(vis & allpred), 'hid_n': len(hid), 'hid_hit': len(hid & allpred),
            'all_present': gt_all <= allpred, 'in_order_frac': hits / len(gt_seq) if gt_seq else 1.0,
            'n_states': len(states_smiles), 'parse_rate': (sum(p is not None for p in parsed) / len(parsed)) if parsed else 0.0,
            'balanced_steps': (sum(balance) / len(balance)) if balance else None,
            'heavy_events': {('-'.join(t)): bool(i['keys'] & allpred) for t, i in events.items() if i['keys'] and i['heavy']},
            'n_hidden_events': sum(1 for i in events.values() if i['keys'])}


def pgl_species(case, actions):
    states, _ = fm.replay(case['initial_state_pgl'], actions)
    return [C._molecule_smiles(s, c) for s in states for c in C._components(s)]


def analyze(cases):
    rows = []
    for cid, case in cases.items():
        m1, m2, e = PL.load(OUT / 'M1', cid), PL.load(OUT / 'M2', cid), PL.load(P.OUT / 'E', cid)
        if not (m1 and m2 and e):
            continue
        s1, s2 = score(case, m1['states']), score(case, m2['states'])
        se = score(case, ['.'.join(pgl_species(case, e['actions_pgl']))])
        narr = []
        for folder in (P.OUT / 'S', P.OUT / 'G', ACT / 'N1', ACT / 'N2', ACT / 'N3'):
            rec = PL.load(folder, cid)
            if rec:
                narr += rec['intermediates']
        known = {k for s in narr for k, _ in P.smiles_keys(s)}
        rows.append({'case': cid, 'M1': s1, 'M2': s2, 'E': se,
                     'known': {t: bool(P.hidden_events(case)[tuple(t.split('-'))]['keys'] & known) for t in s1['heavy_events']}})
    ids = [r['case'] for r in rows]

    def frac(arm, num, den):
        n, d = sum(r[arm][num] for r in rows), sum(r[arm][den] for r in rows)
        return round(n / d, 3) if d else None

    rep = {'cases': len(rows)}
    for arm in ('M1', 'M2', 'E'):
        ev = [(r['case'], v) for r in rows for v in r[arm]['heavy_events'].values()]
        rep[arm] = {'visible_species_recall': frac(arm, 'vis_hit', 'vis_n'), 'hidden_species_recall': frac(arm, 'hid_hit', 'hid_n'),
                    'heavy_hidden_event_recall': phase2.ci([float(v) for _, v in ev], [c for c, _ in ev]) if ev else None,
                    'all_GT_intermediates_present': phase2.ci([float(r[arm]['all_present']) for r in rows], ids),
                    'in_order_fraction': phase2.ci([r[arm]['in_order_frac'] for r in rows], ids)}
        if arm != 'E':
            rep[arm]['parse_rate'] = round(statistics.mean(r[arm]['parse_rate'] for r in rows), 3)
            bal = [r[arm]['balanced_steps'] for r in rows if r[arm]['balanced_steps'] is not None]
            rep[arm]['balanced_step_rate'] = round(statistics.mean(bal), 3) if bal else None
            rep[arm]['median_states'] = statistics.median(r[arm]['n_states'] for r in rows)
        strata = {}
        for r in rows:
            k = r[arm]['n_hidden_events']
            b = '0' if k == 0 else ('1-2' if k <= 2 else '>=3')
            strata.setdefault(b, []).append(float(r[arm]['all_present']))
        rep[arm]['all_present_by_hidden_events'] = {b: f'{sum(v):.0f}/{len(v)}' for b, v in sorted(strata.items())}
    om = [(r['case'], r['M2']['heavy_events'][t], r['known'][t]) for r in rows for t, v in r['M1']['heavy_events'].items() if not v]
    rep['inertness_smiles'] = {'heavy_omitted_in_M1': len(om),
                               'reappear_in_M2': phase2.ci([float(a) for _, a, _ in om], [c for c, _, _ in om]) if om else None,
                               'known_in_other_narratives': phase2.ci([float(b) for _, _, b in om], [c for c, _, _ in om]) if om else None}
    (OUT / 'SUMMARY_V2.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=500)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--analyze-only', action='store_true')
    cli = ap.parse_args()
    sys.path.insert(0, str(P.ROOT / 'experiments' / 'agent_v2'))
    import j3_dev
    cases = j3_dev.load_cases()
    if cli.limit:
        cases = dict(list(cases.items())[:cli.limit])
    OUT.mkdir(parents=True, exist_ok=True)
    if not cli.analyze_only:
        PL.pool(job, [(t, c) for t in ('M1', 'M2') for c in cases.values()], cli.workers)
    analyze(cases)


if __name__ == '__main__':
    main()
