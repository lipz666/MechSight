"""Analysis of the SMILES-mode runs (run.py): law, inertness and visibility on J3; baselines on T3/oMe."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location('smiles_run', HERE / 'run.py')  # 'run' is taken by p23_planning_probe
R = importlib.util.module_from_spec(_spec)
sys.modules['smiles_run'] = R
_spec.loader.exec_module(R)
P, PL, SM, phase2 = R.P, R.PL, R.SM, R.phase2
import j4_score  # noqa: E402
from rdkit import Chem  # noqa: E402

ACT = P.ROOT / 'outputs' / 'activation_pilot_20261103'


def narratives(model, cid):
    base = P.ROOT / 'outputs' / 'inert_knowledge_20261102'
    if model != 'gemini':
        base = base / 'models' / model
    folders = [base / 'S', base / 'G'] + ([ACT / 'N1', ACT / 'N2', ACT / 'N3'] if model == 'gemini' else [])
    out = []
    for f in folders:
        rec = PL.load(f, cid)
        if rec:
            out += rec['intermediates']
    return out


def j3_block(cases, model):
    d = R.OUT / 'J3' / model
    rows = []
    for cid, case in cases.items():
        m1, m2 = PL.load(d / 'M1', cid), PL.load(d / 'M2', cid)
        if not (m1 and m2):
            continue
        s1, s2 = SM.score(case, m1['states']), SM.score(case, m2['states'])
        known = {k for s in narratives(model, cid) for k, _ in P.smiles_keys(s)}
        ev = {t: i for t, i in P.hidden_events(case).items() if i['keys']}
        vp, v0 = PL.load(d / 'Vp', cid), PL.load(d / 'V0', cid)
        vis = None
        if vp:
            sh = R.shown(case)
            spv = SM.score(case, vp['states'])
            sv0 = SM.score(case, v0['states']) if v0 else None
            shown_ev = [t for t, i in ev.items() if sh and sh[1] in i['keys']]
            vis = [{'heavy': ev[t]['heavy'], 'Vp': spv['heavy_events'].get('-'.join(t), bool(ev[t]['keys'] & {k for s in vp['states'] for k, _ in P.smiles_keys(s)})),
                    'V0': None if not v0 else bool(ev[t]['keys'] & {k for s in v0['states'] for k, _ in P.smiles_keys(s)}),
                    'base': (bool(ev[t]['keys'] & {k for s in m1['states'] for k, _ in P.smiles_keys(s)}) +
                             bool(ev[t]['keys'] & {k for s in m2['states'] for k, _ in P.smiles_keys(s)})) / 2} for t in shown_ev]
        rows.append({'case': cid, 'M1': s1, 'M2': s2, 'known': {t: bool(ev[tuple(t.split('-'))]['keys'] & known) for t in s1['heavy_events']},
                     'vis': vis})
    if not rows:
        return {'cases': 0}
    ids = [r['case'] for r in rows]
    rep = {'cases': len(rows)}
    for arm in ('M1', 'M2'):
        vis_n = sum(r[arm]['vis_n'] for r in rows); hid_n = sum(r[arm]['hid_n'] for r in rows)
        ev = [(r['case'], v) for r in rows for v in r[arm]['heavy_events'].values()]
        strata = {}
        for r in rows:
            k = r[arm]['n_hidden_events']
            strata.setdefault('0' if k == 0 else ('1-2' if k <= 2 else '>=3'), []).append(r[arm]['all_present'])
        rep[arm] = {'visible_species_recall': round(sum(r[arm]['vis_hit'] for r in rows) / vis_n, 3) if vis_n else None,
                    'hidden_species_recall': round(sum(r[arm]['hid_hit'] for r in rows) / hid_n, 3) if hid_n else None,
                    'heavy_hidden_event_recall': phase2.ci([float(v) for _, v in ev], [c for c, _ in ev]) if ev else None,
                    'all_present': phase2.ci([float(r[arm]['all_present']) for r in rows], ids),
                    'in_order_fraction': phase2.ci([r[arm]['in_order_frac'] for r in rows], ids),
                    'parse_rate': round(statistics.mean(r[arm]['parse_rate'] for r in rows), 3),
                    'all_present_by_hidden': {b: f'{sum(v)}/{len(v)}' for b, v in sorted(strata.items())}}
    om = [(r['case'], r['M2']['heavy_events'][t], r['known'][t]) for r in rows for t, v in r['M1']['heavy_events'].items() if not v]
    rep['inertness'] = {'heavy_omitted_in_M1': len(om),
                        'reappear_in_M2': phase2.ci([float(a) for _, a, _ in om], [c for c, _, _ in om]) if om else None,
                        'known_in_narratives': phase2.ci([float(b) for _, _, b in om], [c for c, _, _ in om]) if om else None}
    vr = [(r['case'], x) for r in rows if r['vis'] for x in r['vis']]
    paired = [(c, x) for c, x in vr if x['V0'] is not None]
    if vr:
        rep['visibility'] = {'shown_events_all': len(vr),
                             'Vp_minus_base_all': phase2.ci([x['Vp'] - x['base'] for _, x in vr], [c for c, _ in vr]),
                             'paired_events': len(paired)}
        if paired:
            rep['visibility'].update({'Vp_paired': phase2.ci([float(x['Vp']) for _, x in paired], [c for c, _ in paired]),
                                      'V0_paired': phase2.ci([float(x['V0']) for _, x in paired], [c for c, _ in paired]),
                                      'Vp_minus_V0': phase2.ci([x['Vp'] - x['V0'] for _, x in paired], [c for c, _ in paired])})
            hp = [(c, x) for c, x in paired if x['heavy']]
            if hp:
                rep['visibility']['Vp_minus_V0_heavy'] = phase2.ci([x['Vp'] - x['V0'] for _, x in hp], [c for c, _ in hp])
    return rep


def ome_score(case, states):
    pred = []
    for s in states:
        keys = set()
        for part in s.split('.'):
            m = Chem.MolFromSmiles(part)
            if m is not None:
                keys.add(j4_score.key_of_mol(m))
        pred.append(keys)
    gold = [k for g in case['gold_steps'] if (k := j4_score.keys_of_smiles(g['smiles']))]
    hit = cursor = 0
    for keys in gold:
        found = next((i for i in range(cursor, len(pred)) if keys <= pred[i]), None)
        if found is not None:
            hit, cursor = hit + 1, found + 1
    prod = set()
    for p in case['products_smiles']:
        prod |= j4_score.keys_of_smiles(p) or set()
    res = {'in_order_fraction': hit / len(gold) if gold else 0.0, 'in_order_full': bool(gold) and hit == len(gold),
           'product_reached': bool(pred) and bool(prod) and prod <= pred[-1]}
    if j4_score.oMeS is not None:
        steps, prev = [], set()
        for s in states:
            parts = [x for x in s.split('.') if x]
            new = [x for x in parts if x not in prev] or parts
            steps.append(('x', max(new, key=lambda x: (Chem.MolFromSmiles(x).GetNumHeavyAtoms() if Chem.MolFromSmiles(x) else 0))))
            prev = set(parts)
        gsteps = [('x', g['smiles'], float(g.get('weight') or 1.0)) for g in case['gold_steps']]
        r = j4_score.oMeS(gsteps, steps)
        tot = sum(w for _, _, w in gsteps) or 1.0
        res.update(oMeS_total=r.S_total / tot, oMeS_partial=r.S_partial / tot)
    return res


def bench_block(cases, dataset, model, ids_subset=None):
    d = R.OUT / dataset / model
    rows = []
    for cid, case in cases.items():
        if ids_subset and cid not in ids_subset:
            continue
        rec = PL.load(d / 'M1', cid)
        if rec:
            rows.append((cid, ome_score(case, rec['states']) if dataset == 'oMe' else SM.score(case, rec['states'])))
    if not rows:
        return {'cases': 0}
    ids = [c for c, _ in rows]
    keys = (['in_order_fraction', 'in_order_full', 'product_reached', 'oMeS_total', 'oMeS_partial'] if dataset == 'oMe'
            else ['all_present', 'in_order_frac'])
    out = {'cases': len(rows), **{k: phase2.ci([float(s.get(k) or 0) for _, s in rows], ids) for k in keys}}
    if dataset != 'oMe':
        ev = [(c, v) for c, s in rows for v in s['heavy_events'].values()]
        out['heavy_hidden_event_recall'] = phase2.ci([float(v) for _, v in ev], [c for c, _ in ev]) if ev else None
    return out


def main():
    sets = R.load_cases()
    split = json.loads((P.ROOT / 'outputs/benchmarks_20261103/splits.json').read_text())
    rep = {}
    for model in ('gemini', 'gpt-6-luna', 'gemini-4-argon', 'claude-opus-5-5-high'):
        rep[model] = {'J3': j3_block(sets['J3'], model),
                      'T3_all': bench_block(sets['T3'], 'T3', model),
                      'T3_dev': bench_block(sets['T3'], 'T3', model, set(split['T3_dev'])),
                      'oMe': bench_block(sets['oMe'], 'oMe', model)}
    (R.OUT / 'SUMMARY.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
