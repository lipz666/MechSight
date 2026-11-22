"""P23: offline reviewer analyses (no API calls) -> SUMMARY_P23_OFFLINE.json.

A  over-generation: steps, intermediate precision, strict in-order, Direct vs Mech-Agent
B  inert vs blind spot inside the method (Fukuyama sets, hidden GT species missed by Direct)
C  gap-rule false-positive rate on exact reference steps (heavy-atom sigma changes per GT step)
D  specialist retention (same metric, in-template USPTO vs out-of-template T3_confirm) + reached product
E  Direct accuracy by hidden-intermediate count on all expert-curated sets (Fukuyama 242, oMe-Gold,
   oMe-Template when available)
"""
from __future__ import annotations

import json
from collections import Counter

import common4 as K
import p10_memory as P10
import p11_mechagent as M
import p12_controls as C
import p16_supplement_g38 as P16


def keyset(m):
    return set().union(*[K.species_keys(s.get('intermediate_smiles') or '') for s in (m or [])] or [set()])


def gt_keys(case):
    if 'pgl' in case:
        _, seq = K.SM.gt_species(case['pgl'])
        return set().union(*seq) if seq else set()
    return {k for g in case['gold'] for k in K.species_keys(g['intermediate_smiles'])}


def part_a():
    out = {}
    for ds in ('T3_confirm', 'gold', 'T3_dev', 'J3'):
        cases = M.sets((ds,))[ds]
        agg = {a: Counter() for a in ('Direct', 'MechSight')}
        for cid, case in cases.items():
            if K.cached(f'{ds}/V5_rev1', cid) is None:
                continue
            gt = gt_keys(case)
            ends = {k for s in case['reactants'] + case['products'] for k in K.species_keys(s)}
            for a, ans in (('Direct', K.cached(f'{ds}/base_default', cid)), ('MechSight', M.final(ds, cid, 2))):
                m = K.parse_mech(ans)
                sp = keyset(m) - ends
                s = K.score(case, m)
                g = agg[a]
                g['n'] += 1; g['steps'] += len(m or []); g['species'] += len(sp); g['correct'] += len(sp & gt)
                g['strict'] += float(s.get('in_order_full', s['in_order_frac'] >= 0.999))
        out[ds] = {a: {'n': g['n'], 'steps_per_path': round(g['steps'] / g['n'], 2),
                       'intermediates_per_path': round(g['species'] / g['n'], 2),
                       'precision': round(g['correct'] / max(g['species'], 1), 3),
                       'strict_in_order': round(g['strict'] / g['n'] * 100, 1)} for a, g in agg.items()}
    return out


def part_b():
    tot = Counter()
    for ds in ('T3_confirm', 'T3_dev', 'J3'):
        for cid, case in M.sets((ds,))[ds].items():
            if K.cached(f'{ds}/V5_rev1', cid) is None:
                continue
            hid, _ = C.gt_intermediates(case)
            direct = keyset(K.parse_mech(K.cached(f'{ds}/base_default', cid)))
            final = keyset(K.parse_mech(M.final(ds, cid, 2)))
            pool = set()
            for m in P10.pool(ds, cid).values():
                pool |= keyset(m)
            states = set()
            for tag in ('V5_comp1', 'V5_comp2'):
                for _, smi in M.checked_states(case, K.cached(f'{ds}/{tag}', cid) or ''):
                    states |= K.species_keys(smi)
            for smi in hid:
                k = K.species_keys(smi)
                if not k or k & direct:
                    continue
                tot['missed_by_direct'] += 1
                if k & pool:
                    tot['latent_in_own_pool'] += 1
                    tot['latent_recovered' if k & final else 'latent_not_recovered'] += 1
                elif k & states:
                    tot['new_from_completion'] += 1
                    tot['new_recovered' if k & final else 'new_not_recovered'] += 1
                else:
                    tot['blind_spot'] += 1
    n = tot['missed_by_direct']
    return {k: [v, round(v / n * 100, 1)] for k, v in tot.items()}


def part_c():
    fm = K.P.fm
    dist = Counter()
    for ds in ('J3', 'T3_dev', 'T3_confirm'):
        for case in K.load_fukuyama(ds).values():
            for st in fm.gold_steps_from_clean(case['pgl']):
                dist[sum(1 for ev in fm.sigma_events(st) if 'H*' not in ev[1:])] += 1
    tot = sum(dist.values())
    fl = sum(v for k, v in dist.items() if k >= 3)
    return {'reference_steps': tot, 'heavy_sigma_changes_distribution': dict(sorted(dist.items())),
            'reference_steps_flagged_ge3': [fl, round(fl / tot * 100, 1)],
            'note': 'Diels-Alder (2 formed) and [3,3] shifts (1+1) stay below 3; flagged correct steps are mostly '
                    'oxidative additions, three-membered onium-ring formations and concerted fragmentations'}


def part_d():
    d = json.loads((K.OUT / 'SUMMARY_P18_SPEC_stdinchi.json').read_text())
    us, t3 = d['uspto31k']['ChRIMP'], d['T3_confirm']['ChRIMP']
    m = lambda x: round(x[0] * 100, 1)  # noqa: E731
    out = {'ChRIMP': {'in_template_uspto': m(us['ChRIMP']['all_present']), 'out_of_template_T3': m(t3['ChRIMP']['all_present']),
                      'reached_in': m(us['ChRIMP']['reached']), 'reached_out': m(t3['ChRIMP']['reached']),
                      'coverage_uspto': us['coverage'], 'coverage_T3': t3['coverage']},
           'LLM_Direct': {'in_template_uspto': m(us['LLM_direct']['all_present']), 'out_of_template_T3': m(t3['LLM_direct']['all_present']),
                          'reached_in': m(us['LLM_direct']['reached']), 'reached_out': m(t3['LLM_direct']['reached'])},
           'FlowER_T3': {k: m(v) for k, v in d['T3_confirm']['FlowER']['FlowER'].items() if k != 'n'} | {'n': d['T3_confirm']['FlowER']['FlowER']['n']}}
    for k in ('ChRIMP', 'LLM_Direct'):
        out[k]['retention_pct'] = round(out[k]['out_of_template_T3'] / out[k]['in_template_uspto'] * 100, 1)
    us_v5 = K.OUT / 'SUMMARY_USPTO_V5.json'
    if us_v5.exists():
        u = json.loads(us_v5.read_text())
        out['MechSight_uspto'] = {a: {'n': u[a]['n'], 'all_present': m(u[a]['all_present'])} for a in ('Direct', 'MechAgent') if a in u}
    return out


def part_e():
    strata = {}
    def add(src, b, v):
        strata.setdefault(src, {}).setdefault(b, []).append(v)
    for ds in ('J3', 'T3_dev', 'T3_confirm'):
        for cid, case in K.load_fukuyama(ds).items():
            a = K.cached(f'{ds}/base_default', cid)
            if a is None:
                continue
            s = K.score(case, K.parse_mech(a))
            h = s['n_hidden_events']
            add('Fukuyama', '0' if h == 0 else '1-2' if h <= 2 else '>=3', s['all_present'])
    for ds, loader in (('gold', K.load_gold), ('template', lambda: {c['id']: c for c in K.load_template()})):
        for cid, case in loader().items():
            a = K.cached(f'{ds}/base_default', cid)
            if a is None:
                continue
            hid, _ = P16.gold_gt_intermediates(case)
            add(ds, '0' if not hid else '1-2' if len(hid) <= 2 else '>=3', K.score(case, K.parse_mech(a))['all_present'])
    out = {src: {b: [len(v), round(sum(v) / len(v) * 100, 1)] for b, v in sorted(d.items())} for src, d in strata.items()}
    allb = {}
    for d in strata.values():
        for b, v in d.items():
            allb.setdefault(b, []).extend(v)
    out['all_expert_sets'] = {b: [len(v), round(sum(v) / len(v) * 100, 1)] for b, v in sorted(allb.items())}
    return out


if __name__ == '__main__':
    rep = {'A_overgeneration': part_a(), 'B_inert_vs_blind': part_b(), 'C_gap_rule_on_reference_steps': part_c(),
           'D_specialist_retention': part_d(), 'E_direct_by_hidden_count': part_e()}
    (K.OUT / 'SUMMARY_P23_OFFLINE.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
