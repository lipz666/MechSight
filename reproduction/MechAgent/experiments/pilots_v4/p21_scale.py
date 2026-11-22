"""P21: scale-up confirmation of the frozen Mech-Agent v5 on oMe-Template (167) + oMe-Silver (600).

Registration: OUT/REGISTRATION_v5_scale.json (written before any call on these sets). v5 uses no
retrieval and these sets were never used to build or tune it. Sets: all of oMe-Template; 600
Silver reactions drawn with a fixed seed from Silver minus Silver-dev, after removing reactions
whose reactants>>products duplicate oMe-Gold, Template or another drawn reaction.
Arms: Direct, CoT, Self-consistency (MBR over the same pool), Mech-Agent v5 (+ format-only
relabel for official S_partial, as in DEVIATION_v5_confirm.md, applied from the start here).
Own stepmap request/cache files (STEPMAP_REQ / STEPMAP_CACHE set by the launcher).
"""
from __future__ import annotations

import argparse
import json
import random

from rdkit import Chem, RDLogger

import common4 as K
import p2_enumerate as P2
import p3_selector as P3
import p6_baselines as P6
import p10_memory as P10
import p11_mechagent as M
import p15_relabel as RL
import p16_supplement_g38 as P16

RDLogger.DisableLog('rdApp.*')
SPLIT = K.OUT / 'scale_split.json'
SEED, N_SILVER = 20261104, 600


def rxn_key(c):
    can = lambda xs: '.'.join(sorted(Chem.MolToSmiles(m) if (m := Chem.MolFromSmiles(x)) else x for x in xs))  # noqa: E731
    return can(c['reactants']) + '>>' + can(c['products'])


def sets():
    tmpl = {c['id']: c for c in K.load_template()}
    if not SPLIT.exists():
        dev = set(json.loads((K.OUT / 'silver_dev.json').read_text())['ids'])
        seen = {rxn_key(c) for c in K.load_gold().values()} | {rxn_key(c) for c in tmpl.values()}
        pool = sorted((c for c in K.load_silver() if c['id'] not in dev), key=lambda c: c['id'])
        random.Random(SEED).shuffle(pool)
        pick = []
        for c in pool:
            k = rxn_key(c)
            if k in seen or len(c['gold']) < 2:
                continue
            seen.add(k)
            pick.append(c['id'])
            if len(pick) == N_SILVER:
                break
        SPLIT.write_text(json.dumps({'seed': SEED, 'silver_ids': pick, 'rule': 'Silver minus Silver-dev, >= 2 steps, '
                                     'deduplicated against oMe-Gold, Template and each other'}, indent=1))
    ids = set(json.loads(SPLIT.read_text())['silver_ids'])
    silver = {c['id']: c for c in K.load_silver() if c['id'] in ids}
    return {'template': tmpl, 'silver_conf': silver}


def job_pool(ds, arm, case):
    return P2.job(ds, case) if arm == 'enum3' else P6.job(ds, arm, case)


def job_relabel(ds, case):
    cid = case['id']
    orig = K.parse_mech(M.final(ds, cid, 2))
    if not orig:
        return 'noparse'
    steps = '\n'.join(f"  step {j}: {s['intermediate_smiles']}" for j, s in enumerate(orig, 1))
    return 'ok' if K.call(f'{ds}/RL_MechAgent', cid, RL.PROMPT.format(allowed=json.dumps(RL.ALLOWED, indent=1),
                                                                      steps=steps)) else 'error'


def relabelled(ds, cid):
    orig = K.parse_mech(M.final(ds, cid, 2))
    if not orig:
        return None
    lab = K.parse_mech(K.cached(f'{ds}/RL_MechAgent', cid)) or []
    out = []
    for i, s in enumerate(orig):
        t = (lab[i].get('type'), lab[i].get('subtype')) if i < len(lab) else (None, None)
        if t not in RL.PAIRS:
            t = (s.get('type'), s.get('subtype'))
        out.append({'step': i + 1, 'type': t[0], 'subtype': t[1], 'intermediate_smiles': s['intermediate_smiles']})
    return out


def analyze(all_sets):
    rep, pooled = {}, {}
    keys = ['all_present', 'in_order_full', 'S_partial']
    for ds, cases in all_sets.items():
        rows, strata = {}, {}
        for cid, case in cases.items():
            if K.cached(f'{ds}/V5_rev1', cid) is None or K.cached(f'{ds}/base_default', cid) is None:
                continue
            p = P10.pool(ds, cid)
            sc = P3.consensus(p, list(p)) if len(p) >= 2 else None
            base = K.parse_mech(K.cached(f'{ds}/base_default', cid))
            arms = {'Direct': base, 'CoT': K.parse_mech(K.cached(f'{ds}/base_cot', cid)),
                    'SelfConsistency': p.get(sc) if sc else base,
                    'MechSight': relabelled(ds, cid) if K.cached(f'{ds}/RL_MechAgent', cid) else K.parse_mech(M.final(ds, cid, 2))}
            hid, _ = P16.gold_gt_intermediates(case)
            b = '0' if not hid else '1-2' if len(hid) <= 2 else '>=3'
            for a, m in arms.items():
                s = K.score(case, m)
                rows.setdefault(a, []).append((cid, s))
                pooled.setdefault(a, []).append((f'{ds}:{cid}', {k: s.get(k, 0.0) for k in keys}))
                strata.setdefault(b, {}).setdefault(a, []).append((cid, s))
        block = {a: K.summarize(r, keys) for a, r in rows.items()}
        for a in ('CoT', 'SelfConsistency', 'Direct'):
            for k in ('all_present', 'S_partial'):
                block[f'MechSight_minus_{a}_{k}'] = K.paired(rows['MechSight'], rows[a], k)
        block['strata'] = {b: {'n': len(d['Direct']), **{a: round(sum(s['all_present'] for _, s in r) / len(r), 3)
                                                          for a, r in d.items()},
                               'MechSight_minus_Direct': K.paired(d['MechSight'], d['Direct'], 'all_present')}
                           for b, d in sorted(strata.items())}
        rep[ds] = block
    rep['pooled_primary'] = {a: K.summarize(r, keys) for a, r in pooled.items()}
    for a in ('Direct', 'CoT', 'SelfConsistency'):
        for k in ('all_present', 'S_partial'):
            rep['pooled_primary'][f'MechSight_minus_{a}_{k}'] = K.paired(pooled['MechSight'], pooled[a], k)
    (K.OUT / 'SUMMARY_P21_SCALE.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['split', 'pool', 'requests', 'r1', 'r2', 'relabel', 'analyze'], required=True)
    ap.add_argument('--workers', type=int, default=200)
    cli = ap.parse_args()
    s = sets()
    jobs = [(ds, c) for ds, cs in s.items() for c in cs.values()]
    if cli.phase == 'split':
        print({k: len(v) for k, v in s.items()})
    elif cli.phase == 'pool':
        arms = ['base_default', 'base_cot'] + [f'samp{i}' for i in range(1, 5)] + ['enum3']
        K.run_pool(job_pool, [(ds, a, c) for ds, c in jobs for a in arms], cli.workers)
    elif cli.phase == 'requests':
        M.requests(s)
    elif cli.phase in ('r1', 'r2'):
        K.run_pool(M.job_r1 if cli.phase == 'r1' else M.job_r2, jobs, cli.workers)
    elif cli.phase == 'relabel':
        K.run_pool(job_relabel, jobs, cli.workers)
    else:
        analyze(s)
