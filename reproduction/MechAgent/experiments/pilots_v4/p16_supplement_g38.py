"""P16: Supplementary gemini-3.8-flash experiments for the MechSight manuscript storyline (2026-11-04).

1. 2026 Post-Cutoff Literature (lit, 29 cases):
   - Completes candidate pool (base_default from A0, samp4, enum3)
   - Runs RXNMapper stepmap gap detection
   - Runs clue-free MechSight v5 (V5_comp1 -> V5_rev1 -> V5_comp2 -> V5_rev2) and M_ms_none
   - Runs clue-guided MechSight v5 (V5_obs: combines V5 gap-completed states + experimental clues)
   - Evaluates the 3-tier ladder: Direct (A0) -> MechSight (clue-free) -> MechSight + Clues (O_ms / V5_obs)

2. oMe-Gold (gold, 196 cases) Oracle-hidden & Oracle-visible controls:
   - Partitions GT intermediates into hidden (distinct heavy-atom sigma skeleton from endpoints)
     vs visible (same heavy-atom sigma skeleton as endpoints, e.g. proton/electron transfer states)
   - Runs O_hid and O_vis with the exact P12 ORACLE prompt

3. In-template benchmark (uspto31k, 201 cases):
   - Runs pool (base_cot, samp1..4, enum3), stepmap, M_ms_none, and MechSight v5
"""
from __future__ import annotations

import argparse
import json
import shutil

import common4 as K
import p2_enumerate as P2
import p3_selector as P3
import p6_baselines as P6
import p8_observations as P8
import p9_literature as P9
import p10_memory as P10
import p11_mechagent as M
import p12_controls as P12
import stepmap as SMAP

V5_OBS_REVISE = (
    'Reactants and reagents (SMILES): {reactants}\nProducts (SMILES): {products}\nReaction conditions: {conditions}\n\n'
    'Experimental observations from mechanistic control experiments on this reaction:\n{obs}\n\n'
    'Candidate mechanisms proposed for this reaction (intermediate SMILES after each step; steps flagged by an '
    'atom-mapping check change several bonds at once):\n\n{cands}\n\nIntermediates proposed for the flagged steps '
    '(valid structures; not all of them need to be right):\n{states}\n\nFirst use the experimental observations and '
    'elementary-step soundness to discard candidates that miss or contradict the evidence or skip intermediates. '
    'Then write the final mechanism: keep the most plausible candidate or repair it, incorporating both the '
    'experimentally supported intermediates and the chemically justified gap-completing intermediates. Return the '
    'final mechanism between [ANSWER] and [/ANSWER] as a JSON list of steps, each '
    '{{"step": n, "type": ..., "subtype": ..., "intermediate_smiles": ...}}.'
)


# ------------------------------------------------------------------ 1. Literature (2026 OA)
def lit_cases():
    pids = sorted(p.name for p in P9.PAPERS.iterdir() if (p / 'main.pdf').exists())
    return {p: c for p in pids if (c := P9.case_of(p))}


def sync_lit_base():
    src_dir = K.OUT / 'lit' / 'A0'
    dst_dir = K.OUT / 'lit' / 'base_default'
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in src_dir.glob('*.json'):
        dst = dst_dir / f.name
        if not dst.exists():
            shutil.copy2(f, dst)


def job_lit_pool(arm, case):
    if arm == 'samp4':
        return 'ok' if K.call('lit/samp4', case['id'], K.official_prompt(case), temperature=0.7) else 'error'
    if arm == 'enum3':
        return P2.job('lit', case)
    return 'skip'


def job_lit_v5_obs(case):
    if not case.get('clues'):
        return 'noclues'
    cid = case['id']
    cands = M.cand_list('lit', cid)
    if len(cands) < 2:
        return 'skip'
    text = M.render(case, cands)
    comp = K.cached('lit/V5_comp1', cid)
    st = M.checked_states(case, comp) if comp else []
    states = '\n'.join(f'- ({loc}) {s}' for loc, s in st) or '- none'
    prompt = V5_OBS_REVISE.format(
        cands=text, states=states, obs=case['clues'], **M.fields(case)
    )
    return 'ok' if K.call('lit/V5_obs', cid, prompt) else 'error'


def analyze_lit():
    cases = lit_cases()
    arms = ('A0', 'M_ms_none', 'V5_r1', 'V5', 'O_llm', 'O_ms', 'V5_obs')
    rows = {a: [] for a in arms}
    per_paper = []
    for pid, c in sorted(cases.items()):
        akeys = [k for s in c['author'] if (k := K.species_keys(s))]
        if not akeys:
            continue
        rec_row = {'id': pid, 'n_author_intermediates': len(akeys), 'has_clues': bool(c.get('clues'))}
        for arm in arms:
            if arm == 'V5_r1':
                ans = M.final('lit', pid, 1)
            elif arm == 'V5':
                ans = M.final('lit', pid, 2)
            else:
                ans = K.cached(f'lit/{arm}', pid)
            m = K.parse_mech(ans)
            if m is None:
                continue
            pk = {k for st in m for k in K.species_keys(st.get('intermediate_smiles') or '')}
            hits = sum(bool(k & pk) for k in akeys)
            rec = hits / len(akeys)
            all_p = float(hits == len(akeys))
            rows[arm].append((pid, {'author_intermediate_recall': rec, 'all_present': all_p}))
            rec_row[arm] = {'recall': round(rec, 3), 'hits': hits, 'all_present': bool(all_p)}
        per_paper.append(rec_row)

    rep = {a: K.summarize(r, ['author_intermediate_recall', 'all_present']) for a, r in rows.items() if r}
    for a in ('M_ms_none', 'V5_r1', 'V5', 'O_llm', 'O_ms', 'V5_obs'):
        if rows[a] and rows['A0']:
            rep[f'{a}_minus_A0_recall'] = K.paired(rows[a], rows['A0'], 'author_intermediate_recall')
            rep[f'{a}_minus_A0_all_present'] = K.paired(rows[a], rows['A0'], 'all_present')
    for a in ('O_ms', 'V5_obs'):
        if rows[a] and rows['O_llm']:
            rep[f'{a}_minus_O_llm_recall'] = K.paired(rows[a], rows['O_llm'], 'author_intermediate_recall')
        if rows[a] and rows['V5']:
            rep[f'{a}_minus_V5_recall'] = K.paired(rows[a], rows['V5'], 'author_intermediate_recall')
    (K.OUT / 'SUMMARY_P9_V5.json').write_text(json.dumps({'summary': rep, 'per_paper': per_paper}, indent=1))
    print(json.dumps(rep, indent=1))


# ------------------------------------------------------------------ 2. oMe-Gold Oracle Controls
def gold_gt_intermediates(case):
    """Partition oMe-Gold GT intermediates into hidden (distinct heavy-atom sigma skeleton) vs visible."""
    end_skel = {s for x in case['reactants'] + case['products'] for _, s in K.P.smiles_keys(x)}
    end_key = {k for x in case['reactants'] + case['products'] for k, _ in K.P.smiles_keys(x)}
    hid, vis, seen = [], [], set()
    for st in case['gold'][:-1]:
        smi = st.get('intermediate_smiles') or ''
        ks = K.P.smiles_keys(smi)
        if not ks:
            continue
        kset = {k for k, _ in ks} - end_key
        if not kset:
            continue
        sig = tuple(sorted(kset))
        if sig in seen:
            continue
        seen.add(sig)
        sset = {s for _, s in ks}
        if not (sset <= end_skel):
            hid.append(smi)
        else:
            vis.append(smi)
    return hid, vis


def job_gold_oracle(case):
    hid, vis = gold_gt_intermediates(case)
    if not hid:
        return 'nohidden'
    for tag, sp in (('O_hid', hid), ('O_vis', vis)):
        if sp:
            p = K.official_prompt(case).replace(
                '\n\n## Input', P12.ORACLE.format(species='\n'.join(f'- {s}' for s in sp)), 1
            )
            K.call(f'gold/{tag}', case['id'], p)
    return 'ok'


def analyze_gold_oracle():
    import p14_confirm as P14
    gold = K.load_gold()
    rows = {a: [] for a in ('Direct', 'MechAgent', 'O_hid', 'O_vis')}
    strata = {}
    for cid, case in gold.items():
        base = K.cached('gold/base_default', cid)
        if base is None:
            continue
        hid, vis = gold_gt_intermediates(case)
        ans = P14.answers('gold', cid, case)
        ans['O_hid'] = K.parse_mech(K.cached('gold/O_hid', cid)) if hid else ans['Direct']
        ans['O_vis'] = (K.parse_mech(K.cached('gold/O_vis', cid)) if vis else ans['Direct']) if hid else ans['Direct']
        b = '0' if not hid else ('1-2' if len(hid) <= 2 else '>=3')
        for a in ('Direct', 'MechAgent', 'O_hid', 'O_vis'):
            sc = K.score(case, ans[a])
            rows[a].append((cid, sc))
            strata.setdefault(b, {}).setdefault(a, []).append(sc['all_present'])
    keys = ['all_present', 'in_order_frac', 'S_partial', 'in_order_full']
    rep = {a: K.summarize(r, keys) for a, r in rows.items() if r}
    for a in ('MechAgent', 'O_hid', 'O_vis'):
        rep[f'{a}_minus_Direct_all_present'] = K.paired(rows[a], rows['Direct'], 'all_present')
        rep[f'{a}_minus_Direct_S_partial'] = K.paired(rows[a], rows['Direct'], 'S_partial')
    rep['strata_all_present'] = {
        b: {a: [len(v), round(sum(v) / len(v), 3)] for a, v in d.items()} for b, d in sorted(strata.items())
    }
    (K.OUT / 'SUMMARY_GOLD_ORACLE.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


# ------------------------------------------------------------------ 3. USPTO-31k (In-Template)
def job_uspto_pool(arm, case):
    if arm == 'enum3':
        return P2.job('uspto31k', case)
    return P6.job('uspto31k', arm, case)


def analyze_uspto():
    cases = K.load_uspto()
    arms = ('Direct', 'CoT', 'SelfConsistency', 'M_ms_none', 'MechAgent')
    rows = {a: [] for a in arms}
    for cid, case in cases.items():
        base = K.cached('uspto31k/base_default', cid)
        if base is None or K.cached('uspto31k/V5_rev1', cid) is None:
            continue
        p = P10.pool('uspto31k', cid)
        sc_pick = P3.consensus(p, list(p)) if len(p) >= 2 else None
        ans = {
            'Direct': K.parse_mech(base),
            'CoT': K.parse_mech(K.cached('uspto31k/base_cot', cid)),
            'SelfConsistency': p.get(sc_pick) if sc_pick else K.parse_mech(base),
            'M_ms_none': K.parse_mech(K.cached('uspto31k/M_ms_none', cid)),
            'MechAgent': K.parse_mech(M.final('uspto31k', cid, 2)),
        }
        for a, m in ans.items():
            rows[a].append((cid, K.score(case, m)))
    keys = ['all_present', 'in_order_full', 'in_order_frac']
    rep = {a: K.summarize(r, keys) for a, r in rows.items() if r}
    for a in ('CoT', 'SelfConsistency', 'M_ms_none', 'MechAgent'):
        if rows[a]:
            rep[f'{a}_minus_Direct_all_present'] = K.paired(rows[a], rows['Direct'], 'all_present')
            rep[f'{a}_minus_Direct_in_order_full'] = K.paired(rows[a], rows['Direct'], 'in_order_full')
    (K.OUT / 'SUMMARY_USPTO_V5.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--task',
        choices=[
            'lit_pool', 'lit_requests', 'lit_r1', 'lit_r2', 'lit_obs', 'lit_analyze',
            'gold_oracle', 'gold_oracle_analyze',
            'uspto_pool', 'uspto_requests', 'uspto_r1', 'uspto_r2', 'uspto_analyze',
        ],
        required=True,
    )
    ap.add_argument('--workers', type=int, default=300)
    cli = ap.parse_args()

    if cli.task == 'lit_pool':
        sync_lit_base()
        cs = lit_cases()
        K.run_pool(job_lit_pool, [(a, c) for a in ('samp4', 'enum3') for c in cs.values()], cli.workers)
    elif cli.task == 'lit_requests':
        sync_lit_base()
        M.requests({'lit': lit_cases()})
    elif cli.task == 'lit_r1':
        cs = lit_cases()
        K.run_pool(M.job_none, [('lit', c) for c in cs.values()], cli.workers)
        K.run_pool(M.job_r1, [('lit', c) for c in cs.values()], cli.workers)
    elif cli.task == 'lit_r2':
        cs = lit_cases()
        K.run_pool(M.job_r2, [('lit', c) for c in cs.values()], cli.workers)
    elif cli.task == 'lit_obs':
        cs = lit_cases()
        K.run_pool(job_lit_v5_obs, [(c,) for c in cs.values()], cli.workers)
    elif cli.task == 'lit_analyze':
        analyze_lit()
    elif cli.task == 'gold_oracle':
        cs = K.load_gold()
        K.run_pool(job_gold_oracle, [(c,) for c in cs.values()], cli.workers)
        analyze_gold_oracle()
    elif cli.task == 'gold_oracle_analyze':
        analyze_gold_oracle()
    elif cli.task == 'uspto_pool':
        cs = K.load_uspto()
        arms = ['base_cot'] + [f'samp{i}' for i in range(1, 5)] + ['enum3']
        K.run_pool(job_uspto_pool, [(a, c) for a in arms for c in cs.values()], cli.workers)
    elif cli.task == 'uspto_requests':
        M.requests({'uspto31k': K.load_uspto()})
    elif cli.task == 'uspto_r1':
        cs = K.load_uspto()
        K.run_pool(M.job_none, [('uspto31k', c) for c in cs.values()], cli.workers)
        K.run_pool(M.job_r1, [('uspto31k', c) for c in cs.values()], cli.workers)
    elif cli.task == 'uspto_r2':
        cs = K.load_uspto()
        K.run_pool(M.job_r2, [('uspto31k', c) for c in cs.values()], cli.workers)
    elif cli.task == 'uspto_analyze':
        analyze_uspto()
