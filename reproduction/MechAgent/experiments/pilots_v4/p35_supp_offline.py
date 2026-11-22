"""P35: offline supplementary analyses requested in review (2026-11-06; no model calls).

1a  Pool-Vote@7 + Clues on the 2026 literature: species-level majority vote (support >= 4 of the same 7
    proposals that MechSight-Guided uses; >= 3 of 7 as sensitivity), scored like Table 2.
1c  SC@9 for Table 1: MBR-consensus vote (p3_selector.consensus, the Table 1 self-consistency rule) over the
    7-path candidate pool plus the two self-refine rounds; and the same vote restricted to RDKit-parsable paths
    that reach the product.
1t  Token and call accounting per method, from the usage records in the call traces.
2b  Literature results split by whether the clues contain computational content (keyword rule).

Writes OUT/supp_offline_20261106/SUMMARY_SUPP_OFFLINE.json
"""
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict

import common4 as K
import p3_selector as P3
import p10_memory as P10
import p11_mechagent as M
import p28_lit172_v7 as G
import p29_gold_rebuild as GOLD
import p30_lit172_gold as P
import p31_guided71 as T

RUN = K.OUT / 'supp_offline_20261106'
COMP = re.compile(r"\bDFT\b|density functional|computation|computed|calculat|B3LYP|wB97|ωB97|M06|def2|SMD\(|CASSCF|"
                  r"\bNBO\b|\bIRC\b|MECP|transition state|\bTS\d|kcal\s*/?\s*mol|ΔG|Delta G|Gibbs|energy profile|"
                  r"free energy|theoretical|xTB|semi-empirical|\bBDE\b|HOMO|LUMO|spin density", re.I)


def lit_rows(which):
    T.configure(which)
    cs = P.cases()
    P.CS = cs
    rec = P.records()
    ds = P.DS
    lst = [c for c in cs.values() if c['clues'] and any(K.species_keys(s) for s in c['author'])]
    rows = defaultdict(list)
    comp = {}
    for c in lst:
        cid = c['id']
        refs = GOLD.substrate_refs(rec[cid]['record'])
        sup = G.proposals(c)
        a1 = K.cached(f'{ds}/A1_place', cid)
        arms = {'Direct': K.parse_mech(K.cached(f'{ds}/base_default', cid)),
                'Direct_clues': K.parse_mech(K.cached(f'{ds}/O_llm', cid)),
                'MechSight_Guided': K.parse_mech(a1 if a1 is not None else K.cached(f'{ds}/G71_rev', cid)),
                'PoolVote7_maj': [{'intermediate_smiles': s} for s, n in sup.items() if n >= 4],
                'PoolVote7_ge3': [{'intermediate_smiles': s} for s, n in sup.items() if n >= 3]}
        for a, m in arms.items():
            s = T.score(m, c, refs)
            s['complete'] = float(s['recall'] >= 0.999)
            rows[a].append((cid, s))
        comp[cid] = bool(COMP.search(c['clues']))
    return rows, comp


def summarize(rows, keys=('recall', 'complete', 'f1', 'n_pred')):
    return {a: K.summarize(r, list(keys)) for a, r in rows.items()}


def diffs(rows, pairs, keys=('recall', 'complete')):
    return {f'{h}-{l}': {k: K.paired(rows[h], rows[l], k) for k in keys} for h, l in pairs}


def subset(rows, ids):
    return {a: [r for r in rs if r[0] in ids] for a, rs in rows.items()}


def part_1a_2b():
    out = {}
    pooled = defaultdict(list)
    pairs = [('MechSight_Guided', 'PoolVote7_maj'), ('MechSight_Guided', 'Direct_clues'),
             ('PoolVote7_maj', 'Direct_clues'), ('PoolVote7_ge3', 'Direct_clues')]
    split = {}
    for which in ('dev', 'test'):
        rows, comp = lit_rows(which)
        for a, r in rows.items():
            pooled[a] += r
        split.update(comp)
        clean = {c for c, v in comp.items() if not v}
        dirty = {c for c, v in comp.items() if v}
        out[which] = {'n': len(rows['Direct']), 'arms': summarize(rows), 'diffs': diffs(rows, pairs),
                      'no_computational_clues': {'n': len(clean), 'arms': summarize(subset(rows, clean)),
                                                 'diffs': diffs(subset(rows, clean), pairs[1:2] + [('Direct_clues', 'Direct')])},
                      'computational_clues': {'n': len(dirty), 'arms': summarize(subset(rows, dirty)),
                                              'diffs': diffs(subset(rows, dirty), pairs[1:2] + [('Direct_clues', 'Direct')])}}
    clean = {c for c, v in split.items() if not v}
    out['pooled'] = {'n': len(pooled['Direct']), 'arms': summarize(pooled), 'diffs': diffs(pooled, pairs),
                     'no_computational_clues': {'n': len(clean), 'arms': summarize(subset(pooled, clean)),
                                                'diffs': diffs(subset(pooled, clean), pairs[1:2] + [('Direct_clues', 'Direct')])}}
    out['computational_clue_papers'] = sorted(c for c, v in split.items() if v)
    return out


def valid_path(case, m):
    from rdkit import Chem
    ok = all(Chem.MolFromSmiles(s.get('intermediate_smiles') or '') is not None for s in m or [])
    return bool(m) and ok and P3.reaches_product(case, m)


def part_1c():
    out, pooled = {}, defaultdict(list)
    for ds, cases in M.sets(('T3_confirm', 'gold')).items():
        rows = defaultdict(list)
        for cid, case in cases.items():
            if K.cached(f'{ds}/V5_rev1', cid) is None or K.cached(f'{ds}/base_default', cid) is None:
                continue
            p = dict(P10.pool(ds, cid))
            p7 = dict(p)
            for t in ('SR1', 'SR2'):
                m = K.parse_mech(K.cached(f'{ds}/{t}', cid))
                if m:
                    p[t] = m
            base = K.parse_mech(K.cached(f'{ds}/base_default', cid))
            pick = lambda pp, names: pp[P3.consensus(pp, names)] if len(names) >= 2 else base
            valid = [n for n in p if valid_path(case, p[n])]
            arms = {'Direct': base, 'SC_pool7': pick(p7, list(p7)), 'SC9': pick(p, list(p)),
                    'SC9_rdkit_valid': pick(p, valid) if len(valid) >= 2 else (p[valid[0]] if valid else pick(p, list(p))),
                    'MechSight_Auto': K.parse_mech(M.final(ds, cid, 2)), 'n_paths': None}
            arms.pop('n_paths')
            for a, m in arms.items():
                s = K.score(case, m)
                rows[a].append((cid, {'all_present': s['all_present']}))
                pooled[a].append((f'{ds}:{cid}', {'all_present': s['all_present']}))
            rows['_paths'].append((cid, {'all_present': float(len(p))}))
        n_paths = [m['all_present'] for _, m in rows.pop('_paths')]
        out[ds] = {'n': len(rows['Direct']), 'mean_paths_in_SC9': round(statistics.mean(n_paths), 2),
                   'arms': {a: K.summarize(r, ['all_present']) for a, r in rows.items()},
                   'diffs': {f'MechSight_Auto-{a}': K.paired(rows['MechSight_Auto'], rows[a], 'all_present')
                             for a in ('Direct', 'SC_pool7', 'SC9', 'SC9_rdkit_valid')}}
    out['pooled'] = {'n': len(pooled['Direct']), 'arms': {a: K.summarize(r, ['all_present']) for a, r in pooled.items()},
                     'diffs': {f'MechSight_Auto-{a}': K.paired(pooled['MechSight_Auto'], pooled[a], 'all_present')
                               for a in ('Direct', 'SC_pool7', 'SC9', 'SC9_rdkit_valid')}}
    return out


def trace_usage(ds, tag):
    """Per-paper (calls, prompt, completion, reasoning) from successful events in OUT/<ds>/<tag>/traces."""
    d = K.OUT / ds / tag / 'traces'
    res = {}
    if not d.exists():
        return res
    for f in d.glob('*.jsonl'):
        calls = pt = ct = rt = 0
        for line in f.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = e.get('usage')
            if not u:
                continue
            calls += 1
            pt += u.get('prompt_tokens') or 0
            ct += u.get('completion_tokens') or 0
            rt += ((u.get('completion_tokens_details') or {}).get('reasoning_tokens') or 0)
        if calls:
            res[f.stem] = (calls, pt, ct, rt)
    return res


LIT_METHODS = {
    'Direct': ['base_default'],
    'Direct_clues': ['O_llm'],
    'MechSight_Auto': ['base_default', 'samp1', 'samp2', 'samp3', 'samp4', 'enum3', 'V5_comp1', 'V5_rev1', 'V5_comp2', 'V5_rev2'],
    'MechSight_Guided': ['base_default', 'samp1', 'samp2', 'samp3', 'samp4', 'enum3', 'V5_comp1', 'V5_rev1', 'V5_comp2',
                         'V5_rev2', 'O_llm', 'G7L_clue1', 'G7L_clue2', 'G7L_hyp0', 'G7L_hyp1', 'G7L_hyp2', 'G7L_hyp3',
                         'G71_rev', 'A1_place'],
    'SC15_clues': ['O_llm'] + [f'G7L_clue{k}' for k in range(1, 15)] + ['SC15_rev'],
}
T1_METHODS = {
    'Direct': ['base_default'], 'CoT': ['base_cot'], 'SelfRefine': ['base_default', 'SR1', 'SR2'],
    'SelfConsistency_pool': ['base_default', 'samp1', 'samp2', 'samp3', 'samp4', 'enum3'],
    'MechSight_Auto': ['base_default', 'samp1', 'samp2', 'samp3', 'samp4', 'enum3', 'V5_comp1', 'V5_rev1', 'V5_comp2', 'V5_rev2'],
}


def account(ds_list, methods):
    out = {}
    for meth, tags in methods.items():
        per = defaultdict(lambda: [0, 0, 0, 0])
        tag_cov = {}
        for ds in ds_list:
            for tag in tags:
                u = trace_usage(ds, tag)
                tag_cov[f'{ds}/{tag}'] = len(u)
                for cid, v in u.items():
                    acc = per[f'{ds}:{cid}']
                    for i in range(4):
                        acc[i] += v[i]
        if not per:
            continue
        vals = list(per.values())
        mean = lambda i: round(statistics.mean(v[i] for v in vals), 1)
        out[meth] = {'papers_with_traces': len(vals), 'calls_mean': mean(0), 'prompt_tokens_mean': mean(1),
                     'completion_tokens_mean': mean(2), 'reasoning_tokens_mean': mean(3), 'trace_coverage': tag_cov}
    return out


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    rep = {'note': 'offline, no model calls; post hoc analyses requested in review (not pre-registered)',
           '1a_2b_literature': part_1a_2b(), '1c_table1_SC9': part_1c(),
           '1t_tokens': {'literature_dev': account(['lit172g'], LIT_METHODS),
                         'literature_confirm': account(['litnew_g'], LIT_METHODS),
                         'table1_T3_confirm_and_oMe_Gold': account(['T3_confirm', 'gold'], T1_METHODS)}}
    (RUN / 'SUMMARY_SUPP_OFFLINE.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rep.items() if k != '1a_2b_literature'}, indent=1)[:6000])
    lit = rep['1a_2b_literature']
    for w in ('dev', 'test', 'pooled'):
        print('==', w, 'n', lit[w]['n'])
        for a, v in lit[w]['arms'].items():
            print(f"  {a:18s} R {v['recall']}  complete {v['complete']}  n_pred {v['n_pred'][0]}")
        for d, v in lit[w]['diffs'].items():
            print(f"  {d:36s} R {v['recall']}  complete {v['complete']}")
        nc = lit[w]['no_computational_clues']
        print('  no-comp subset n', nc['n'], {d: v['recall'] for d, v in nc['diffs'].items()})
        if 'computational_clues' in lit[w]:
            cc = lit[w]['computational_clues']
            print('  comp subset n', cc['n'], {d: v['recall'] for d, v in cc['diffs'].items()})


if __name__ == '__main__':
    main()
