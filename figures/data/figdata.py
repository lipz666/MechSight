"""Every number shown in Figs. 1-3 (MechSight manuscript), read from frozen result files. Writes data/figdata.json.

Values that are NOT read from a run file are marked 'quoted' (published numbers) and listed in data/PROVENANCE.md.
"""
import json
import os
from pathlib import Path

R = Path(os.environ.get('MECHSIGHT_ROOT', '.'))
O = R / 'MechAgent/outputs'
V4 = O / 'pilots_v4_20261103'
PK = R / 'MechAgent_Storyline_Experiments_Package'
HERE = Path(__file__).parent
J = lambda p: json.loads(Path(p).read_text())
pc = lambda x: [round(100 * v + 1e-9, 1) for v in x] if isinstance(x, list) else round(100 * x + 1e-9, 1)

c14 = J(V4 / 'SUMMARY_P14_confirm.json'); t3, gold, pool = c14['T3_confirm'], c14['gold'], c14['pooled_primary']
go = J(V4 / 'SUMMARY_GOLD_ORACLE.json')
po = J(HERE / 'pooled_oracle.json')
spec = J(V4 / 'SUMMARY_P18_SPEC.json')['T3_confirm']
argon = J(V4 / 'SUMMARY_ARGON_CONFIRM.json')
lit = J(V4 / 'SUMMARY_P9_V5.json')
con = J(O / 'contamination_20261103/SUMMARY.json')
p5 = J(V4 / 'SUMMARY_P5.json')['b_J3_mechanism']
mf = J(PK / '01_Problem_Template_Degeneration/outputs/summary.json')['arms']
pub = J(PK / '01_Problem_Template_Degeneration/baseline_comparison/publication_exp1_summary.json')['arms']
cov = J(PK / '01_Problem_Template_Degeneration/outputs/fukuyama_coverage.json')['summary']['all']
fl = J(PK / '01_Problem_Template_Degeneration/baseline_comparison/flower_template_summary.json')
cmp1a = J(PK / '02_LLM_vs_FlowER_FukuyamaBench/outputs/COMPARE_1a.json')

ARMS = ['Direct', 'CoT', 'SelfRefine', 'SelfConsistency', 'MechAgent']
d = {}

# same-metric specialist comparison (all 444 reactions run; resonance/tautomer-tolerant species keys)
sp = J(V4 / 'SUMMARY_P18_SPEC_stdinchi.json')
_u = sp['uspto31k']['FlowER']          # 196 mech-USPTO reactions completed by FlowER (ChRIMP and LLM on the same set)
_t = sp['T3_confirm']['FlowER']        # 48/48
_g = sp['gold']['FlowER']              # 196/196
spec_same = dict(
    n_in=_u['coverage'], n_t3=_t['coverage'], n_gold=_g['coverage'],
    **{arm: dict(inn=pc(_u[key]['all_present']) if key in _u else None,
                 t3=pc(_t[key]['all_present']), gold=pc(_g[key]['all_present']),
                 reach_in=pc(_u[key]['reached'][0]) if key in _u else None,
                 reach_t3=pc(_t[key]['reached'][0]), reach_gold=pc(_g[key]['reached'][0]))
       for arm, key in (('chrimp', 'ChRIMP'), ('flower', 'FlowER'), ('direct', 'LLM_direct'), ('mechsight', 'MechAgent'))})

# ---------------- Fig. 1
d['fig1'] = dict(
    spec=spec_same,
    template=dict(
        retrieval=dict(inn=pc(mf['retrieval']['levels']['L1_transfer_set']['rate']), inn_ci=pc(mf['retrieval']['levels']['L1_transfer_set']['ci95']),
                       n_in=80, out=0.0, out_note=f"0/{cov['cases']} covered"),
        flower=dict(inn=88.8, inn_quoted=True, inn_rerun=pc(fl['pathway_top1_exact']), n_in=fl['sample'],
                    out=pc(spec['FlowER']['FlowER']['all_present'][0]), n_out=spec['FlowER']['coverage'],
                    diag=f"0/{cmp1a['dev_24']['cases_with_flower_result'] + cmp1a['heldout_primary_123']['cases_with_flower_result']}"),
        chrimp=dict(out=pc(spec['ChRIMP']['ChRIMP']['all_present'][0]), out_ci=pc(spec['ChRIMP']['ChRIMP']['all_present']), n_out=spec['ChRIMP']['coverage']),
        direct=dict(inn=pc(pub['direct']['levels']['L1_transfer_set']['rate']), n_in=80, out=pc(t3['Direct']['all_present'][0]),
                    out_ci=pc(t3['Direct']['all_present']), n_out=48),
        mechsight=dict(out=pc(t3['MechAgent']['all_present'][0]), out_ci=pc(t3['MechAgent']['all_present']), n_out=48)),
    homolog=[dict(lab='product', model='gemini', o=pc(con['gemini']['original'][0]), h=pc(con['gemini']['homologue'][0]), n=109),
             dict(lab='product', model='gpt-6-luna', o=pc(con['gpt-6-luna']['original'][0]), h=pc(con['gpt-6-luna']['homologue'][0]), n=109),
             dict(lab='hidden recall', model='gemini', o=pc(p5['original']['heavy_hidden_recall'][0]), h=pc(p5['homologue']['heavy_hidden_recall'][0]), n=74)],
    strata={'T3': {k: dict(n=v['Direct'][0], v=pc(v['Direct'][1])) for k, v in t3['strata_all_present'].items()},
            'oMe': {k: dict(n=v['Direct'][0], v=pc(v['Direct'][1])) for k, v in go['strata_all_present'].items()}},
    # inert_knowledge_20261102/REPORT.md (pre-registered, gemini, PGL): 42 missed heavy-atom hidden events
    inert=dict(n_events=42, inert=57.1, inert_ci=[42.5, 72.2], resample=7.1, resample_ci=[0.0, 18.6]),
    supply={'T3': dict(Direct=pc(t3['Direct']['all_present']), vis=pc(t3['Oracle_visible']['all_present']), hid=pc(t3['Oracle_hidden']['all_present']),
                       d_vis=pc(t3['Oracle_visible_minus_Direct']), d_hid=pc(t3['Oracle_hidden_minus_Direct']), n=48),
            'oMe': dict(Direct=pc(go['Direct']['all_present']), vis=pc(go['O_vis']['all_present']), hid=pc(go['O_hid']['all_present']),
                        d_vis=pc(go['O_vis_minus_Direct_all_present']), d_hid=pc(go['O_hid_minus_Direct_all_present']), n=196)})

# ---------------- Fig. 2
def grp(src, extra):
    g = {a: pc(src[a]['all_present']) for a in ARMS}
    g.update(extra)
    return g
d['fig2'] = dict(
    bench={'T3': grp(t3, {'vis': pc(t3['Oracle_visible']['all_present']), 'hid': pc(t3['Oracle_hidden']['all_present'])}),
           'oMe': grp(gold, {'vis': pc(go['O_vis']['all_present']), 'hid': pc(go['O_hid']['all_present'])}),
           'pooled': grp(pool, {'vis': pc(po['O_vis']['all_present']), 'hid': pc(po['O_hid']['all_present'])})},
    n={'T3': 48, 'oMe': 196, 'pooled': 244},
    gain={'T3': pc(t3['MechAgent_minus_Direct']), 'oMe': pc(gold['MechAgent_minus_Direct']), 'pooled': pc(pool['MechAgent_minus_Direct'])},
    recovered=round(100 * (t3['MechAgent']['all_present'][0] - t3['Direct']['all_present'][0]) /
                    (t3['Oracle_hidden']['all_present'][0] - t3['Direct']['all_present'][0])),
    strata={'T3': {k: dict(n=v['Direct'][0], Direct=pc(v['Direct'][1]), Auto=pc(v['MechAgent'][1]), hid=pc(v['Oracle_hidden'][1]))
                   for k, v in t3['strata_all_present'].items()},
            'oMe': {k: dict(n=v['Direct'][0], Direct=pc(v['Direct'][1]), Auto=pc(v['MechAgent'][1]), hid=pc(v['O_hid'][1]))
                    for k, v in go['strata_all_present'].items()}},
    gapflag=dict(flagged=29.9, n_flagged=77, clear=56.0, n_clear=116))

# ---------------- Fig. 3
at = argon['T3_confirm']
ag = argon.get('gold')
L = lit['summary']
pp = {p['id']: p for p in lit['per_paper']}
d['fig3'] = dict(
    argon={'T3': dict(Direct=pc(at['Direct']['all_present']), SC=pc(at['SelfConsistency']['all_present']), Auto=pc(at['MechAgent']['all_present']),
                      d_auto=pc(at['MechAgent_minus_Direct']), d_auto_sc=pc(at['MechAgent_minus_SelfConsistency']), n=48,
                      strata={k: {a: pc(v[a][1]) for a in ('Direct', 'SelfConsistency', 'MechAgent')} | {'n': v['Direct'][0]}
                              for k, v in at['strata_all_present'].items()}),
           'oMe': dict(Direct=pc(ag['Direct']['all_present']), SC=pc(ag['SelfConsistency']['all_present']), Auto=pc(ag['MechAgent']['all_present']),
                       d_auto=pc(ag['MechAgent_minus_Direct']), d_auto_sc=pc(ag['MechAgent_minus_SelfConsistency']), n=ag['Direct']['n']) if ag else None,
           'pooled': dict(Direct=pc(argon['pooled_primary']['Direct']['all_present']),
                          SC=pc(argon['pooled_primary']['SelfConsistency']['all_present']),
                          Auto=pc(argon['pooled_primary']['MechAgent']['all_present']),
                          d_auto=pc(argon['pooled_primary']['MechAgent_minus_Direct']),
                          n=argon['pooled_primary']['Direct']['n']) if 'pooled_primary' in argon else None},
    lit=dict(n=L['A0']['n'], arms={a: dict(recall=pc(L[a]['author_intermediate_recall']), all=pc(L[a]['all_present']), n=L[a]['n'])
                                   for a in ('A0', 'O_llm', 'V5', 'V5_obs')},
             d_recall={a: pc(L[f'{a}_minus_A0_recall']) for a in ('O_llm', 'V5', 'V5_obs')},
             d_all={a: pc(L[f'{a}_minus_A0_all_present']) for a in ('O_llm', 'V5', 'V5_obs')}),
    cases={pid: {a: pp[pid][a] for a in ('A0', 'V5', 'V5_obs') if a in pp[pid]} | {'n': pp[pid]['n_author_intermediates']}
           for pid in ('PMC13530165', 'PMC13036046')})

# ---------------- New Blueprint Data (M3, M2, Decomp251, Ladder, Lit165, Blind76, SI)
m3 = J(V4 / 'm3_counterfactual_20261106/SUMMARY_M3.json')['summary']
m2 = J(V4 / 'm2_exposure_20261106/SUMMARY_M2.json')['summary']
g71 = J(V4 / 'g71_dev_20261106/SUMMARY_G71.json')['summary']
lit172 = J(V4 / 'lit172_gold_argon_20261105/SUMMARY_LIT172_GOLD_ARGON.json')['summary']
lad = J(V4 / 'SUMMARY_P19_LADDER.json')
off = J(V4 / 'SUMMARY_P23_OFFLINE.json')
blind76 = J(V4 / 'SUMMARY_P27_G7_CONFIRM.json')
randsmi = J(V4 / 'SUMMARY_P22_RANDSMI.json')

d['fig1']['m3'] = dict(
    n_triplets=m3['analyzed'], n_reactions=m3['analyzed'] * 3, n_unseen=m3['novel_both'],
    agree=pc(m3['judge_agreement']),
    parent=m3['kstar_use']['parent'][:3],
    preserve=m3['kstar_use']['preserve'][:3],
    brk=m3['kstar_use']['break'][:3],
    d_all=m3['PRIMARY_break_minus_preserve'][:3],
    d_unseen=m3['novel_only_break_minus_preserve'][:3],
    d_H=m3['exploratory_by_requirement']['H']['break_minus_preserve'][:3],
    d_group=m3['exploratory_by_requirement']['group']['break_minus_preserve'][:3],
    ghost=m3['ghost_any']['break'][:3],
    ghost_smarts=m3['ghost_smarts_break'][:3],
)
d['fig1']['m2'] = dict(
    low=dict(n=85, Direct=round(m2['means']['low']['Direct_strict'], 1), Auto=round(m2['means']['low']['Auto_strict'], 1)),
    mid=dict(n=83, Direct=round(m2['means']['mid']['Direct_strict'], 1), Auto=round(m2['means']['mid']['Auto_strict'], 1)),
    high=dict(n=76, Direct=round(m2['means']['high']['Direct_strict'], 1), Auto=round(m2['means']['high']['Auto_strict'], 1)),
    new2026=dict(n=165, Direct=round(m2['means']['recent']['Direct_strict'], 1), Auto=round(m2['means']['recent']['Auto_strict'], 1)),
    d_adj=[round(x, 1) for x in m2['P1_high_minus_recent_Direct_strict_adj']['est_ci_pp']],
)
d['fig1']['decomp251'] = dict(
    n_missed=251, n_inert=129, pct_inert=51.4, n_inert_recovered=77, pct_inert_recovered=59.7,
    n_blind=120, pct_blind=47.8, n_stage2_new=2, pct_stage2_new=0.8,
)
d['fig2']['ladder'] = dict(
    pooled={k: pc(lad['pooled'][k]['all_present']) for k in ('L0_Direct', 'L1_sampling_revision', 'L2_plus_enumeration', 'L3_MechSight')},
    t3={k: pc(lad['T3_confirm'][k]['all_present']) for k in ('L0_Direct', 'L1_sampling_revision', 'L2_plus_enumeration', 'L3_MechSight')},
    gold={k: pc(lad['gold'][k]['all_present']) for k in ('L0_Direct', 'L1_sampling_revision', 'L2_plus_enumeration', 'L3_MechSight')},
)
d['fig2']['precision'] = dict(
    t3_dir=round(100 * off['A_overgeneration']['T3_confirm']['Direct']['precision'], 1),
    t3_auto=round(100 * off['A_overgeneration']['T3_confirm']['MechSight']['precision'], 1),
    t3_inord_dir=off['A_overgeneration']['T3_confirm']['Direct']['strict_in_order'],
    t3_inord_auto=off['A_overgeneration']['T3_confirm']['MechSight']['strict_in_order'],
    gold_dir=round(100 * off['A_overgeneration']['gold']['Direct']['precision'], 1),
    gold_auto=round(100 * off['A_overgeneration']['gold']['MechSight']['precision'], 1),
    gold_inord_dir=off['A_overgeneration']['gold']['Direct']['strict_in_order'],
    gold_inord_auto=off['A_overgeneration']['gold']['MechSight']['strict_in_order'],
)
# 2026-11-06: MechSight-Guided = A1 (frozen). All Table 2 dev arms come from one scorer (p31 T.score, the scorer
# pre-registered for the confirmation run), so bars and brackets are consistent: TABLE2_DEV_A1.json (offline, 0 calls).
t2 = J(V4 / 'p34b_a1_38_20261106/TABLE2_DEV_A1.json')
t2s, t2p = t2['summary'], t2['per_paper']
_arm = lambda k: dict(recall=pc(t2s['arms'][k]['recall']), path=pc(t2s['arms'][k]['complete']), f1=pc(t2s['arms'][k]['f1']))
_d = lambda h, l, m='recall': pc(t2s['diffs'][f'{h}-{l}'][m])
d['fig3']['lit165'] = dict(
    n_dev=165, n_confirm=271, n_total=436, n_gold_int=920,
    Direct=_arm('Direct'), Auto=_arm('MechSight_Auto'), DirectClues=_arm('Direct_clues'), SC15Clues=_arm('SC15'),
    Guided=_arm('MechSight_Guided_A1'),
    d_auto_dir=_d('MechSight_Auto', 'Direct'), d_clues_dir=_d('Direct_clues', 'Direct'),
    d_guided_dir=_d('MechSight_Guided_A1', 'Direct'), d_guided_clues=_d('MechSight_Guided_A1', 'Direct_clues'),
    d_guided_sc15=_d('MechSight_Guided_A1', 'SC15'),
    d_guided_clues_path=_d('MechSight_Guided_A1', 'Direct_clues', 'complete'),
)
# Fig. 3c bottom case (2026-11-06): PMC13321810, Chen et al., Chem. Sci. 2026, 17, 15443 (dev set, argon gold, 2 author
# intermediates). Hits per arm = strict recall x n from the same per-paper scores.
_pp = {a: dict(rows) for a, rows in t2p.items()}
d['fig3']['case_guided'] = dict(pid='PMC13321810', n=2, **{
    k: round(_pp[a]['PMC13321810']['recall'] * 2) for k, a in
    (('Direct', 'Direct'), ('Auto', 'MechSight_Auto'), ('DirectClues', 'Direct_clues'), ('Guided', 'MechSight_Guided_A1'))})
# Fig. 3c top case (2026-11-06 review): PMC13171995, Nat. Commun. 2026 (doi 10.1038/s41467-026-70618-6), EDA-photocatalysed
# hydrosulfonylation; dev set, argon gold, 5 author intermediates; hits = strict recall x n (same scorer as Table 2).
d['fig3']['case_auto'] = dict(pid='PMC13171995', n=5, **{
    k: round(_pp[a]['PMC13171995']['recall'] * 5) for k, a in (('Direct', 'Direct'), ('Auto', 'MechSight_Auto'))})
d['fig3']['blind76'] = dict(
    n=76,
    Direct=[0.0, 0.0, 0.0],
    Auto=[0.0, 0.0, 0.0],
    DirectClues=[50.0, 38.2, 60.5],
    Guided=[56.6, 46.1, 67.1],
    Guided_n74=pc(blind76['pooled']['Guided_v7']['blind_species_written']),
)

# Pre-registered literature confirmation set (n = 271, MechSight-Guided = A1): recall from the summary; complete pathway
# = share of papers with every author intermediate recalled (recall == 1), the same definition as TABLE2_DEV_A1
# 'complete' (checked: identical on the dev set); 95 % CI by paper-level bootstrap (2000 resamples, seed 20261106).
g71c = J(V4 / 'litnew_confirm_g71/SUMMARY_G71.json')
def _boot_complete(rows, n_boot=2000, seed=20261106):
    import random
    v = [1.0 if r['recall'] >= 1 - 1e-9 else 0.0 for _, r in rows]
    rng = random.Random(seed); m = sum(v) / len(v)
    bs = sorted(sum(rng.choice(v) for _ in v) / len(v) for _ in range(n_boot))
    return [pc(m), pc(bs[int(0.025 * n_boot)]), pc(bs[int(0.975 * n_boot) - 1])]
_g = g71c['summary']
d['fig3']['lit271'] = dict(n=_g['n_papers'], **{
    k: dict(recall=pc(_g['arms'][a]['recall']), path=_boot_complete(g71c['per_paper'][a]))
    for k, a in (('Direct', 'Direct'), ('Auto', 'MechSight_Auto'), ('DirectClues', 'Direct_clues'), ('Guided', 'MechSight_Guided'))},
    d_guided_clues=pc(_g['diffs']['MechSight_Guided_minus_Direct_clues']['recall']),
    d_guided_dir=pc(_g['diffs']['MechSight_Guided_minus_Direct']['recall']))

# ---------------- 2026-11-07 redesign (图表深度优化方案): new panels, every value from a run file
_spec = J(V4 / 'SUMMARY_P18_SPEC_stdinchi.json')
d['fig1']['spec2'] = {st: {a: dict(path=pc(b[a]['all_present']), reached=pc(b[a]['reached']) if 'reached' in b[a] else None)
                           for a in b if isinstance(b[a], dict)}
                      for st, b in (('in', _spec['uspto31k']['FlowER']), ('T3', _spec['T3_confirm']['FlowER']), ('oMe', _spec['gold']['FlowER']))}
d['fig1']['m2_strata'] = {k: dict(n=m2['n'][k], Direct=round(m2['means'][k]['Direct_strict'], 1)) for k in ('low', 'mid', 'high', 'recent')}
d['fig1']['m3_sub'] = dict(H=dict(n=m3['exploratory_by_requirement']['H']['n'], d=m3['exploratory_by_requirement']['H']['break_minus_preserve']),
                           group=dict(n=m3['exploratory_by_requirement']['group']['n'], d=m3['exploratory_by_requirement']['group']['break_minus_preserve']))
_gap = J(V4 / 'SUMMARY_DIRECT_GAP.json')['pooled']
_sc = J(V4 / 'supp_offline_20261106/SUMMARY_SUPP_OFFLINE.json')['1c_table1_SC9']['pooled']
d['fig2']['oracle_pooled_hid'] = pc(po['O_hid_minus_Direct'][0])
d['fig2']['ladder2'] = dict(
    L0=pc(lad['pooled']['L0_Direct']['all_present']), L1=pc(lad['pooled']['L1_sampling_revision']['all_present']),
    L2=pc(lad['pooled']['L2_plus_enumeration']['all_present']), L3=pc(lad['pooled']['L3_MechSight']['all_present']),
    d01=pc(lad['pooled']['L1_sampling_revision_minus_L0_Direct']), d12=pc(lad['pooled']['L2_plus_enumeration_minus_L1_sampling_revision']),
    d23=pc(lad['pooled']['L3_MechSight_minus_L2_plus_enumeration']),
    gap_only=pc(_gap['all']['Direct_gap_only']['all_present']), d_gap_only=pc(_gap['all_diffs']['Direct_gap_only-Direct']),
    vote7=pc(_sc['arms']['SC_pool7']['all_present']))
_lu = J(V4 / 'SUMMARY_LUNA_T3_high.json')['T3_confirm']
d['fig2']['luna'] = dict(n=_lu['n'], Direct=pc(_lu['Direct']['all_present']), SC=pc(_lu['SelfConsistency']['all_present']),
                         Auto=pc(_lu['MechSight_Auto']['all_present']), d_sc=pc(_lu['MechSight_Auto_minus_SelfConsistency']))
_c = off['C_gap_rule_on_reference_steps']
_dist = _c['heavy_sigma_changes_distribution']
d['fig2']['steps1483'] = dict(n=_c['reference_steps'], le2=sum(v for k, v in _dist.items() if int(k) <= 2), ge3=_c['reference_steps_flagged_ge3'][0])
_sup = J(V4 / 'supp_offline_20261106/SUMMARY_SUPP_OFFLINE.json')['1a_2b_literature']
_nd = J(V4 / 'nodft_confirm_20261106/SUMMARY_NODFT.json')['all_271']
d['fig3']['vote7'] = dict(dev=dict(recall=pc(_sup['dev']['arms']['PoolVote7_maj']['recall']), path=pc(_sup['dev']['arms']['PoolVote7_maj']['complete'])),
                          conf=dict(recall=pc(_sup['test']['arms']['PoolVote7_maj']['recall']), path=pc(_sup['test']['arms']['PoolVote7_maj']['complete'])))
_nc = _sup['test']['no_computational_clues']
d['fig3']['wetlab'] = dict(
    sub_n=_nc['n'], sub_clues=pc(_nc['arms']['Direct_clues']['recall']), sub_guided=pc(_nc['arms']['MechSight_Guided']['recall']),
    sub_d=pc(_nc['diffs']['MechSight_Guided-Direct_clues']['recall']),
    nodft_n=_nd['n'], nodft_clues=pc(_nd['arms']['Direct_clues_nodft']['recall']), nodft_guided=pc(_nd['arms']['MechSight_Guided_nodft']['recall']),
    nodft_d=pc(_nd['diffs']['MechSight_Guided_nodft-Direct_clues_nodft']['recall']))
_b74 = blind76['pooled']
d['fig3']['blind74'] = dict(n=blind76['n'], Direct=pc(_b74['Direct']['blind_species_written']), Auto=pc(_b74['Auto_V5']['blind_species_written']),
                            DirectClues=pc(_b74['Direct_clue']['blind_species_written']), Guided=pc(_b74['Guided_v7']['blind_species_written']),
                            d_clues=pc(_b74['Guided_v7_minus_Direct_clue']['blind_species_written']))

# ---------------- 2026-11-07 V2 taxonomy of the 251 omitted hidden intermediates
d['fig1']['taxonomy251'] = [
    dict(lab='zwitterion / ylide', n=12, pool=11, rec=8, blind=1),
    dict(lab='3-ring halonium / onium', n=12, pool=9, rec=7, blind=3),
    dict(lab='tetrahedral adduct', n=15, pool=10, rec=5, blind=5),
    dict(lab='cyclic / tautomer relay', n=70, pool=40, rec=18, blind=30),
    dict(lab='oxocarbenium / iminium', n=56, pool=28, rec=15, blind=28),
    dict(lab='organometallic / main-group', n=70, pool=26, rec=19, blind=42, gap=2),
    dict(lab='rearranging carbocation', n=8, pool=3, rec=3, blind=5),
    dict(lab='open-shell radical / SET', n=8, pool=2, rec=2, blind=6),
]
assert sum(r['n'] for r in d['fig1']['taxonomy251']) == 251
assert sum(r['pool'] for r in d['fig1']['taxonomy251']) == 129
assert sum(r['rec'] for r in d['fig1']['taxonomy251']) == 77
assert sum(r['blind'] for r in d['fig1']['taxonomy251']) == 120

(HERE / 'figdata.json').write_text(json.dumps(d, indent=1))
print(json.dumps({k: list(v) for k, v in d.items()}))
