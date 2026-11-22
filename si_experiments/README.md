# Supporting Information Experiments & Verification Index (`si_experiments/`)

This directory organizes every experiment, probe, ablation, control, and prompt template in the repository strictly according to the **Supporting Information (Sections S1–S12)** of:

> **Knowledge-Driven Reaction Mechanism Elucidation: Unlocking Inert Chemical Knowledge and Bridging Blind Spots in Large Language Models**

Each subdirectory (`S01_*` – `S12_*`) documents the exact underlying scripts in `reproduction/`, the frozen summary files in `results/`, and provides a runnable `reproduce.py`:
- **Default mode** (`python si_experiments/S0X_.../reproduce.py`): verifies all reported manuscript and SI numbers directly from the uncompressed frozen JSON summaries in `results/`.
- **Offline recomputation mode** (`--recompute`, after running `./unpack_raw_outputs.sh`): re-executes the underlying scoring and statistical scripts over the ~140,700 archived raw model outputs with **zero LLM API calls**.

## Master SI-to-Code Index

| SI Section | Directory | Paper / SI Items Covered | Key Underlying Scripts (`reproduction/` & `results/`) |
| :--- | :--- | :--- | :--- |
| **S1** | [`S01_environment_and_stats/`](S01_environment_and_stats/) | Software versions, Python 3.14 & RXNMapper 0.4.3 environments, bootstrap/permutation methodology | `requirements_llm_py3.14.txt`, `requirements_rxnmapper.txt`, `python_version_rxnmapper.txt` |
| **S2** | [`S02_benchmarks_and_splits/`](S02_benchmarks_and_splits/) | Table S1; FukuyamaBench (48 test / 194 dev / 242 total), oMe-Gold (196), mech-USPTO-31k (200), template retrieval | `data/fukuyama_*.json*`, `pilots_v4/p1_retrieval.py`, `p1_template_baseline/fukuyama_coverage.py` |
| **S3** | [`S03_evaluation_and_specialists/`](S03_evaluation_and_specialists/) | Main Table 1, Fig. 1b, Tables S2–S4 (FlowER/ChRIMP vs LLMs, strict vs equiv matching, single-step $S_{k-1}$) | `v4_analyses_20261108/table1_recall_spec4090.py`, `pilots_v4/{p18_spec_compare,p40_equiv_rescore,p44_stepwise}.py` |
| **S4** | [`S04_causal_probes/`](S04_causal_probes/) | Fig. 1c–d, Fig. S1, Tables S5–S9 (M3 reactive-site blocking, M2 citation exposure, M1/M4, homologues & random SMILES) | `pilots_v4/{p32_m3_counterfactual,p31_m2_exposure,p33_m1_m4,p5_contamination,p22_extras}.py`, `exposure_rule.py` |
| **S5** | [`S05_hidden_intermediate_bottleneck/`](S05_hidden_intermediate_bottleneck/) | Fig. 2, Fig. S2, Fig. S3b–c, Tables S10–S11 (visible vs hidden events, 115-step granularity, 251 hidden taxonomy, oracles) | `inert_knowledge/{probe,visibility,phase2}.py`, `recall_probe/audit_gap.py`, `pilots_v4/{p4_inertness,p12_controls,p23_offline}.py` |
| **S6** | [`S06_falsified_alternatives/`](S06_falsified_alternatives/) | Fig. S3a, Table S12 (Mayr scale probe, 1-arrow search, shortlist, pairwise/listwise judges) | `scale_probe/probe.py`, `mechsight/activation.py`, `pilots_v4/{p3_selector,p11_mechagent,p12_controls}.py` |
| **S7** | [`S07_mechsight_auto/`](S07_mechsight_auto/) | Fig. 3, Tables S13–S14 (MechSight-Auto benchmarks, ablation ladder, equal-compute & 4-vendor cross-model transfer) | Standalone [`mechsight/auto.py`](../mechsight/auto.py); `pilots_v4/{p14_confirm,p19_ladder,p46_backbone,p38_luna_t3,p17_argon_confirm}.py`, `backbone_rule.py` |
| **S8** | [`S08_rigor_controls_and_pericyclic/`](S08_rigor_controls_and_pericyclic/) | Fig. S4, Tables S15–S16 (over-generation controls, 214 inserted-species audit, $\Delta_{\mathrm{bond}}\ge 3$ & pericyclic check) | `pilots_v4/{p23_offline,p42_auto_inserted_audit,p39_direct_gap}.py`, `smiles/map_check.py` |
| **S9** | [`S09_literature_corpus_curation/`](S09_literature_corpus_curation/) | Fig. S5a (436-paper 2026 OA curation, 285 post-cutoff split, 4,388-line modality & 50-paper PDF audit) | `lit150/`, `lit_more/`, `lit_old/`, `evidence_p0/`, `pilots_v4/{p29_gold_rebuild,p36_clue_audit}.py` |
| **S10** | [`S10_mechsight_guided/`](S10_mechsight_guided/) | Main Table 2, Fig. 4a–b, Fig. S5b–c, Tables S17–S19 (285 post-cutoff evaluation, knowledge strata, wet-lab-only controls) | Standalone [`mechsight/guided.py`](../mechsight/guided.py); `v4_analyses_20261108/{lit_bench_rule,lit_bench_rule_wl,merged_report}.py`, `pilots_v4/{p31_guided71,p31_confirm_core,p37_nodft,p41_knowledge_strata}.py` |
| **S11** | [`S11_case_studies_and_per_paper/`](S11_case_studies_and_per_paper/) | Fig. 4c, Tables S20–S21 (Wickens 2026 `PMC13449957`, Chen 2026 `PMC13321810`, Hu 2026 `PMC13171995`, 165 dev & 271 test tables) | `v4_analyses_20261108/bench_{strict,eq}/rows_{dev,test}.json`, `pub_dates.json`, `raw_outputs/pilots_v4_20261103/{lit172g,litnew_g}/` |
| **S12** | [`S12_prompt_templates/`](S12_prompt_templates/) | Verbatim prompt templates S12.1–S12.8 | `S12_*.txt`, `verify_prompts.py`, [`mechsight/prompts.py`](../mechsight/prompts.py) |

## Quick Verification

```bash
# 1. Verify all frozen summaries & prompt parity (works immediately on a fresh clone)
python si_experiments/verify_all.py

# 2. Unpack archived raw outputs & run full offline recomputation (zero API calls)
./unpack_raw_outputs.sh
python si_experiments/verify_all.py --full
```
