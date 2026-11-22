# MechSight: Knowledge-Driven Reaction Mechanism Elucidation

Code, prompts, benchmark splits, frozen analysis tables, figure-build scripts, and lossless raw model outputs accompanying the manuscript:

> **Knowledge-Driven Reaction Mechanism Elucidation: Unlocking Inert Chemical Knowledge and Bridging Blind Spots in Large Language Models**
> Peize Li, Yimi Wang, Chengchun Liu, and Fanyang Mo\* (*Angewandte Chemie International Edition*, 2026)

This repository contains every script, frozen summary, canonical analysis table, figure source, and raw model output archive needed to verify all numbers, tables, and figures in the manuscript (Tables 1–2, Figures 1–4) and Supporting Information (Tables S1–S21, Figures S1–S5) offline without LLM API calls. No API keys or third-party copyrighted article PDFs are included; all 436 open-access 2026 literature studies (including 285 post-cutoff studies from April–November 2026) are individually identified by PMC ID and DOI in Supporting Information Tables S20–S21.

---

## Repository Structure

| Path | Description |
| :--- | :--- |
| `code/MechAgent/experiments/pilots_v4/` | Primary experiment, scoring, and cross-backbone evaluation scripts (`common4.py` contains shared prompts, model callers, and RDKit/InChIKey scorers; `spec_server/` holds FlowER & ChRIMP GPU runners). |
| `code/MechAgent/experiments/{mechsight,smiles,inert_knowledge,agent_v2}/` | Core MechSight libraries: RXNMapper + RDKit elementary bond-change checker ($\Delta_{\mathrm{bond}} \le 2$), cross-draft splicing, consensus experimental anchoring, and hidden-intermediate probes. |
| `code/MechAgent/experiments/{lit150,lit_more,lit_scale,lit_old,evidence_p0}/` | 2026 open-access literature curation pipeline (Europe PMC retrieval, multimodal scheme/observation extraction, conclusion stripping, RDKit validation; third-party article PDFs omitted). |
| `code/MechAgent/develop/`, `testcases/` | FukuyamaBench PGL conversion, deterministic reaction-state diffs, and unit tests. |
| `code/external/oMeBench/` | Unmodified oMeBench data, prompts, and scoring scripts (MIT License). |
| `code/environment/` | Clean pinned Python environment specifications (`requirements_llm_py3.14.txt` for LLM/RDKit 2026.3.5; `requirements_rxnmapper.txt` + `python_version_rxnmapper.txt` for Python 3.10 + RXNMapper 0.4.3). |
| `figure_build/` | Figure-build scripts (`v4_20261108/build/fig{1..4}.py`, `build_si_figures.py`), ChemDraw (`.cdxml`) structures, and final publication SVGs (`figure_build/final_svgs/`) for Figures 1–4 and S1–S5. |
| `results/paper_summaries/` | Uncompressed frozen experiment summaries (`SUMMARY*.json`, `REGISTRATION*.json`, deviation notes, reports) for immediate inspection. |
| `results/canonical_analyses/` | Uncompressed canonical tables & scripts backing Tables 1–2, Figures 1–4, and SI Tables S1–S21 (`v4_analyses_20261108/`, `v3_analyses_20261107/`, `stepwise_20261108/`, `nodft_case_20261108/`, `pilots_v4_summaries/`). |
| `results/raw_outputs_archives/` | 7 lossless `.tar.gz` archives containing all ~140,700 raw model outputs and trace logs (every archive < 25 MB for GitHub compatibility) plus `SHA256SUMS.txt`. |
| `unpack_raw_outputs.sh` | One-command SHA-256 verification and extraction script restoring the exact `results/raw_outputs/` directory tree. |
| `LICENSE`, `CITATION.cff` | MIT License (with third-party benchmark attributions) and citation metadata. |

### Raw-Output Archives (`results/raw_outputs_archives/`)

| Archive | Restored Directories | Contents |
| :--- | :--- | :--- |
| `01_benchmarks_confirm_and_ablation.tar.gz` | `T3_confirm/`, `T3_dev/`, `J3/`, `gold/`, `uspto31k/`, `spec_v5/`, `spec_v6_4090/`, `cards/` | Primary *gemini-3.8-flash* benchmark runs (FukuyamaBench, oMe-Gold, mech-USPTO-31k), ablation ladder, and FlowER/ChRIMP native RTX 4090 outputs. |
| `02_causal_probes_and_crossmodel.tar.gz` | `argon/`, `bb_claude-sonnet-5-5-high/`, `bb_gpt-6_1-sol_high/`, `luna/`, `luna_high/`, `m1/`–`m4/`, `litoldg/`, `lit_old_argon/` | Causal probes (M1–M4) and cross-model transfer outputs for *claude-sonnet-5.5*, *gpt-6-luna* (high & no-reasoning), *gemini-4-argon*, and supplementary *gpt-6.1-sol*. |
| `02_crossmodel_deepseek_v4_1_flash.tar.gz` | `bb_deepseek-v4_1-flash_high/`, `bb_deepseek-v4_1-flash_nothink/` | Cross-model transfer outputs for *deepseek-v4.1-flash* (high reasoning & no-reasoning, Table S14). |
| `03_lit2026_dev_165.tar.gz` | `lit172g/`, `lit172g_nodft/`, `lit172g_p36/`, `lit150/`, `lit172_v7_real_20261105/`, `lit_gold_argon/` | 2026 literature development cohort ($n = 165$ studies) and wet-lab-only re-evaluations. |
| `04_lit2026_test_271_and_nodft.tar.gz` | `litnew_g/`, `litnew_g_nodft/`, `litnew_confirm_g71/`, `lit_scale/`, `lit_scale_gold/`, `supp_clue_audit_20261106/` | Pre-registered 2026 literature test cohort ($n = 271$ studies), pure wet-lab-only re-evaluations, and clue audits. |
| `05_pilots_v4_other_runs.tar.gz` | `g71_dev_20261106/`, `p34_pilot36_20261106/`, `p34b_a1_38_20261106/`, `silver_dev/`, `silver_conf/`, `stepwise_llm/`, `nodft_confirm_20261106/`, `supp_offline_20261106/`, `template/` | Auxiliary pilot, single-step LLM, oMe-Silver, and offline verification caches. |
| `06_early_exploration_storyline.tar.gz` | `03_early_exploration_storyline_package/` | Early diagnostic probes (template degeneration, inert-knowledge, hidden-intermediate visibility, and falsified interventions). |

---

## Quick Start: Offline Verification (Zero API Calls)

### 1. Verify Checksums and Unpack Raw Outputs

```bash
./unpack_raw_outputs.sh
```

This verifies all 7 archives against `results/raw_outputs_archives/SHA256SUMS.txt` via `shasum -a 256 -c` and restores the complete `results/raw_outputs/pilots_v4_20261103/` and `results/raw_outputs/03_early_exploration_storyline_package/` directory trees (including all canonical analyses and summary JSONs).

### 2. Link Restored Outputs for Analysis Scripts

To re-run offline scoring and aggregation scripts from `code/MechAgent/experiments/pilots_v4/`:

```bash
mkdir -p code/MechAgent/outputs
ln -s "$(pwd)/results/raw_outputs/pilots_v4_20261103" code/MechAgent/outputs/pilots_v4_20261103
```

---

## Offline Verification & Reproduction Commands

Every number reported in the manuscript and Supporting Information can be inspected directly from the uncompressed JSON files under `results/canonical_analyses/` and `results/paper_summaries/`, or recomputed offline from the unpacked raw outputs:

### Example Offline Recomputation Commands

```bash
# 1. Verify archive integrity only
(cd results/raw_outputs_archives && shasum -a 256 -c SHA256SUMS.txt)

# 2. Recompute Main-Text Table 1 & SI Table S3 (strict & chemically equivalent matching, N = 244)
(cd code/MechAgent/experiments/pilots_v4 && \
  python ../../../results/canonical_analyses/v4_analyses_20261108/table1_recall_spec4090.py)

# 3. Recompute Main-Text Table 2 & SI Tables S17–S18 (285 post-cutoff 2026 studies)
python results/canonical_analyses/v4_analyses_20261108/merged_report.py \
  results/canonical_analyses/v4_analyses_20261108/bench_strict
python results/canonical_analyses/v4_analyses_20261108/merged_report.py \
  results/canonical_analyses/v4_analyses_20261108/bench_eq

# 4. Recompute cross-backbone transfer & offline structural/over-generation audits (SI Tables S13–S16)
(cd code/MechAgent/experiments/pilots_v4 && \
  python p14_confirm.py --analyze-only && \
  python p23_offline.py && \
  python p42_auto_inserted_audit.py && \
  python p46_backbone.py --model deepseek-v4.1-flash --effort high --phase analyze && \
  python p46_backbone.py --model deepseek-v4.1-flash --phase analyze && \
  python p46_backbone.py --model claude-sonnet-5-5-high --phase analyze)

# 5. Re-render Main-Text Figures 1–4 and SI Figures S1–S5 from frozen data
(cd figure_build/v4_20261108/build && python fig1.py && python fig2.py && python fig3.py && python fig4.py)
(cd figure_build && python build_si_figures.py)
```

### Complete Table & Figure Provenance Map

| Manuscript / SI Item | Frozen Summary / Canonical Data | Analysis / Build Script |
| :--- | :--- | :--- |
| **Table 1** (Hierarchical evaluation, $N = 244$) | `results/canonical_analyses/v4_analyses_20261108/table1_recall_spec4090.json`, `results/canonical_analyses/pilots_v4_summaries/SUMMARY_P14_confirm.json` | `results/canonical_analyses/v4_analyses_20261108/table1_recall_spec4090.py`, `code/MechAgent/experiments/pilots_v4/p14_confirm.py` |
| **Table 2** (285 post-cutoff 2026 studies) | `results/canonical_analyses/v4_analyses_20261108/bench_{strict,eq}/merged_apr_oct.json` | `results/canonical_analyses/v4_analyses_20261108/lit_bench_rule.py`, `merged_report.py` |
| **Figures 1–4** (Main-text figures) | `figure_build/final_svgs/fig{1_knowledge,2_hidden,3_mechsight,4_guided}_v3.svg` | `figure_build/v4_20261108/build/fig{1,2,3,4}.py` |
| **Figures S1–S5** (SI figures) | `figure_build/final_svgs/figS{1_origin_probes,2_length_and_audit,3_falsified_and_tracking,4_pericyclic_and_ablation,5_blindspot_and_lit_pipeline}.svg` | `figure_build/build_si_figures.py` |
| **Table S1** (Benchmarks & template coverage) | `results/canonical_analyses/pilots_v4_summaries/SUMMARY_P1.json` | `code/MechAgent/experiments/pilots_v4/p1_retrieval.py` |
| **Table S2** (FlowER/ChRIMP vs LLMs: in-template & out-of-template) | `results/canonical_analyses/pilots_v4_summaries/SUMMARY_P18_SPEC_spec_v6_4090.json`, `SUMMARY_USPTO_V5.json`, `figure_build/v4_20261108/data/intemplate_equiv_spec4090.json` | `code/MechAgent/experiments/pilots_v4/p18_spec_compare.py`, `figure_build/v4_20261108/data/intemplate_equiv_spec4090.py` |
| **Table S3** (Strict vs chemically equivalent complete-pathway) | `results/canonical_analyses/v3_analyses_20261107/equiv_table1_spec4090.json`, `results/canonical_analyses/v4_analyses_20261108/table1_recall_spec4090.json` | `code/MechAgent/experiments/pilots_v4/p40_equiv_rescore.py`, `table1_recall_spec4090.py` |
| **Table S4** (Single-step accuracy given $S_{k-1}$) | `results/canonical_analyses/stepwise_20261108/SUMMARY_STEPWISE.json` | `code/MechAgent/experiments/pilots_v4/p44_stepwise.py` |
| **Table S5** (M3 reactive-site blocking, $n = 84$ triplets) | `results/paper_summaries/pilots_v4_20261103/m3_counterfactual_20261106/SUMMARY_M3.json` | `code/MechAgent/experiments/pilots_v4/p32_m3_counterfactual.py` |
| **Tables S6–S7** (M2 citation exposure & covariate adjustment) | `results/canonical_analyses/v4_analyses_20261108/exposure_unified.json`, `results/paper_summaries/pilots_v4_20261103/m2_exposure_20261106/SUMMARY_M2.json` | `results/canonical_analyses/v4_analyses_20261108/exposure_rule.py`, `code/MechAgent/experiments/pilots_v4/p31_m2_exposure.py` |
| **Table S8** (M1 bibliographic recall & M4 counterfactual override) | `results/paper_summaries/pilots_v4_20261103/m1_m4_20261106/SUMMARY_M1_M4.json` | `code/MechAgent/experiments/pilots_v4/p33_m1_m4.py` |
| **Table S9** (Homologue & randomized SMILES probes) | `results/canonical_analyses/pilots_v4_summaries/SUMMARY_P5.json`, `SUMMARY_P22_RANDSMI.json` | `code/MechAgent/experiments/pilots_v4/p5_contamination.py`, `p22_extras.py` |
| **Table S10** (Visible vs hidden bond events & granularity audit) | `results/paper_summaries/inert_knowledge_20261102/SUMMARY_visibility.json`, `results/canonical_analyses/pilots_v4_summaries/SUMMARY_P23_OFFLINE.json` | `code/MechAgent/experiments/pilots_v4/p23_offline.py` |
| **Table S11** (Inert-knowledge probes, 251 hidden taxonomy, oracle) | `results/paper_summaries/inert_knowledge_20261102/SUMMARY.json`, `SUMMARY_phase2.json`, `results/canonical_analyses/pilots_v4_summaries/SUMMARY_P4.json`, `SUMMARY_GOLD_ORACLE.json` | `code/MechAgent/experiments/pilots_v4/p4_inertness.py` |
| **Table S12** (Falsified alternative interventions) | `results/paper_summaries/activation_pilot_20261103/SUMMARY.json`, `results/paper_summaries/scale_probe_20261031/SUMMARY.json`, `results/canonical_analyses/pilots_v4_summaries/SUMMARY_P12.json` | `code/MechAgent/experiments/pilots_v4/p12_controls.py` |
| **Table S13** (MechSight-Auto benchmarks, hidden strata, ablation ladder) | `results/canonical_analyses/pilots_v4_summaries/SUMMARY_P14_confirm.json`, `SUMMARY_P19_LADDER.json` | `code/MechAgent/experiments/pilots_v4/p14_confirm.py`, `p19_ladder.py` |
| **Table S14** (Zero-shot transfer across 4 additional backbones / 4 vendors: *claude-sonnet-5.5*, *gpt-6-luna*, *deepseek-v4.1-flash*, *gemini-4-argon*) | `results/canonical_analyses/pilots_v4_summaries/SUMMARY_BB_claude-sonnet-5-5-high.json`, `SUMMARY_LUNA_T3_high.json`, `SUMMARY_LUNA_T3.json`, `SUMMARY_BB_deepseek-v4_1-flash_high.json`, `SUMMARY_BB_deepseek-v4_1-flash_nothink.json`, `SUMMARY_ARGON_CONFIRM.json`, `results/canonical_analyses/v4_analyses_20261108/backbone_rule.json` | `code/MechAgent/experiments/pilots_v4/p46_backbone.py`, `p38_luna_t3.py`, `p17_argon_confirm.py`, `results/canonical_analyses/v4_analyses_20261108/backbone_rule.py` |
| **Table S15** (Over-generation controls, 214 inserted-species audit, repairs) | `results/canonical_analyses/v3_analyses_20261107/auto_inserted_audit_and_examples.json`, `results/canonical_analyses/pilots_v4_summaries/SUMMARY_P23_OFFLINE.json` | `code/MechAgent/experiments/pilots_v4/p42_auto_inserted_audit.py`, `p23_offline.py` |
| **Table S16** ($\Delta_{\mathrm{bond}} \ge 3$ shortcut-detection rule & pericyclic audit) | `results/canonical_analyses/pilots_v4_summaries/SUMMARY_P23_OFFLINE.json`, `SUMMARY_DIRECT_GAP.json` | `code/MechAgent/experiments/pilots_v4/p23_offline.py`, `p39_direct_gap.py` |
| **Table S17** (Out-of-time 2026 literature by subset & wet-lab-only) | `results/canonical_analyses/v4_analyses_20261108/bench_{strict,eq}/merged_apr_oct.json`, `results/canonical_analyses/v4_analyses_20261108/wl/rows_test.json`, `results/canonical_analyses/nodft_case_20261108/SUMMARY.json` | `results/canonical_analyses/v4_analyses_20261108/lit_bench_rule_wl.py`, `merged_report.py` |
| **Table S18** (Knowledge strata across 1,571 post-cutoff intermediates) | `results/canonical_analyses/v4_analyses_20261108/bench_strict/merged_apr_oct.json`, `results/canonical_analyses/v3_analyses_20261107/knowledge_strata_{dev,test}.json` | `code/MechAgent/experiments/pilots_v4/p41_knowledge_strata.py`, `merged_report.py` |
| **Table S19** (Full 436-paper 2026 corpus & wet-lab-only controls) | `results/paper_summaries/pilots_v4_20261103/lit172_v7_real_20261105/SUMMARY_LIT172_V7_REAL.json`, `litnew_confirm_g71/SUMMARY_G71.json`, `results/canonical_analyses/nodft_case_20261108/SUMMARY.json` | `code/MechAgent/experiments/pilots_v4/p28_lit172_v7.py`, `p31_guided71.py` |
| **Tables S20–S21** (Per-paper 2026 dev $n = 165$ & test $n = 271$ results) | `results/canonical_analyses/v4_analyses_20261108/bench_{strict,eq}/rows_{dev,test}.json`, `pub_dates.json` | `results/canonical_analyses/v4_analyses_20261108/lit_bench_rule.py` |

---

## Mapping Between Internal Code Names and Manuscript Terms

| Code / Folder Name | Manuscript Term |
| :--- | :--- |
| `T3_confirm` | FukuyamaBench benchmark set ($n = 48$) |
| `T3_dev`, `J3` | Additional FukuyamaBench diagnostic reactions (242-reaction expanded corpus) |
| `gold` | oMe-Gold ($n = 196$) |
| `uspto31k` | mech-USPTO-31k in-template benchmark ($n = 200$) |
| `base_default` / `base_cot` / `SR2` / `SelfConsistency` | Direct / Chain-of-thought / Self-refinement (2 rounds) / Self-consistency |
| `MechAgent` (`V5_rev1`) | **MechSight-Auto** |
| `A1` (`G71_rev` + `A1_place`) | **MechSight-Guided** |
| `O_llm` / `MV7` | Direct + Clues / Majority vote + Clues |
| `bb_claude-sonnet-5-5-high` / `luna_high`, `luna` / `bb_deepseek-v4_1-flash_{high,nothink}` / `argon` | Cross-backbone transfer models in Table S14 (*claude-sonnet-5.5*, *gpt-6-luna*, *deepseek-v4.1-flash*, *gemini-4-argon*) |
| `lit172g` (165) + `litnew_g` (271) | 2026 literature corpus (436 studies; **285 post-cutoff studies**, April–November 2026) |
| `lit172g/A1_place/PMC13449957.json` | Main-text Figure 4c purely wet-lab case study (Wickens, *J. Am. Chem. Soc.* 2026, *148*, 31641) |
| `litnew_g_nodft/`, `lit172g_nodft/` | Pure wet-lab-only re-evaluations with all DFT statements removed |
| `spec_v6_4090/` | FlowER & ChRIMP native evaluations on NVIDIA RTX 4090 GPUs |

---

## Re-Running LLM Inference

1. Create the two Python environments from `code/environment/`:
   - Primary LLM / RDKit / figure environment (Python 3.14): `pip install -r code/environment/requirements_llm_py3.14.txt` (core packages: `rdkit==2026.3.5`, `openai`, `numpy`, `networkx`, `matplotlib`).
   - RXNMapper atom-mapping environment (Python 3.10): `pip install -r code/environment/requirements_rxnmapper.txt` (`rxnmapper==0.4.3`, `rdkit==2024.3.3`, `torch`).
2. Export `OPENAI_API_KEY` (and optionally `FALLBACK_API_KEY`) and replace placeholder endpoints `LLM-GATEWAY-1.example` / `LLM-GATEWAY-2.example` in `code/MechAgent/experiments/mechsight/failover.py` (and `p46_backbone.py` / `p38_luna_t3.py`) with your OpenAI-compatible API endpoint.
3. Specialist baselines (FlowER, ChRIMP) were executed externally using their official open-source weights and code via the wrapper scripts in `code/MechAgent/experiments/pilots_v4/spec_server/`.

---

## Privacy, Sanitization & Licensing

- All private gateway hostnames, internal proxy error IPs, GPU server addresses, and local developer paths were replaced with generic placeholders prior to archiving; no scientific data, prompts, model predictions, or scores were altered.
- Original code, prompts, and analysis artifacts are released under the **MIT License** (`LICENSE`); third-party benchmarks (oMeBench, FukuyamaBench, mech-USPTO-31k) retain their respective licenses and attributions.
- Citation metadata is provided in `CITATION.cff`.
