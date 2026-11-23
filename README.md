# MechSight: Knowledge-Driven Reaction Mechanism Elucidation

Official standalone package, prompt library, benchmark splits, frozen analysis tables, figure-build scripts, and lossless raw model output archives accompanying the manuscript:

> **Knowledge-Driven Reaction Mechanism Elucidation: Unlocking Inert Chemical Knowledge and Bridging Blind Spots in Large Language Models**
> Peize Li, Yimi Wang, Chengchun Liu, and Fanyang Mo\* (2026)

This repository provides two cleanly separated entry points:
1. **Standalone Main Method (`mechsight/`)** — A pip-installable, dependency-minimal Python package implementing **MechSight-Auto** and **MechSight-Guided** (exact prompts, RDKit chemical-equivalence checks, RXNMapper elementary bond-change gap verification with $\Delta_{\mathrm{bond}} \ge 3$, cross-draft splicing, consensus experimental anchoring, and SI S3.1 / S1.3 statistical evaluation) with zero dependencies on internal experiment runners.
2. **SI-Aligned Reproduction (`si_experiments/S01..S12/`)** — Every experiment, causal probe, ablation, benchmark split, prompt verification, and figure script organized 1-to-1 by **Supporting Information Sections S1–S12**, supporting both instant summary verification and full offline recomputation over ~140,700 archived raw model outputs with **zero LLM API calls**.

---

## Quick Start

### Track 1: Use the Standalone `mechsight` Package (Main Method)

```bash
# Install editable package into your Python >= 3.10 environment (RDKit + OpenAI client)
pip install -e .

# Optional: configure OpenAI-compatible LLM credentials and RXNMapper environment
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"        # or any OpenAI-compatible endpoint
export MECHSIGHT_MODEL="gemini-3.8-flash"
export MECHSIGHT_RXNMAPPER_PYTHON="/path/to/rxnmapper-env/bin/python"
```

#### Offline Sanity Checks (No API Key Required)

```bash
# Run unit tests (chemistry, bond-change checker, parsers/prompts, mock pipeline, evaluation)
python -m unittest discover -s tests -v

# Inspect prompts and offline Delta_bond >= 3 gap detection on included case studies
python examples/run_mechsight_auto_example.py --dry-run
python examples/run_mechsight_guided_example.py --dry-run
```

#### Python API (`MechSightAuto` & `MechSightGuided`)

```python
from mechsight import (
    BondChangeChecker,
    MechSightAuto,
    MechSightGuided,
    OpenAICompatibleClient,
    ReactionCase,
    evaluate_pathway,
)

case = ReactionCase(
    case_id="demo_reaction",
    reactants=["CC(=O)c1ccccc1", "Br"],
    products=["O=C(CBr)c1ccccc1"],
    conditions="HBr / AcOH",
    observations="Radical-clock cyclopropyl substrate opens to ring-opened bromide.",
)

llm = OpenAICompatibleClient()          # reads OPENAI_API_KEY, OPENAI_BASE_URL, MECHSIGHT_MODEL
checker = BondChangeChecker()          # uses in-process rxnmapper or MECHSIGHT_RXNMAPPER_PYTHON

# Stage 1: Clue-free autonomous prior scaffold (MechSight-Auto)
auto_result = MechSightAuto(llm=llm, bond_checker=checker).run(case)
print(auto_result.final_pathway)

# Stage 2: Evidence-anchored pathway via >= 3/7 consensus (MechSight-Guided)
guided_result = MechSightGuided(llm=llm, bond_checker=checker).run(case)
print("Consensus anchors:", guided_result.consensus_anchors)
print(guided_result.final_pathway)
```

#### Command-Line Interface (`mechsight` / `python -m mechsight`)

```bash
mechsight auto --input examples/sample_reactions.json --output out_auto.json
mechsight guided --input reaction.json --output out_guided.json
mechsight check-gaps --input reaction.json --mechanism out_auto.json
mechsight evaluate --input reaction_with_ref.json --mechanism out_guided.json
```

---

### Track 2: Reproduce All Manuscript & SI Tables and Figures Offline (Zero API Calls)

#### 1. Verify Archived Outputs & Unpack Checksums

```bash
./unpack_raw_outputs.sh
```

This verifies SHA-256 checksums across all 7 lossless archives in [`results/raw_outputs_archives/`](results/raw_outputs_archives/), unpacks the raw model outputs into `results/raw_outputs/`, copies the canonical analysis tables into place, and wires the symlinks under `reproduction/MechAgent/outputs/` (including the FukuyamaBench / oMe-Gold benchmark split files stored in [`si_experiments/S02_benchmarks_and_splits/data/`](si_experiments/S02_benchmarks_and_splits/data/)).

#### 2. Verify All 12 Supporting Information Sections

```bash
# Instant verification against frozen summaries + SI S12 prompt-template parity check
python si_experiments/verify_all.py

# Full offline recomputation over unpacked ~140,700 raw model outputs (zero API calls)
python si_experiments/verify_all.py --full

# Or run any individual SI section script:
python si_experiments/S07_mechsight_auto/reproduce.py --recompute
python si_experiments/S10_mechsight_guided/reproduce.py --recompute
```

#### 3. Re-Render Publication Figures (Figures 1–4, TOC, Figures S1–S5)

```bash
# Main-text Figures 1-4 and TOC graphic (writes to figures/generated/main/)
for s in fig1.py fig2.py fig3.py fig4.py toc.py; do
  FIG_OUT=figures/generated/main python "figures/main_figures/build/$s"
done

# Supporting Information Figures S1-S5 (writes to figures/generated/si/)
python figures/build_si_figures.py
```

---

## Repository Architecture

```text
MechSight/
├── pyproject.toml                   # Standalone mechsight==1.0.0 package metadata & CLI entry
├── LICENSE                          # MIT License + third-party benchmark attributions
├── CITATION.cff                     # Citation metadata
├── unpack_raw_outputs.sh            # SHA-256 check, archive extraction, and symlink setup
│
├── mechsight/                       # Standalone main-method package (zero pilots_v4 deps)
│   ├── schema.py                    # ReactionCase, MechanismStep, GapFlag, Auto/GuidedResult
│   ├── chemistry.py                 # RDKit canonicalization, SI S3.1 sigma/equiv keys, parsers
│   ├── bond_checker.py              # SI Eq. S8 heavy-bond change Delta_bond & gap flagger
│   ├── prompts.py                   # Verbatim SI S12.1-S12.8 prompt templates
│   ├── llm.py                       # LLMClient protocol + OpenAI-compatible client w/ cache
│   ├── auto.py                      # MechSightAuto (drafting, Delta_bond>=3 completion, splicing)
│   ├── guided.py                    # MechSightGuided (>=3/7 consensus anchors + non-destructive insert)
│   ├── evaluation.py                # SI S3.1 metrics + SI S1.3 bootstrap CI & permutation tests
│   ├── cli.py                       # CLI: auto, guided, check-gaps, evaluate
│   └── assets/                      # Bundled oMeBench baseline prompt templates
│
├── examples/                        # Runnable examples + sample_reactions.json (--dry-run works offline)
├── tests/                           # 15 unit tests (chemistry, bond checker, parsers, mock pipelines, eval)
│
├── si_experiments/                  # 1-to-1 reproduction index by Supporting Information Sections S1-S12
│   ├── verify_all.py                # Master verifier (--full triggers offline recomputation)
│   ├── S01_environment_and_stats/   # Pinned Python 3.14 & RXNMapper 0.4.3 env specs + stats audit
│   ├── S02_benchmarks_and_splits/   # Table S1 + FukuyamaBench / oMe-Gold / mech-USPTO-31k splits
│   ├── S03_evaluation_and_specialists/ # Table 1, Fig. 1b, Tables S2-S4
│   ├── S04_causal_probes/           # Fig. 1c-d, Fig. S1, Tables S5-S9 (M1-M4, homologues, RandSMILES)
│   ├── S05_hidden_intermediate_bottleneck/ # Fig. 2, Fig. S2, Tables S10-S11 (inert vs blind spot)
│   ├── S06_falsified_alternatives/  # Fig. S3a, Table S12 (Mayr probe, 1-arrow, shortlist, judges)
│   ├── S07_mechsight_auto/          # Fig. 3, Tables S13-S14 (MechSight-Auto, ladder, 4-vendor transfer)
│   ├── S08_rigor_controls_and_pericyclic/ # Fig. S4, Tables S15-S16 (over-generation, 214-species, pericyclic)
│   ├── S09_literature_corpus_curation/ # Fig. S5a (436-paper curation, 285 post-cutoff, PDF audit)
│   ├── S10_mechsight_guided/        # Table 2, Fig. 4a-b, Fig. S5b-c, Tables S17-S19 (285 studies, wet-lab)
│   ├── S11_case_studies_and_per_paper/ # Fig. 4c, Tables S20-S21 (Wickens, Chen, Hu + 165 dev / 271 test)
│   └── S12_prompt_templates/        # Verbatim SI S12.1-S12.8 .txt files + automated parity checker
│
├── figures/                         # Main Figures 1-4, TOC, and SI Figures S1-S5 + ChemDraw assets
│   ├── final_svgs/                  # Final publication SVGs matching the manuscript
│   ├── main_figures/                # Scripts (build/fig{1..4}.py, toc.py), mols/, data/, out/
│   └── build_si_figures.py          # Build script for Figures S1-S5
│
├── results/
│   ├── canonical_analyses/          # Uncompressed canonical summary JSONs & aggregation scripts
│   ├── paper_summaries/             # Uncompressed frozen SUMMARY_*.json outputs per experiment
│   └── raw_outputs_archives/        # 7 lossless .tar.gz archives (~77 MB total) + SHA256SUMS.txt
│
└── reproduction/                    # Frozen original research codebase & unmodified external/oMeBench
```

---

## Supporting Information Section-by-Section Provenance (S1–S12)

| SI Section | Entry Directory | Manuscript / SI Items | Underlying Code (`reproduction/`, `mechsight/`) & Frozen Results |
| :--- | :--- | :--- | :--- |
| **S1** | [`S01_environment_and_stats/`](si_experiments/S01_environment_and_stats/) | Software versions, statistics, random-seed & deduplication rules | `requirements_llm_py3.14.txt`, `requirements_rxnmapper.txt`, [`mechsight/evaluation.py`](mechsight/evaluation.py) |
| **S2** | [`S02_benchmarks_and_splits/`](si_experiments/S02_benchmarks_and_splits/) | **Table S1**; FukuyamaBench (48 test / 194 dev / 242 total), oMe-Gold (196), mech-USPTO-31k (200) | `si_experiments/S02_benchmarks_and_splits/data/*.json*`, `pilots_v4/p1_retrieval.py`, `p1_template_baseline/fukuyama_coverage.py`, `SUMMARY_P1.json` |
| **S3** | [`S03_evaluation_and_specialists/`](si_experiments/S03_evaluation_and_specialists/) | **Table 1**, **Fig. 1b**, **Tables S2–S4** (FlowER/ChRIMP vs LLMs, strict vs equiv matching, single-step $S_{k-1}$) | [`mechsight/evaluation.py`](mechsight/evaluation.py), `table1_recall_spec4090.py`, `pilots_v4/{p18_spec_compare,p40_equiv_rescore,p44_stepwise}.py`, `SUMMARY_STEPWISE.json` |
| **S4** | [`S04_causal_probes/`](si_experiments/S04_causal_probes/) | **Fig. 1c–d**, **Fig. S1**, **Tables S5–S9** (M3 reactive-site blocking, M2 citation exposure, M1/M4, homologues, RandSMILES) | `pilots_v4/{p32_m3_counterfactual,p31_m2_exposure,p33_m1_m4,p5_contamination,p22_extras}.py`, `exposure_rule.py`, `SUMMARY_M3.json`, `SUMMARY_M2.json`, `SUMMARY_M1_M4.json` |
| **S5** | [`S05_hidden_intermediate_bottleneck/`](si_experiments/S05_hidden_intermediate_bottleneck/) | **Fig. 2**, **Fig. S2**, **Fig. S3b–c**, **Tables S10–S11** (visible vs hidden bond events, 251 hidden taxonomy, oracles) | `inert_knowledge/{probe,visibility,phase2}.py`, `recall_probe/audit_gap.py`, `pilots_v4/{p4_inertness,p12_controls,p23_offline}.py`, `SUMMARY_P23_OFFLINE.json`, `SUMMARY_P4.json` |
| **S6** | [`S06_falsified_alternatives/`](si_experiments/S06_falsified_alternatives/) | **Fig. S3a**, **Table S12** (Mayr scale probe, 1-arrow search, shortlist, pairwise/listwise judges) | `scale_probe/probe.py`, `mechsight/activation.py`, `pilots_v4/{p3_selector,p11_mechagent,p12_controls}.py`, `SUMMARY_P12.json` |
| **S7** | [`S07_mechsight_auto/`](si_experiments/S07_mechsight_auto/) | **Fig. 3**, **Tables S13–S14** (**MechSight-Auto**, ablation ladder, 4-vendor cross-model transfer) | Standalone [`mechsight/auto.py`](mechsight/auto.py); `pilots_v4/{p14_confirm,p19_ladder,p46_backbone,p38_luna_t3,p17_argon_confirm}.py`, `backbone_rule.py`, `SUMMARY_P14_confirm.json` |
| **S8** | [`S08_rigor_controls_and_pericyclic/`](si_experiments/S08_rigor_controls_and_pericyclic/) | **Fig. S4**, **Tables S15–S16** (over-generation controls, 214 inserted-species audit, $\Delta_{\mathrm{bond}}\ge 3$ & pericyclic check) | `pilots_v4/{p23_offline,p42_auto_inserted_audit,p39_direct_gap}.py`, `smiles/map_check.py`, `auto_inserted_audit_and_examples.json`, `SUMMARY_DIRECT_GAP.json` |
| **S9** | [`S09_literature_corpus_curation/`](si_experiments/S09_literature_corpus_curation/) | **Fig. S5a** (436-paper 2026 OA curation, 285 post-cutoff split, 4,388-line modality & 50-paper PDF audit) | `lit150/`, `lit_more/`, `lit_old/`, `evidence_p0/`, `pilots_v4/{p29_gold_rebuild,p36_clue_audit}.py`, `pub_dates.json` |
| **S10** | [`S10_mechsight_guided/`](si_experiments/S10_mechsight_guided/) | **Table 2**, **Fig. 4a–b**, **Fig. S5b–c**, **Tables S17–S19** (**MechSight-Guided**, 285 post-cutoff studies, wet-lab-only controls) | Standalone [`mechsight/guided.py`](mechsight/guided.py); `v4_analyses_20261108/{lit_bench_rule,lit_bench_rule_wl,merged_report}.py`, `pilots_v4/{p31_guided71,p31_confirm_core,p37_nodft,p41_knowledge_strata}.py` |
| **S11** | [`S11_case_studies_and_per_paper/`](si_experiments/S11_case_studies_and_per_paper/) | **Fig. 4c**, **Tables S20–S21** (Wickens `PMC13449957`, Chen `PMC13321810`, Hu `PMC13171995`, 165 dev & 271 test rows) | `v4_analyses_20261108/bench_{strict,eq}/rows_{dev,test}.json`, `raw_outputs/pilots_v4_20261103/{lit172g,litnew_g}/` |
| **S12** | [`S12_prompt_templates/`](si_experiments/S12_prompt_templates/) | **Prompt Templates S12.1–S12.8** | `S12_*.txt`, [`verify_prompts.py`](si_experiments/S12_prompt_templates/verify_prompts.py), [`mechsight/prompts.py`](mechsight/prompts.py) |

### Raw-Output Archives (`results/raw_outputs_archives/`)

| Archive | Restored Directories | Contents |
| :--- | :--- | :--- |
| `01_benchmarks_confirm_and_ablation.tar.gz` | `T3_confirm/`, `T3_dev/`, `J3/`, `gold/`, `uspto31k/`, `spec_v5/`, `spec_v6_4090/`, `cards/` | Primary *gemini-3.8-flash* benchmark runs (FukuyamaBench, oMe-Gold, mech-USPTO-31k), ablation ladder, and FlowER/ChRIMP RTX 4090 outputs. |
| `02_causal_probes_and_crossmodel.tar.gz` | `argon/`, `bb_claude-sonnet-5-5-high/`, `bb_gpt-6_1-sol_high/`, `luna/`, `luna_high/`, `m1/`–`m4/`, `litoldg/`, `lit_old_argon/` | Causal probes (M1–M4) and cross-model transfer outputs for *claude-sonnet-5.5*, *gpt-6-luna* (high & no-reasoning), *gemini-4-argon*, and supplementary *gpt-6.1-sol*. |
| `02_crossmodel_deepseek_v4_1_flash.tar.gz` | `bb_deepseek-v4_1-flash_high/`, `bb_deepseek-v4_1-flash_nothink/` | Cross-model transfer outputs for *deepseek-v4.1-flash* (high reasoning & no-reasoning, Table S14). |
| `03_lit2026_dev_165.tar.gz` | `lit172g/`, `lit172g_nodft/`, `lit172g_p36/`, `lit150/`, `lit172_v7_real_20261105/`, `lit_gold_argon/` | 2026 literature development cohort ($n = 165$ studies) and wet-lab-only re-evaluations. |
| `04_lit2026_test_271_and_nodft.tar.gz` | `litnew_g/`, `litnew_g_nodft/`, `litnew_confirm_g71/`, `lit_scale/`, `lit_scale_gold/`, `supp_clue_audit_20261106/` | Pre-registered 2026 literature test cohort ($n = 271$ studies), wet-lab-only re-evaluations, and clue audits. |
| `05_pilots_v4_other_runs.tar.gz` | `g71_dev_20261106/`, `p34_pilot36_20261106/`, `p34b_a1_38_20261106/`, `silver_dev/`, `silver_conf/`, `stepwise_llm/`, `nodft_confirm_20261106/`, `supp_offline_20261106/`, `template/` | Auxiliary pilot, single-step LLM, oMe-Silver, and offline verification caches. |
| `06_early_exploration_storyline.tar.gz` | `03_early_exploration_storyline_package/` | Early diagnostic probes (template degeneration, inert-knowledge, hidden-intermediate visibility, and falsified interventions). |

---

## Mapping Between Internal Code Identifiers and Manuscript Terms

| Code / Folder Identifier | Manuscript / SI Term |
| :--- | :--- |
| `MechAgent` (`V5_rev1`) / `MechSightAuto` | **MechSight-Auto** |
| `A1_place` (`V5_place`) / `MechSightGuided` | **MechSight-Guided** |
| `T3_confirm` | FukuyamaBench held-out test set ($n = 48$) |
| `T3_dev` ($n = 47$), `J3` ($n = 147$) | FukuyamaBench expanded development ($n = 194$ non-overlapping dev) & diagnostic subsets ($242$ total reactions) |
| `gold` | oMe-Gold ($n = 196$) |
| `uspto31k` | mech-USPTO-31k in-template benchmark ($n = 200$) |
| `base_default` / `base_cot` / `SR2` / `SelfConsistency` | Direct / Chain-of-Thought / Self-Refinement (2 rounds) / Self-Consistency |
| `O_llm` / `MV7` | Direct + Clues / Majority Vote + Clues |
| `bb_claude-sonnet-5-5-high` / `luna_high`, `luna` / `bb_deepseek-v4_1-flash_{high,nothink}` / `argon` | Cross-model transfer backbones in Table S14 (*claude-sonnet-5.5*, *gpt-6-luna*, *deepseek-v4.1-flash*, *gemini-4-argon*; identifiers preserved to match archived raw logs) |
| `lit172g` (165) + `litnew_g` (271) | 2026 open-access literature corpus (436 studies; **285 post-cutoff studies**, April–November 2026) |
| `lit172g/A1_place/PMC13449957.json` | Figure 4c wet-lab case study (Wickens, *J. Am. Chem. Soc.* 2026, *148*, 31641) |
| `lit172g_nodft/`, `litnew_g_nodft/` | Pure wet-lab-only controls with all DFT statements removed |
| `spec_v6_4090/` | FlowER & ChRIMP native evaluations on NVIDIA RTX 4090 GPUs |

---

## Environments, Sanitization & License

- **Pinned Python environments** are documented under [`si_experiments/S01_environment_and_stats/`](si_experiments/S01_environment_and_stats/) (`requirements_llm_py3.14.txt` for Python 3.14 + RDKit `2026.03.5` + `openai` + `numpy`; `requirements_rxnmapper.txt` + `python_version_rxnmapper.txt` for Python 3.10 + `rxnmapper==0.4.3`).
- **Privacy & sanitization**: All private gateway hostnames, internal proxy IPs, GPU server addresses, and local developer paths were replaced with generic placeholders (`LLM-GATEWAY-1.example`, `LLM-GATEWAY-2.example`, environment variables); no scientific data, prompts, model predictions, or scores were altered, and zero third-party copyrighted article PDFs are redistributed.
- **License**: Original code, prompts, and analysis artifacts are released under the **MIT License** ([`LICENSE`](LICENSE)). Third-party benchmarks (oMeBench under `reproduction/external/oMeBench/`, FukuyamaBench, mech-USPTO-31k) retain their original licenses and attributions.
- **Citation**: See [`CITATION.cff`](CITATION.cff).
