# SI Section S3: Four Evaluation Levels, Matching Rules and Specialist-Model Comparison

Covers **Supporting Information Section S3** (S3.1 Formal evaluation definitions; S3.2 Specialist-model execution and cross-paradigm comparison; S3.3 Single-step accuracy given $S_{k-1}$), backing **Main Table 1**, **Main Figure 1b**, and **Tables S2–S4**.

## Underlying Scripts & Frozen Outputs

- Standalone formal evaluation module: [`mechsight/evaluation.py`](../../mechsight/evaluation.py)
- Main Table 1 & Table S3: `results/canonical_analyses/v4_analyses_20261108/table1_recall_spec4090.py` → `table1_recall_spec4090.json`, `reproduction/MechAgent/experiments/pilots_v4/p40_equiv_rescore.py` → `v3_analyses_20261107/equiv_table1_spec4090.json`
- Table S2 (FlowER/ChRIMP vs LLMs): `reproduction/MechAgent/experiments/pilots_v4/p18_spec_compare.py` (`SPEC_DIR=spec_v6_4090`) → `SUMMARY_P18_SPEC_spec_v6_4090.json`, `figures/main_figures/data/intemplate_equiv_spec4090.py`
- Table S4 (Single-step accuracy given $S_{k-1}$): `reproduction/MechAgent/experiments/pilots_v4/p44_stepwise.py score` → `stepwise_20261108/SUMMARY_STEPWISE.json`
- RTX 4090 specialist runner scripts: `reproduction/MechAgent/experiments/pilots_v4/spec_server/`

## Verification

```bash
python si_experiments/S03_evaluation_and_specialists/reproduce.py
python si_experiments/S03_evaluation_and_specialists/reproduce.py --recompute
```
