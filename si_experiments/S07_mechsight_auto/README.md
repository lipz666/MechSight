# SI Section S7: MechSight-Auto: Prior-Scaffold Construction, Ablations and Cross-Model Generalization

Covers **Supporting Information Section S7** (S7.1 Implementation of MechSight-Auto; S7.2 Benchmark evaluation, cumulative ablations, equal-compute controls and token accounting; S7.3 Cross-model transfer across *claude-sonnet-5.5*, *gpt-6-luna*, *deepseek-v4.1-flash*, and *gemini-4-argon*), backing **Main Figure 3**, **Table 1**, and **Tables S13–S14**.

## Code Mapping

- **Standalone Main-Method Implementation**: [`mechsight/auto.py`](../../mechsight/auto.py) (`MechSightAuto`) & [`mechsight/bond_checker.py`](../../mechsight/bond_checker.py) (`BondChangeChecker`)
- **Benchmark Confirmation (`Table S13`, `SUMMARY_P14_confirm.json`)**: `reproduction/MechAgent/experiments/pilots_v4/p14_confirm.py`
- **Cumulative Ablation Ladder (`Table S13`, `SUMMARY_P19_LADDER.json`)**: `reproduction/MechAgent/experiments/pilots_v4/p19_ladder.py --phase analyze`
- **Cross-Model Transfer (`Table S14`)**:
  - `results/canonical_analyses/v4_analyses_20261108/backbone_rule.py` → `backbone_rule.json`
  - `reproduction/MechAgent/experiments/pilots_v4/p46_backbone.py` (*claude-sonnet-5.5*, *deepseek-v4.1-flash* high & no-reasoning, *gpt-6.1-sol*)
  - `reproduction/MechAgent/experiments/pilots_v4/p38_luna_t3.py` (*gpt-6-luna* high & no-reasoning)
  - `reproduction/MechAgent/experiments/pilots_v4/p17_argon_confirm.py` (*gemini-4-argon*)

## Verification

```bash
python si_experiments/S07_mechsight_auto/reproduce.py
python si_experiments/S07_mechsight_auto/reproduce.py --recompute
```
