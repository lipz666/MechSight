# SI Section S10: MechSight-Guided: Experimental-Evidence Anchoring, Post-Cutoff Elucidation and Controls

Covers **Supporting Information Section S10** (S10.1 Implementation of MechSight-Guided; S10.2 Out-of-time generalization across 285 post-cutoff studies, subset analyses & wet-lab-only controls; S10.3 Knowledge strata across 1,571 intermediates; S10.4 Full 436-paper corpus & wet-lab-only controls), backing **Main Table 2**, **Main Figure 4a–b**, **Figure S5b–c**, and **Tables S17–S19**.

## Code Mapping

- **Standalone Main-Method Implementation**: [`mechsight/guided.py`](../../mechsight/guided.py) (`MechSightGuided`)
- **Main Table 2 & Tables S17–S18 (`merged_apr_oct.json`)**: `results/canonical_analyses/v4_analyses_20261108/{lit_bench_rule,lit_bench_rule_wl,merged_report}.py`
- **Knowledge Strata (`Table S18`, `knowledge_strata_{dev,test}.json`)**: `reproduction/MechAgent/experiments/pilots_v4/p41_knowledge_strata.py`
- **Full 436-paper corpus & wet-lab-only controls (`Table S19`)**: `reproduction/MechAgent/experiments/pilots_v4/{p31_guided71,p31_confirm_core,p37_nodft,p37b_case_nodft}.py` → `SUMMARY_G71.json`, `nodft_case_20261108/SUMMARY.json`

## Verification

```bash
python si_experiments/S10_mechsight_guided/reproduce.py
python si_experiments/S10_mechsight_guided/reproduce.py --recompute
```
