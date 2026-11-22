# SI Section S5: Hidden-Intermediate Bottleneck, Pathway-Length Controls and Single-Step Audit

Covers **Supporting Information Section S5** (S5.1 Visible vs hidden bond events & length controls; S5.2 Single-step granularity audit on 115 steps; S5.3 Tracking 251 omitted hidden intermediates, 8 chemical families, oracle controls), backing **Main Figure 2**, **Figures S2 and S3b–c**, and **Tables S10–S11**.

## Underlying Scripts & Frozen Outputs

- **Table S10 (Visible vs hidden events & 115-step granularity audit)**: `reproduction/MechAgent/experiments/inert_knowledge/visibility.py`, `recall_probe/audit_gap.py`, `pilots_v4/p23_offline.py` → `SUMMARY_visibility.json`, `recall_probe_20261101/gap_audit/SUMMARY.json`, `SUMMARY_P23_OFFLINE.json`
- **Table S11 (Inert-knowledge probes, 251 hidden taxonomy, reference-intermediate oracles)**: `inert_knowledge/{probe,phase2}.py`, `pilots_v4/{p4_inertness,p12_controls,p14_confirm,p16_supplement_g38}.py` → `inert_knowledge_20261102/SUMMARY.json`, `SUMMARY_P4.json`, `SUMMARY_P23_OFFLINE.json`, `SUMMARY_GOLD_ORACLE.json`, `SUMMARY_P14_confirm.json`

## Verification

```bash
python si_experiments/S05_hidden_intermediate_bottleneck/reproduce.py
python si_experiments/S05_hidden_intermediate_bottleneck/reproduce.py --recompute
```
