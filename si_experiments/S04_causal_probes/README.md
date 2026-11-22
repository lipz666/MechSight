# SI Section S4: Causal Probes of Chemical Knowledge versus Memorization

Covers **Supporting Information Section S4** (S4.1 Reactive-site blocking across 84 triplets; S4.2 Citation-exposure stratification; S4.3 Bibliographic recall vs mechanistic reasoning & counterfactual experimental clues; S4.4 Homologue perturbation & randomized SMILES), backing **Main Figure 1c–d**, **Figure S1**, and **Tables S5–S9**.

## Underlying Scripts & Frozen Outputs

- **Table S5 (M3 reactive-site blocking, $n=84$ triplets)**: `reproduction/MechAgent/experiments/pilots_v4/p32_m3_counterfactual.py --phase analyze` → `results/paper_summaries/pilots_v4_20261103/m3_counterfactual_20261106/SUMMARY_M3.json`
- **Tables S6–S7 (M2 citation exposure & covariate adjustment)**: `results/canonical_analyses/v4_analyses_20261108/exposure_rule.py` → `exposure_unified.json`; `reproduction/MechAgent/experiments/pilots_v4/p31_m2_exposure.py --analyze-only` → `m2_exposure_20261106/SUMMARY_M2.json`
- **Table S8 (M1 bibliographic recall & M4 counterfactual polar override, $n=250$)**: `reproduction/MechAgent/experiments/pilots_v4/p33_m1_m4.py --analyze-only` → `m1_m4_20261106/SUMMARY_M1_M4.json`
- **Table S9 (Homologue & randomized SMILES probes)**: `reproduction/MechAgent/experiments/pilots_v4/{p5_contamination,p22_extras,ome_homologs}.py` → `SUMMARY_P5.json`, `SUMMARY_P22_RANDSMI.json`

## Verification

```bash
python si_experiments/S04_causal_probes/reproduce.py
python si_experiments/S04_causal_probes/reproduce.py --recompute
```
