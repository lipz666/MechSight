# SI Section S6: Alternative Interventions That Failed

Covers **Supporting Information Section S6** (**Table S12**, **Figure S3a**):
1. Quantitative Mayr $N/E$ and $\mathrm{p}K_\mathrm{a}$ scales (`reproduction/MechAgent/experiments/scale_probe/probe.py`)
2. Rule-enumerated single-arrow search (`reproduction/MechAgent/experiments/step_search/`)
3. Enumerate-then-shortlist (`reproduction/MechAgent/experiments/heldout_fukuyama/`)
4. Pairwise pathway judge (`reproduction/MechAgent/experiments/agent_v2/j3_dev.py`, `j4_run.py`)
5. Stepwise list-then-select (`reproduction/MechAgent/experiments/recall_probe/select_own.py`)
6. Listwise pool judge without splicing (`reproduction/MechAgent/experiments/pilots_v4/p3_selector.py`, `p11_mechagent.py`)

## Verification

```bash
python si_experiments/S06_falsified_alternatives/reproduce.py
python si_experiments/S06_falsified_alternatives/reproduce.py --recompute
```
