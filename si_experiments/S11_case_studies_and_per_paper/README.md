# SI Section S11: Literature Case-Study Trajectories and Per-Paper Results

Covers **Supporting Information Section S11** (S11.1 Detailed trajectories for `PMC13449957`, `PMC13321810`, `PMC13171995`; S11.2 Per-paper results across the 2026 dev $n = 165$ and test $n = 271$ cohorts), backing **Main Figure 4c** and **Tables S20–S21**.

## Data & Raw Case Outputs

- Per-paper scoring tables (`Tables S20–S21`): `results/canonical_analyses/v4_analyses_20261108/bench_{strict,eq}/rows_{dev,test}.json` and `pub_dates.json`
- Case-study raw trajectories (available after `./unpack_raw_outputs.sh`):
  - `results/raw_outputs/pilots_v4_20261103/lit172g/A1_place/PMC13449957.json` (Wickens, *J. Am. Chem. Soc.* 2026, *148*, 31641 — Main Fig. 4c)
  - `results/raw_outputs/pilots_v4_20261103/lit172g/A1_place/PMC13321810.json` (Chen et al., *Chem. Sci.* 2026, *17*, 15443)
  - `results/raw_outputs/pilots_v4_20261103/lit172g/V5_rev2/PMC13171995.json` or `V5_rev1/PMC13171995.json` (Hu et al., *Nat. Commun.* 2026, *17*, 4350)

## Verification

```bash
python si_experiments/S11_case_studies_and_per_paper/reproduce.py
python si_experiments/S11_case_studies_and_per_paper/reproduce.py --recompute
```
