# SI Section S2: Benchmark Construction and Study Design

Covers **Supporting Information Section S2** (S2.1 Reaction benchmarks and literature cohorts; S2.2 Study design and evaluation protocols; **Table S1**).

## Benchmark Data Included Here (`data/`)

- `fukuyama_splits.json`: Pre-registered split of the 95 fully atom-mapped FukuyamaBench reactions into **47 dev (`T3_dev`)** and **48 held-out test (`T3_confirm`)** reactions (seed `20261103`).
- `fukuyama_clean.jsonl`: 147 FukuyamaBench diagnostic reactions (`J3`).
- `fukuyama_heldout.jsonl` + `fukuyama_registration.json`: Held-out FukuyamaBench PGL reactions (`T3`).
- `p23_cases.json`: Diagnostic subset index.
- `ome_gold_pgl_cases.jsonl`: PGL-converted oMe-Gold reactions (with official oMeBench `oMe_Gold.json` in `reproduction/external/oMeBench/data/`).

## Underlying Scripts (`reproduction/`)

- `reproduction/MechAgent/experiments/pilots_v4/p1_retrieval.py` (Table S1 template retrieval, `SUMMARY_P1.json`)
- `reproduction/MechAgent/experiments/p1_template_baseline/fukuyama_coverage.py` (0/60 pathways & 0/548 step windows covered on FukuyamaBench)
- `reproduction/MechAgent/experiments/pilots_v4/p6_baselines.py` (Direct & CoT baselines)

## Verification

```bash
python si_experiments/S02_benchmarks_and_splits/reproduce.py
python si_experiments/S02_benchmarks_and_splits/reproduce.py --recompute   # after ./unpack_raw_outputs.sh
```
