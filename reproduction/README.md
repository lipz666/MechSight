# Frozen Reproduction Scripts (`reproduction/`)

This directory preserves the original research scripts and third-party benchmark package (`external/oMeBench/`) used to generate and analyze all results reported in the manuscript and Supporting Information.

- **Why keep these scripts untouched internally?** Many analysis scripts in `reproduction/MechAgent/experiments/pilots_v4/` and `results/canonical_analyses/` import each other by exact relative module names and write/read paths under `reproduction/MechAgent/outputs/`. Keeping this internal layout frozen guarantees zero import breakage and exact numerical provenance.
- **For the clean, standalone main method**, see [`mechsight/`](../mechsight/) at the repository root (`MechSightAuto` and `MechSightGuided`).
- **For SI-section-aligned entrypoints**, see [`si_experiments/`](../si_experiments/README.md), which organizes every benchmark, probe, control, ablation, and offline recomputation by **Supporting Information Sections S1–S12**.

## Restoring Unpacked Outputs for `reproduction/`

Run `./unpack_raw_outputs.sh` from the repository root. It verifies all 7 archives in `results/raw_outputs_archives/`, extracts `results/raw_outputs/`, and creates the symlinks inside `reproduction/MechAgent/outputs/` (including the 6 FukuyamaBench/oMe-Gold benchmark split files stored in `si_experiments/S02_benchmarks_and_splits/data/`).
