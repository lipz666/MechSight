# SI Section S8: MechSight-Auto Rigor Controls: Inserted-Species Audit, Bond-Change Rule Diagnostics and Pericyclic Check

Covers **Supporting Information Section S8** (S8.1 Over-generation controls, automated audit of 214 inserted species, and repair taxonomy; S8.2 Validation of $\Delta_{\mathrm{bond}} \ge 3$ rule & concerted pericyclic audit; S8.3 Fritsch–Buttenberg–Wiechell case), backing **Figure S4** and **Tables S15–S16**.

## Underlying Scripts & Frozen Outputs

- **Over-generation & $\Delta_{\mathrm{bond}}$ distribution (`Tables S15–S16`, `SUMMARY_P23_OFFLINE.json`)**: `reproduction/MechAgent/experiments/pilots_v4/p23_offline.py`
- **214 Inserted-species audit & FBW showcase (`Table S15`, `auto_inserted_audit_and_examples.json`)**: `reproduction/MechAgent/experiments/pilots_v4/p42_auto_inserted_audit.py`
- **Single-trajectory $\Delta_{\mathrm{bond}} \ge 3$ control (`Table S16`, `SUMMARY_DIRECT_GAP.json`)**: `reproduction/MechAgent/experiments/pilots_v4/p39_direct_gap.py --phase analyze`
- **RXNMapper step agreement validation (`96.8%` on 857 steps)**: `reproduction/MechAgent/experiments/smiles/map_check.py`

## Verification

```bash
python si_experiments/S08_rigor_controls_and_pericyclic/reproduce.py
python si_experiments/S08_rigor_controls_and_pericyclic/reproduce.py --recompute
```
