# SI Section S12: Prompt Templates

Verbatim prompt templates corresponding to **Supporting Information Sections S12.1–S12.8**:

- `S12_1_auto_enumerate.txt`: MechSight-Auto hidden-intermediate-directed pathway enumeration (`p2_enumerate.py`)
- `S12_2_auto_complete.txt`: MechSight-Auto targeted intermediate completion at flagged shortcuts (`p11_mechagent.py`)
- `S12_3_auto_revise.txt`: MechSight-Auto cross-draft splicing pathway revision (`p11_mechagent.py`)
- `S12_4_guided_hypothesis.txt`: MechSight-Guided observation-driven intermediate hypothesis generation (`p28_lit172_v7.py`)
- `S12_5_guided_revise.txt`: MechSight-Guided consensus-anchored pathway revision (`p31_guided71.py`)
- `S12_6_guided_insert.txt`: MechSight-Guided non-destructive consensus anchor insertion (`p34_pilot36.py`)
- `S12_7_baselines_and_controls.txt`: Self-refinement, Direct + Clues, reference-intermediate oracle controls, and step-type labeling (`p12_controls.py`, `p8_observations.py`, `p15_relabel.py`)
- `S12_8_pdf_extraction.txt`: Multimodal full-PDF literature extraction (`p29_gold_rebuild.py`)

## Verification

```bash
python si_experiments/S12_prompt_templates/verify_prompts.py
```
