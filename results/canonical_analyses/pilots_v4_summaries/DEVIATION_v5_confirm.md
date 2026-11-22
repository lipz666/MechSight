# Deviation from REGISTRATION_v5_confirm.json (2026-11-04)

**What happened:** after the confirmation run, oMe-Gold S_partial was 3.2 for Mech-Agent, 1.7 for M_ms_none and 8.0 for Self-refine, against 49.6 for Direct.

**Cause:** a format problem, not a mechanism problem. The revision prompts (Mech-Agent, M_ms_none, Self-refine) did not include the official oMeBench type/subtype list. Only 3.4%, 1.1% and 9% of their step labels are official pairs, against 100% for Direct, which uses the official prompt. Official oMeS requires a subtype match, so these paths score ~0 regardless of their intermediates.

**Change (post hoc, declared):** `p15_relabel.py` makes one format-only call per final path that assigns an official pair to every step. The intermediate SMILES are copied from the original path in code, so species-level endpoints (all_present, including the primary endpoint) are unchanged by construction. The same step is applied to every arm that went through a revision prompt.

**Reporting:** the primary endpoint (pooled all_present) is unaffected. S_partial is reported with the relabelled paths, and this note goes in the SI. The original S_partial values stay in SUMMARY_P14_confirm.json history (above). For any future set, the revision prompts will carry the official list from the start.
