# SI Section S9: Curation and Quality Audits of the 2026 Literature Corpus

Covers **Supporting Information Section S9** (S9.1 Literature retrieval and multimodal PDF extraction with RDKit conservation verification; S9.2 4,388-line observation-modality classification and 50-paper PDF audit), backing **Figure S5a**.

## Underlying Scripts & Data

- Europe PMC retrieval & selection: `reproduction/MechAgent/experiments/{evidence_p0,lit150,lit_more,lit_old}/`
- Multimodal full-PDF extraction + RDKit validation (`gemini-4-argon`): `reproduction/MechAgent/experiments/pilots_v4/p29_gold_rebuild.py`
- Observation-modality classification (4,388 lines) & 50-paper PDF audit: `reproduction/MechAgent/experiments/pilots_v4/p36_clue_audit.py report` (archived in `results/raw_outputs_archives/04_lit2026_test_271_and_nodft.tar.gz` → `supp_clue_audit_20261106/SUMMARY_CLUE_AUDIT.json`)
- Corpus metadata & publication dates: `results/canonical_analyses/v4_analyses_20261108/pub_dates.json`

## Verification

```bash
python si_experiments/S09_literature_corpus_curation/reproduce.py
python si_experiments/S09_literature_corpus_curation/reproduce.py --recompute
```
