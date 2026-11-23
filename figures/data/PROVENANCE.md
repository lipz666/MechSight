# Figs. 1–3 + ToC for the MechSight manuscript: data provenance and corrections to the draft (2026-11-04)

Every plotted number comes from `data/figdata.py` → `data/figdata.json`, which reads only frozen result files.
Rebuild a figure: `python3 data/figdata.py`, then `python3 <fig dir>/build/fig*.py`, then render with `render.py`.
Structures: `chemdraw.py` (ChemDraw-style, ACS 1996, 69 %); editable copies in `chemdraw_cdxml/`.

## Where each panel's numbers come from
| Panel | Source |
|---|---|
| 1a retrieval 81.3 [72.5, 88.8] | `01_Problem_Template_Degeneration/outputs/summary.json` (retrieval, L1, n = 80); template coverage on Fukuyama 0/60 (`fukuyama_coverage.json`) |
| 1a FlowER | **88.8 is the published value** (open marker); our re-run on its test set gives 55.9 (`flower_template_summary.json`, n = 1000). Out of template: 0/5 on T3_confirm (`SUMMARY_P18_SPEC.json`), 0/45 on the diagnostic set (`COMPARE_1a.json`) |
| 1a ChRIMP 11.1 | `SUMMARY_P18_SPEC.json`: only **9/48** T3_confirm reactions covered |
| 1a LLM Direct 44.4 → 27.1; MechSight-Auto 45.8 | `publication_exp1_summary.json` (MechFinder L1); `SUMMARY_P14_confirm.json` (T3_confirm complete pathway) |
| 1b | `contamination_20261103/SUMMARY.json` (109 pairs); `SUMMARY_P5.json` b_J3_mechanism heavy_hidden_recall (n = 74) |
| 1c strata | `SUMMARY_P14_confirm.json` (T3), `SUMMARY_GOLD_ORACLE.json` (oMe) |
| 1c inert 57 % / blind spot 43 %, resample 7 % | `inert_knowledge_20261102/REPORT.md`: 42 missed heavy-atom hidden events; 57.1 % [42.5, 72.2] written in free text; resampling recovers 7.1 % [0, 18.6] |
| 1d, 2b +visible / +hidden GT | T3: `SUMMARY_P14_confirm.json`; oMe: `SUMMARY_GOLD_ORACLE.json`; pooled: `data/pooled_oracle.json` (computed by `data/pooled_oracle_compute.py`) |
| 2a | J3-B057 run outputs; dev gap flag 23/77 vs 65/116 |
| 2b, 2c | `SUMMARY_P14_confirm.json`, `SUMMARY_GOLD_ORACLE.json`, `pooled_oracle.json` |
| 3a | `SUMMARY_ARGON_CONFIRM.json` (T3 only; the oMe run `p17_argon_confirm.py --sets gold` was still running) |
| 3b | `p34b_a1_38_20261106/TABLE2_DEV_A1.json` (2026-11-06; one scorer `p31 T.score` for every arm; Guided = A1, frozen). Complete pathway = strict recall 1, denominator 165 |
| 3c top | PMC13530165 = Kong & Mu, *Nat. Commun.* 2026, doi 10.1038/s41467-026-75467-x. Direct 2/3 → Auto 3/3; the silicate was proposed by Stage ② (`lit/V5_comp1`, state C1.S2). Evidence pill: the paper's deuterium labelling (>99 % D) and KIE 2.0 |
| 3c bottom | (replaced 2026-11-06) PMC13321810 = Chen, Tang, Bai, Huang, *Chem. Sci.* 2026, 17, 15443, doi 10.1039/d6sc03721f; dev set, argon gold, 2 author intermediates. Hits from `TABLE2_DEV_A1.json` per-paper strict recall (Direct 0/2, Auto 0/2, Direct+Clues 1/2, Guided 2/2); per-intermediate marks from `lit172g/base_default`, Auto final, `lit172g/O_llm`, `lit172g/A1_place` with `K.species_keys`. Triplet 1a* support 4/7 in `G.proposals`. The former case PMC13036046 was dropped: under the argon gold Direct+Clues is already 3/3 |

## Corrections the manuscript text needs (the figures follow the data, not the draft)
1. **"MechSight-Guided (Visible / Hidden-1 / Hidden-Full)" in Table 1 and the Fig. 2 legend.** These arms were run with the oracle prompt on top of Direct (`p12_controls` / `p16 gold_oracle`), not through the MechSight pipeline. The figures therefore label them "+visible GT" / "+hidden GT (reference)". The only genuine MechSight-Guided runs are the literature arm `V5_obs`. Either rename these arms in the text, or run the GT intermediates through the V5 pipeline before calling them MechSight-Guided.
2. **Hidden-1 on T3_confirm (43.8 %, +16.7 [6.2, 29.2]) has no source.** There is no `T3_confirm/O_hid1` run. The only Hidden-1 data are on the development sets (T3_dev 40.4 %, +12.8 [0, 25.5]; J3 56.8 %). The arm is left out of Fig. 2b.
3. **"57–75 % inert knowledge".** 74.8 % matches the pooled `M_ms_none` heavy-hidden recall on T3_dev + J3 (`SUMMARY_P11.json`). That is a pipeline's output recall, not the share of missed intermediates the model knows. The pre-registered number is 57.1 % [42.5, 72.2] (gemini, PGL). Other models and representations are lower (gemini SMILES 38.5 %, gpt-6-luna PGL 24 %, argon SMILES 13 %). Fig. 1c shows 57 / 43 only.
4. **Literature O_llm recall gain.** The draft gives +11.9 [3.7, 20.9]; the file gives +12.0 [1.6, 22.6].
5. **PMC13036046 narrative is inverted.** MechSight-Auto *recovered* the first intermediate (as the ester ketyl radical anion) and the α-fluoro radical. It *missed* the tertiary radical adduct, because it wrote the alkene chain one CH2 too long (`CCCC[C](C)…` instead of `CCC[C](C)…`). MechSight-Guided then wrote the correct adduct. Also, the author SMILES for intermediate (i) has a four-bond carbanion carbon (a valence error); it is drawn as the arene radical anion.
6. **PMC13530165 conditions.** The paper uses K2CO3 or DBU, not "TBD/K2CO3".
7. **FlowER in-template "88.8 %"** is the published number; our re-run gives 55.9 %. Say which one you quote.
8. **Small-n specialist comparisons.** ChRIMP covers 9 and FlowER 5 of the 48 T3_confirm reactions. On those subsets LLM Direct scored 22.2 % (ChRIMP subset) and 0 % (FlowER subset).
