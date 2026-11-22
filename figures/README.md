# MechSight Figures (`figures/`)

Publication SVG figures, ChemDraw (`.cdxml`) structures, and reproducible Python build scripts for all main-text figures (**Figures 1–4**, **TOC Graphic**) and Supporting Information figures (**Figures S1–S5**).

## Directory Layout

| Path | Description |
| :--- | :--- |
| `final_svgs/` | Final publication SVGs matching the submitted *Angewandte Chemie* manuscript (`fig{1..4}_*_v3.svg`, `toc_graphic.svg`, `figS{1..5}_*.svg`). |
| `main_figures/build/` | Build scripts for main-text Figures 1–4 (`fig1.py`–`fig4.py`, `fig4c_cases.py`, `toc.py`, `figkit3.py`, `render.py`). |
| `main_figures/mols/`, `main_figures/data/` | ChemDraw `.cdxml`/`.svg` molecular graphics and auxiliary benchmark JSON slices for Figures 1–4. |
| `main_figures/out/` | Frozen rendered SVG/PDF/PNG outputs from `main_figures/build/`. |
| `build_si_figures.py` | Build script for Supporting Information Figures S1–S5. |
| `figkit.py`, `chemdraw.py`, `chemdraw_cdxml/` | Shared ACS-1996 chemical drawing toolkit (`curly()` electron-pushing arrows, `atom_xy()`, CDXML exporter). |
| `data/`, `si_mols/` | Frozen figure data (`data/figdata.json`) and SI molecular `.svg`/`.cdxml` assets. |

## Re-Rendering Figures Offline

1. First unpack the raw outputs and canonical analyses from the repository root:
   ```bash
   ./unpack_raw_outputs.sh
   ```
2. Re-render **Figures S1–S5** (writes to `figures/generated/si/` by default, or override with `SI_FIG_OUT`):
   ```bash
   python figures/build_si_figures.py
   ```
3. Re-render **Figures 1–4** and the **TOC graphic** (set `FIG_OUT` to write into a scratch output folder, or omit to write into `figures/main_figures/out/`):
   ```bash
   FIG_OUT=figures/generated/main python figures/main_figures/build/fig1.py
   FIG_OUT=figures/generated/main python figures/main_figures/build/fig2.py
   FIG_OUT=figures/generated/main python figures/main_figures/build/fig3.py
   FIG_OUT=figures/generated/main python figures/main_figures/build/fig4.py
   FIG_OUT=figures/generated/main python figures/main_figures/build/toc.py
   ```
   All scripts verify every plotted value with inline `assert` statements against the frozen canonical JSON summaries.
