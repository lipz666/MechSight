# SI Section S1: General Information, Software, and Statistical Analysis

Covers **Supporting Information Section S1** (S1.1 Language models and decoding parameters; S1.2 Cheminformatics software and hardware; S1.3 Statistical analysis).

## Pinned Environments in This Directory

- `requirements_llm_py3.14.txt`: Primary Python 3.14 environment (`rdkit==2026.03.5`, `openai`, `numpy`, `networkx`, `matplotlib`) used for LLM inference, RDKit scoring, statistics, and figure rendering.
- `requirements_rxnmapper.txt` + `python_version_rxnmapper.txt`: Isolated Python 3.10.22 environment (`rxnmapper==0.4.3`, `rdkit==2024.03.3`, `torch`) used for atom mapping.

## Verification

```bash
python si_experiments/S01_environment_and_stats/reproduce.py
```
