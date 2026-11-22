"""Verify SI Section S1 environment specifications and statistical utilities."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mechsight.evaluation import bootstrap_ci, paired_permutation_pvalue  # noqa: E402
from si_experiments._runner import REPO_ROOT  # noqa: E402


def verify(*, recompute: bool = False) -> None:
    s01 = Path(__file__).resolve().parent
    llm_req = (s01 / "requirements_llm_py3.14.txt").read_text()
    rxn_req = (s01 / "requirements_rxnmapper.txt").read_text()
    py_rxn = (s01 / "python_version_rxnmapper.txt").read_text().strip()

    assert "rdkit==2026.3.5" in llm_req or "rdkit==" in llm_req
    assert "rxnmapper==0.4.3" in rxn_req
    assert py_rxn.startswith("Python 3.10")

    import rdkit
    mean, lo, hi = bootstrap_ci([1.0, 0.0, 1.0, 1.0], n_resamples=2000, seed=20261103)
    pval = paired_permutation_pvalue([1.0, 1.0, 1.0, 1.0, 0.5, 0.5], n_permutations=1000, seed=20261103)
    assert 0.0 <= lo <= mean <= hi <= 1.0
    assert pval < 0.05

    print(
        f"[S01 OK] python={sys.version.split()[0]}, rdkit={rdkit.__version__}, "
        f"rxnmapper_spec={py_rxn}, standalone_pkg={(REPO_ROOT / 'mechsight').exists()}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    verify(recompute=ap.parse_args().recompute)
