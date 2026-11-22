"""Verify and optionally recompute SI Section S9 (2026 literature curation & audits, Fig. S5a)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from si_experiments._runner import (  # noqa: E402
    CANONICAL_ROOT,
    RAW_OUTPUTS_V4,
    load_json,
    require_raw_outputs,
    run_pilots_script,
)


def verify(*, recompute: bool = False) -> None:
    r_dev = load_json(CANONICAL_ROOT / "v4_analyses_20261108/bench_eq/rows_dev.json")
    r_test = load_json(CANONICAL_ROOT / "v4_analyses_20261108/bench_eq/rows_test.json")
    pub_dates = load_json(CANONICAL_ROOT / "v4_analyses_20261108/pub_dates.json")

    assert len(r_dev) == 165
    assert len(r_test) == 271
    all_rows = r_dev + r_test
    assert len(all_rows) == 436
    post_cutoff = [r for r in all_rows if r["date"] >= "2026-04"]
    pre_cutoff = [r for r in all_rows if r["date"] < "2026-04"]
    assert len(post_cutoff) == 285
    assert len(pre_cutoff) == 151
    assert sum(len(r["strata"]) for r in post_cutoff) == 1571
    assert len(pub_dates) >= 436

    audit_msg = ""
    if recompute:
        require_raw_outputs()
        run_pilots_script("reproduction/MechAgent/experiments/pilots_v4/p36_clue_audit.py", ["report"])
        audit = json.loads((RAW_OUTPUTS_V4 / "supp_clue_audit_20261106/SUMMARY_CLUE_AUDIT.json").read_text())["summary"]
        assert audit["papers_classified"] == 436
        assert audit["lines"] == 4388
        assert audit["experimental"] == 4009
        assert audit["computational"] == 302
        assert audit["mixed"] == 50
        assert audit["author_conclusion"] == 27
        assert audit["audit"]["papers"] == 50
        audit_msg = f"; 4,388-line audit verified ({audit['experimental']}/4388 wet-lab, 50-paper PDF audit ok)"

    print(
        f"[S09 OK] 2026 corpus: {len(all_rows)} papers (dev={len(r_dev)}, test={len(r_test)}; "
        f"post-cutoff Apr-Nov={len(post_cutoff)}, Jan-Mar={len(pre_cutoff)}; 1,571 post-cutoff intermediates)"
        + audit_msg
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    verify(recompute=ap.parse_args().recompute)
