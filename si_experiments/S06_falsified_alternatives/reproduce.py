"""Verify and optionally recompute SI Section S6 (Table S12, Fig. S3a)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from si_experiments._runner import (  # noqa: E402
    CANONICAL_ROOT,
    PAPER_SUMMARIES_ROOT,
    load_json,
    require_raw_outputs,
    run_pilots_script,
)


def verify(*, recompute: bool = False) -> None:
    scale = load_json(PAPER_SUMMARIES_ROOT / "scale_probe_20261031/SUMMARY.json")
    act = load_json(PAPER_SUMMARIES_ROOT / "activation_pilot_20261103/SUMMARY.json")
    p3 = load_json(CANONICAL_ROOT / "pilots_v4_summaries/SUMMARY_P3.json")
    p11 = load_json(CANONICAL_ROOT / "pilots_v4_summaries/SUMMARY_P11.json")
    p12 = load_json(CANONICAL_ROOT / "pilots_v4_summaries/SUMMARY_P12.json")

    assert scale["steps"] > 0
    assert "T3_dev" in p3 and "J3" in p3
    assert "T3_dev" in p11 and "J3" in p11
    assert "T3_dev" in p12 and "J3" in p12
    assert act["cases"] > 0

    if recompute:
        require_raw_outputs()
        run_pilots_script("reproduction/MechAgent/experiments/pilots_v4/p3_selector.py", ["--analyze-only"])
        run_pilots_script("reproduction/MechAgent/experiments/pilots_v4/p11_mechagent.py", ["--phase", "analyze"])

    print(
        f"[S06 OK] Verified falsified alternative intervention summaries: "
        f"scale_probe steps={scale['steps']}, activation cases={act['cases']}, P3/P11/P12 dev controls."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    verify(recompute=ap.parse_args().recompute)
