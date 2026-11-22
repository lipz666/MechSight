"""Verify SI Section S11 (case studies and per-paper 165 dev / 271 test results, Tables S20-S21)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from si_experiments._runner import (  # noqa: E402
    CANONICAL_ROOT,
    RAW_OUTPUTS_V4,
    load_json,
    require_raw_outputs,
)


def verify(*, recompute: bool = False) -> None:
    r_dev = {r["id"]: r for r in load_json(CANONICAL_ROOT / "v4_analyses_20261108/bench_eq/rows_dev.json")}
    r_test = {r["id"]: r for r in load_json(CANONICAL_ROOT / "v4_analyses_20261108/bench_eq/rows_test.json")}
    assert len(r_dev) == 165 and len(r_test) == 271

    for cid in ("PMC13449957", "PMC13321810", "PMC13171995"):
        assert cid in r_dev

    wickens = r_dev["PMC13449957"]
    assert wickens["date"] >= "2026-04"
    assert len(wickens["strata"]) == 6
    assert sum(s["Direct"] for s in wickens["strata"]) == 2
    assert sum(s["Direct+Clues"] for s in wickens["strata"]) == 2
    assert sum(s["Guided"] for s in wickens["strata"]) == 6

    chen = r_dev["PMC13321810"]
    assert chen["date"] >= "2026-04"
    assert sum(s["Direct_strict"] for s in chen["strata"]) == 0
    assert sum(s["Direct+Clues_strict"] for s in chen["strata"]) == 1
    assert sum(s["Guided_strict"] for s in chen["strata"]) == 2

    hu = r_dev["PMC13171995"]
    assert hu["date"] < "2026-04"
    assert sum(s["Direct_strict"] for s in hu["strata"]) == 3
    assert sum(s["Auto_strict"] for s in hu["strata"]) == 5
    assert sum(s["Direct"] for s in hu["strata"]) == 4
    assert sum(s["Auto"] for s in hu["strata"]) == 5

    raw_msg = ""
    if recompute:
        require_raw_outputs()
        for rel in (
            "lit172g/A1_place/PMC13449957.json",
            "lit172g/A1_place/PMC13321810.json",
            "lit172g/V5_rev1/PMC13171995.json",
        ):
            assert (RAW_OUTPUTS_V4 / rel).exists(), f"Missing raw case output: {rel}"
        raw_msg = "; raw case JSONs verified"

    print(
        f"[S11 OK] Per-paper tables verified (165 dev + 271 test); "
        f"PMC13449957 Guided=6/6, PMC13321810 Guided_strict=2/2, PMC13171995 Auto=5/5" + raw_msg
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    verify(recompute=ap.parse_args().recompute)
