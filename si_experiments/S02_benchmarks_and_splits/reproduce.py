"""Verify SI Section S2 benchmark counts, splits, and Table S1 template retrieval."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from si_experiments._runner import (  # noqa: E402
    CANONICAL_ROOT,
    RAW_STORYLINE,
    REPRO_ROOT,
    load_json,
    raw_outputs_ready,
    require_raw_outputs,
    run_pilots_script,
)


def verify(*, recompute: bool = False) -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    splits = json.loads((data_dir / "fukuyama_splits.json").read_text())
    assert len(splits["T3_dev"]) == 47
    assert len(splits["T3_confirm"]) == 48
    assert set(splits["T3_dev"]).isdisjoint(set(splits["T3_confirm"]))

    clean_rows = [json.loads(l) for l in (data_dir / "fukuyama_clean.jsonl").read_text().splitlines() if l]
    heldout_rows = [json.loads(l) for l in (data_dir / "fukuyama_heldout.jsonl").read_text().splitlines() if l]
    reg = json.loads((data_dir / "fukuyama_registration.json").read_text())
    p23 = json.loads((data_dir / "p23_cases.json").read_text())
    j3_ids = set(reg["sets"]["primary_T0_T1_T2"]) | set(p23)
    ome_gold = json.loads((REPRO_ROOT / "external" / "oMeBench" / "data" / "oMe_Gold.json").read_text())
    assert len(clean_rows) == 60
    assert len(heldout_rows) == 218
    assert len(j3_ids) == 147
    assert len(ome_gold) == 196

    p1 = load_json(CANONICAL_ROOT / "pilots_v4_summaries" / "SUMMARY_P1.json")
    assert p1["T3_dev_std"]["A0"]["n"] == 47
    assert p1["J3_std"]["A0"]["n"] == 146
    assert len(p1["T3_dev_std"]) > 0

    if recompute:
        require_raw_outputs()
        run_pilots_script("reproduction/MechAgent/experiments/pilots_v4/p1_retrieval.py", ["--analyze-only"])
        cov_file = RAW_STORYLINE / "01_Problem_Template_Degeneration/outputs/fukuyama_coverage.json"
        if cov_file.exists():
            cov = json.loads(cov_file.read_text())
            assert cov["summary"]["all"]["cases"] == 60
            assert cov["summary"]["all"]["single_template_whole_path"] == 0.0
            assert cov["summary"]["all"]["fully_decomposable_with_oracle_intermediates"] == 0.0

    print(
        f"[S02 OK] FukuyamaBench: test={len(splits['T3_confirm'])}, dev_subset={len(splits['T3_dev'])}, "
        f"diagnostic_J3={len(j3_ids)} (clean={len(clean_rows)}, heldout={len(heldout_rows)}); "
        f"oMe-Gold={len(ome_gold)}; P1 summary verified."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    verify(recompute=ap.parse_args().recompute)
