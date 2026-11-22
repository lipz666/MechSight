"""Verify and optionally recompute SI Section S4 (Tables S5-S9, Fig. 1c-d, Fig. S1)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from si_experiments._runner import (  # noqa: E402
    CANONICAL_ROOT,
    PAPER_SUMMARIES_ROOT,
    RAW_OUTPUTS_V4,
    load_json,
    require_raw_outputs,
    run_pilots_script,
)


def verify(*, recompute: bool = False) -> None:
    m3 = load_json(PAPER_SUMMARIES_ROOT / "pilots_v4_20261103/m3_counterfactual_20261106/SUMMARY_M3.json")["summary"]
    exp_u = load_json(CANONICAL_ROOT / "v4_analyses_20261108/exposure_unified.json")
    m2 = load_json(PAPER_SUMMARIES_ROOT / "pilots_v4_20261103/m2_exposure_20261106/SUMMARY_M2.json")["summary"]
    m14 = load_json(PAPER_SUMMARIES_ROOT / "pilots_v4_20261103/m1_m4_20261106/SUMMARY_M1_M4.json")["summary"]
    p5 = load_json(CANONICAL_ROOT / "pilots_v4_summaries/SUMMARY_P5.json")
    p22 = load_json(CANONICAL_ROOT / "pilots_v4_summaries/SUMMARY_P22_RANDSMI.json")

    assert m3["analyzed"] == 84
    assert m3["kstar_use"]["preserve"][0] == 81.0
    assert m3["kstar_use"]["break"][0] == 9.5
    assert m3["PRIMARY_break_minus_preserve"][0] == -71.4

    assert exp_u["high"]["n"] == 76 and exp_u["2026"]["n"] == 285
    assert exp_u["high"]["equiv"][0] == 40.6 and exp_u["2026"]["equiv"][0] == 43.3
    assert sum(m2["n"].values()) == 409

    assert m14["M4"]["n"] == 250
    assert m14["M4"]["PRIMARY_contradictory_minus_consistent"][0] == 84.6

    assert p5["a_oMe_product_prediction"]["pairs"] == 133
    assert p22["J3_mechanism"]["canonical"]["n"] == 147

    if recompute:
        require_raw_outputs()
        run_pilots_script("reproduction/MechAgent/experiments/pilots_v4/p32_m3_counterfactual.py", ["--phase", "analyze"])
        run_pilots_script("reproduction/MechAgent/experiments/pilots_v4/p31_m2_exposure.py", ["--analyze-only"])
        run_pilots_script("reproduction/MechAgent/experiments/pilots_v4/p33_m1_m4.py", ["--analyze-only"])
        run_pilots_script(
            CANONICAL_ROOT / "v4_analyses_20261108/exposure_rule.py",
            [str(RAW_OUTPUTS_V4 / "v4_analyses_20261108")],
        )
        re_exp = json.loads((RAW_OUTPUTS_V4 / "v4_analyses_20261108/exposure_unified.json").read_text())
        assert re_exp["high"]["equiv"][0] == 40.6

    print(
        f"[S04 OK] M3 n={m3['analyzed']} (break-preserve={m3['PRIMARY_break_minus_preserve'][0]} pp); "
        f"M2 high={exp_u['high']['equiv'][0]}% vs 2026={exp_u['2026']['equiv'][0]}%; "
        f"M4 n={m14['M4']['n']} (+{m14['M4']['PRIMARY_contradictory_minus_consistent'][0]} pp)."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    verify(recompute=ap.parse_args().recompute)
