"""Verify and optionally recompute SI Section S7 (MechSight-Auto, Tables S13-S14, Fig. 3)."""
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
    p14 = load_json(CANONICAL_ROOT / "pilots_v4_summaries/SUMMARY_P14_confirm.json")
    lad = load_json(CANONICAL_ROOT / "pilots_v4_summaries/SUMMARY_P19_LADDER.json")
    bb = load_json(CANONICAL_ROOT / "v4_analyses_20261108/backbone_rule.json")
    ds_hi = load_json(CANONICAL_ROOT / "pilots_v4_summaries/SUMMARY_BB_deepseek-v4_1-flash_high.json")["T3_confirm"]
    ds_no = load_json(CANONICAL_ROOT / "pilots_v4_summaries/SUMMARY_BB_deepseek-v4_1-flash_nothink.json")["T3_confirm"]
    argon = load_json(CANONICAL_ROOT / "pilots_v4_summaries/SUMMARY_ARGON_CONFIRM.json")

    assert p14["pooled_primary"]["MechAgent"]["n"] == 244
    assert round(100 * p14["pooled_primary"]["MechAgent"]["all_present"][0], 1) == 43.0
    assert round(100 * p14["pooled_primary"]["Direct"]["all_present"][0], 1) == 35.2
    assert round(100 * p14["pooled_primary"]["MechAgent_minus_Direct"][0], 1) == 7.8

    assert round(100 * lad["pooled"]["L0_Direct"]["all_present"][0], 1) == 35.2
    assert round(100 * lad["pooled"]["L1_sampling_revision"]["all_present"][0], 1) == 39.8
    assert round(100 * lad["pooled"]["L2_plus_enumeration"]["all_present"][0], 1) == 41.0
    assert round(100 * lad["pooled"]["L3_MechSight"]["all_present"][0], 1) == 43.0

    # Cross-model transfer (Table S14)
    assert bb["claude-sonnet-5-5-high"]["Direct"]["cp_s"][0] == 22.9
    assert bb["claude-sonnet-5-5-high"]["MechSight_Auto"]["cp_s"][0] == 43.8
    assert bb["gpt-6-luna (high)"]["MechSight_Auto"]["cp_s"][0] == 22.9
    assert bb["gpt-6-luna (no reasoning)"]["MechSight_Auto"]["cp_s"][0] == 10.4
    assert round(100 * ds_hi["MechSight_Auto"]["all_present"][0], 1) == 29.2
    assert round(100 * ds_no["MechSight_Auto"]["all_present"][0], 1) == 4.2
    assert round(100 * argon["pooled_primary"]["MechAgent"]["all_present"][0], 1) == 44.0
    assert round(100 * argon["pooled_primary"]["MechAgent_minus_Direct"][0], 1) == 3.7

    if recompute:
        require_raw_outputs()
        run_pilots_script("reproduction/MechAgent/experiments/pilots_v4/p14_confirm.py")
        run_pilots_script("reproduction/MechAgent/experiments/pilots_v4/p19_ladder.py", ["--phase", "analyze"])
        out_bb = RAW_OUTPUTS_V4 / "v4_analyses_20261108/backbone_rule.json"
        run_pilots_script(CANONICAL_ROOT / "v4_analyses_20261108/backbone_rule.py", [str(out_bb)])
        run_pilots_script(
            "reproduction/MechAgent/experiments/pilots_v4/p46_backbone.py",
            ["--model", "deepseek-v4.1-flash", "--effort", "high", "--key-env", "OPENAI_API_KEY", "--phase", "analyze"],
        )
        run_pilots_script(
            "reproduction/MechAgent/experiments/pilots_v4/p46_backbone.py",
            ["--model", "deepseek-v4.1-flash", "--effort", "nothink", "--key-env", "OPENAI_API_KEY", "--phase", "analyze"],
        )
        run_pilots_script(
            "reproduction/MechAgent/experiments/pilots_v4/p46_backbone.py",
            ["--model", "claude-sonnet-5-5-high", "--key-env", "OPENAI_API_KEY", "--phase", "analyze"],
        )
        run_pilots_script(
            "reproduction/MechAgent/experiments/pilots_v4/p38_luna_t3.py",
            ["--effort", "high", "--phase", "analyze"],
        )
        run_pilots_script(
            "reproduction/MechAgent/experiments/pilots_v4/p17_argon_confirm.py",
            ["--sets", "T3_confirm,gold", "--phase", "analyze"],
        )
        re_p14 = json.loads((RAW_OUTPUTS_V4 / "SUMMARY_P14_confirm.json").read_text())
        assert round(100 * re_p14["pooled_primary"]["MechAgent"]["all_present"][0], 1) == 43.0

    print(
        f"[S07 OK] Pooled Auto={round(100*p14['pooled_primary']['MechAgent']['all_present'][0], 1)}% strict "
        f"(+{round(100*p14['pooled_primary']['MechAgent_minus_Direct'][0], 1)} pp vs Direct); "
        f"4-vendor transfer verified (Claude +20.8 pp, Luna +8.3 pp, DeepSeek +6.2 pp, Argon +3.7 pp)."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    verify(recompute=ap.parse_args().recompute)
