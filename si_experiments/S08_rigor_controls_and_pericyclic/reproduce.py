"""Verify and optionally recompute SI Section S8 (Tables S15-S16, Fig. S4)."""
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
    p23 = load_json(CANONICAL_ROOT / "pilots_v4_summaries/SUMMARY_P23_OFFLINE.json")
    audit = load_json(CANONICAL_ROOT / "v3_analyses_20261107/auto_inserted_audit_and_examples.json")
    gap = load_json(CANONICAL_ROOT / "pilots_v4_summaries/SUMMARY_DIRECT_GAP.json")

    assert audit["audit"] == {"exact": 80, "equivalent": 8, "not_in_reference": 126}
    assert sum(audit["audit"].values()) == 214

    c_rule = p23["C_gap_rule_on_reference_steps"]
    assert c_rule["reference_steps"] == 1483
    assert c_rule["reference_steps_flagged_ge3"] == [116, 7.8]

    assert round(100 * gap["pooled"]["all"]["Direct_gap_only"]["all_present"][0], 1) == 35.7
    assert round(100 * gap["pooled"]["all"]["MechSight_Auto"]["all_present"][0], 1) == 43.0

    if recompute:
        require_raw_outputs()
        run_pilots_script("reproduction/MechAgent/experiments/pilots_v4/p23_offline.py")
        out_audit = RAW_OUTPUTS_V4 / "v3_analyses_20261107/auto_inserted_audit_and_examples.json"
        run_pilots_script("reproduction/MechAgent/experiments/pilots_v4/p42_auto_inserted_audit.py", [str(out_audit)])
        run_pilots_script("reproduction/MechAgent/experiments/pilots_v4/p39_direct_gap.py", ["--phase", "analyze"])
        re_audit = json.loads(out_audit.read_text())
        assert re_audit["audit"] == {"exact": 80, "equivalent": 8, "not_in_reference": 126}

    print(
        f"[S08 OK] 214 inserted species audit ({audit['audit']}); "
        f"1,483 ref steps: {1483-116} (92.2%) <= 2 vs 116 (7.8%) >= 3."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    verify(recompute=ap.parse_args().recompute)
