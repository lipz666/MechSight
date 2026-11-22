"""Verify and optionally recompute SI Section S5 (Tables S10-S11, Fig. 2, Fig. S2)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from si_experiments._runner import (  # noqa: E402
    CANONICAL_ROOT,
    MECHAGENT_ROOT,
    PAPER_SUMMARIES_ROOT,
    RAW_OUTPUTS_V4,
    load_json,
    require_raw_outputs,
    run_pilots_script,
)


def verify(*, recompute: bool = False) -> None:
    p23 = load_json(CANONICAL_ROOT / "pilots_v4_summaries/SUMMARY_P23_OFFLINE.json")
    p4 = load_json(CANONICAL_ROOT / "pilots_v4_summaries/SUMMARY_P4.json")
    gold_or = load_json(CANONICAL_ROOT / "pilots_v4_summaries/SUMMARY_GOLD_ORACLE.json")
    inert = load_json(PAPER_SUMMARIES_ROOT / "inert_knowledge_20261102/SUMMARY.json")
    audit = load_json(PAPER_SUMMARIES_ROOT / "recall_probe_20261101/gap_audit/SUMMARY.json")

    b_track = p23["B_inert_vs_blind"]
    assert b_track["missed_by_direct"][0] == 251
    assert b_track["latent_in_own_pool"] == [129, 51.4]
    assert b_track["latent_recovered"] == [77, 30.7]
    assert b_track["new_from_completion"] == [2, 0.8]
    assert b_track["blind_spot"] == [120, 47.8]

    e_hid = p23["E_direct_by_hidden_count"]["all_expert_sets"]
    assert e_hid["0"] == [111, 72.1]
    assert e_hid["1-2"] == [202, 36.6]
    assert e_hid[">=3"] == [125, 16.0]

    assert p4["omitted_heavy_events_M1"] == 26
    assert p4["cases"] == 22
    assert round(100 * p4["union5_narrative"][0], 1) == 38.5
    assert round(100 * p4["union5_resample"][0], 1) == 19.2
    assert gold_or["O_hid"]["n"] == 196
    assert inert["all_events"]["events"] == 174
    assert inert["all_events"]["omitted_in_E"] == 76
    assert round(100 * inert["all_events"]["omitted_known_S_or_G"][0], 1) == 64.5
    assert audit

    if recompute:
        require_raw_outputs()
        run_pilots_script("reproduction/MechAgent/experiments/pilots_v4/p23_offline.py")
        if (MECHAGENT_ROOT / "outputs/smiles_main_20261103/J3/gemini/M1").exists():
            run_pilots_script("reproduction/MechAgent/experiments/pilots_v4/p4_inertness.py", ["--analyze-only"])
        else:
            for r_arm in ("p4_R1", "p4_R2", "p4_R3", "p4_R4", "p4_R5"):
                assert (RAW_OUTPUTS_V4 / "J3" / r_arm).exists(), f"Missing archived {r_arm}"
        run_pilots_script("reproduction/MechAgent/experiments/pilots_v4/p12_controls.py", ["--phase", "analyze"])

    print(
        f"[S05 OK] 251 omitted hidden species: {b_track['latent_in_own_pool'][0]} inert (51.4%), "
        f"{b_track['new_from_completion'][0]} completion, {b_track['blind_spot'][0]} blind spots (47.8%)."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    verify(recompute=ap.parse_args().recompute)
