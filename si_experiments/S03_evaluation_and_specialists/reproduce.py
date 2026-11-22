"""Verify and optionally recompute SI Section S3 (Main Table 1, Fig. 1b, Tables S2-S4)."""
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
    t1 = load_json(CANONICAL_ROOT / "v4_analyses_20261108" / "table1_recall_spec4090.json")
    eq1 = load_json(CANONICAL_ROOT / "v3_analyses_20261107" / "equiv_table1_spec4090.json")
    spec = load_json(CANONICAL_ROOT / "pilots_v4_summaries" / "SUMMARY_P18_SPEC_spec_v6_4090.json")
    step = load_json(CANONICAL_ROOT / "stepwise_20261108" / "SUMMARY_STEPWISE.json")

    assert t1["pooled"]["Direct"]["n"] == 244
    assert t1["pooled"]["Direct"]["strict"][0] == 35.2
    assert t1["pooled"]["Direct"]["equiv"][0] == 41.0
    assert t1["pooled"]["MechAgent"]["strict"][0] == 43.0
    assert t1["pooled"]["MechAgent"]["equiv"][0] == 48.8
    assert t1["pooled"]["FlowER"]["equiv"][0] == 7.8
    assert t1["pooled"]["ChRIMP"]["equiv"][0] == 5.3

    assert round(100 * eq1["pooled"]["MechAgent"]["equiv"][0], 1) == 48.8
    assert spec["uspto31k"]["n_cases"] == 200
    assert step["fuku"]["LLM@1 strict"]["acc"] == 64.6
    assert step["fuku"]["flower@1 strict"]["acc"] == 8.8
    assert step["flowertest"]["flower@1 strict"]["acc"] == 89.2

    if recompute:
        require_raw_outputs()
        out_t1 = RAW_OUTPUTS_V4 / "v4_analyses_20261108" / "table1_recall_spec4090.json"
        out_eq = RAW_OUTPUTS_V4 / "v3_analyses_20261107" / "equiv_table1_spec4090.json"
        run_pilots_script(CANONICAL_ROOT / "v4_analyses_20261108" / "table1_recall_spec4090.py", [str(out_t1)])
        run_pilots_script("reproduction/MechAgent/experiments/pilots_v4/p40_equiv_rescore.py", [str(out_eq)])
        run_pilots_script(
            "reproduction/MechAgent/experiments/pilots_v4/p18_spec_compare.py",
            extra_env={"SPEC_DIR": "spec_v6_4090"},
        )
        run_pilots_script("reproduction/MechAgent/experiments/pilots_v4/p44_stepwise.py", ["score"])

        re_t1 = json.loads(out_t1.read_text())
        assert re_t1["pooled"]["MechAgent"]["equiv"][0] == 48.8
        assert re_t1["pooled"]["MechAgent"]["strict"][0] == 43.0

    print(
        f"[S03 OK] Pooled N={t1['pooled']['Direct']['n']}: "
        f"Direct={t1['pooled']['Direct']['strict'][0]}%/{t1['pooled']['Direct']['equiv'][0]}%, "
        f"Auto={t1['pooled']['MechAgent']['strict'][0]}%/{t1['pooled']['MechAgent']['equiv'][0]}% (strict/equiv); "
        f"Stepwise Fukuyama LLM={step['fuku']['LLM@1 strict']['acc']}% vs FlowER={step['fuku']['flower@1 strict']['acc']}%."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    verify(recompute=ap.parse_args().recompute)
