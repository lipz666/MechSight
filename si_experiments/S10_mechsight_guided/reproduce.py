"""Verify and optionally recompute SI Section S10 (Main Table 2, Fig. 4a-b, Tables S17-S19)."""
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
    m_eq = load_json(CANONICAL_ROOT / "v4_analyses_20261108/bench_eq/merged_apr_oct.json")["apr_oct"]
    m_st = load_json(CANONICAL_ROOT / "v4_analyses_20261108/bench_strict/merged_apr_oct.json")["apr_oct"]
    g71_test = load_json(PAPER_SUMMARIES_ROOT / "pilots_v4_20261103/litnew_confirm_g71/SUMMARY_G71.json")["summary"]
    g71_dev = load_json(PAPER_SUMMARIES_ROOT / "pilots_v4_20261103/g71_dev_20261106/SUMMARY_G71.json")["summary"]
    nodft = load_json(CANONICAL_ROOT / "nodft_case_20261108/SUMMARY.json")

    assert m_eq["papers"] == 285 and m_eq["intermediates"] == 1571
    assert m_eq["Guided"]["recall"][0] == 59.6
    assert m_eq["Guided"]["complete"][0] == 29.8
    assert m_eq["Direct+Clues"]["recall"][0] == 50.2
    assert m_eq["Direct+Clues"]["complete"][0] == 20.4
    assert m_eq["Guided-Direct+Clues"]["recall"][0] == 9.5
    assert m_eq["Guided-Direct+Clues"]["complete"][0] == 9.5

    assert m_st["Guided"]["recall"][0] == 49.4
    assert m_st["Guided"]["complete"][0] == 21.8

    assert m_eq["strata"]["direct"]["n"] == 593
    assert m_eq["strata"]["inert"]["n"] == 104
    assert m_eq["strata"]["blind"]["n"] == 874
    assert m_eq["strata"]["direct"]["Guided"] == 94.9
    assert m_eq["strata"]["inert"]["Guided"] == 77.9
    assert m_eq["strata"]["blind"]["Guided"] == 18.4

    assert g71_test["n_papers"] == 271 and g71_dev["n_papers"] == 165
    assert nodft["cid"] == "PMC13321810"

    if recompute:
        require_raw_outputs()
        v4_out = RAW_OUTPUTS_V4 / "v4_analyses_20261108"
        eq_dir = v4_out / "bench_eq"
        st_dir = v4_out / "bench_strict"
        eq_dir.mkdir(parents=True, exist_ok=True)
        st_dir.mkdir(parents=True, exist_ok=True)
        pub_dates = CANONICAL_ROOT / "v4_analyses_20261108/pub_dates.json"
        for which in ("dev", "test"):
            out_eq = eq_dir / f"rows_{which}.json"
            run_pilots_script(
                CANONICAL_ROOT / "v4_analyses_20261108/lit_bench_rule.py",
                [which, str(pub_dates), str(out_eq)],
            )
            rows = json.loads(out_eq.read_text())
            for r in rows:
                for st in r["strata"]:
                    for arm in ("Direct", "Auto", "Direct+Clues", "MV7", "Guided"):
                        st[arm] = st[f"{arm}_strict"]
            (st_dir / f"rows_{which}.json").write_text(json.dumps(rows))

        run_pilots_script(CANONICAL_ROOT / "v4_analyses_20261108/merged_report.py", [str(eq_dir)])
        run_pilots_script(CANONICAL_ROOT / "v4_analyses_20261108/merged_report.py", [str(st_dir)])
        re_eq = json.loads((eq_dir / "merged_apr_oct.json").read_text())["apr_oct"]
        re_st = json.loads((st_dir / "merged_apr_oct.json").read_text())["apr_oct"]
        assert re_eq["Guided"]["recall"][0] == 59.6 and re_eq["Guided"]["complete"][0] == 29.8
        assert re_st["Guided"]["recall"][0] == 49.4 and re_st["Guided"]["complete"][0] == 21.8

        ks_dir = RAW_OUTPUTS_V4 / "v3_analyses_20261107"
        run_pilots_script(
            "reproduction/MechAgent/experiments/pilots_v4/p41_knowledge_strata.py",
            ["dev", str(ks_dir / "knowledge_strata_dev.json")],
        )
        run_pilots_script(
            "reproduction/MechAgent/experiments/pilots_v4/p41_knowledge_strata.py",
            ["test", str(ks_dir / "knowledge_strata_test.json")],
        )
        run_pilots_script("reproduction/MechAgent/experiments/pilots_v4/p31_guided71.py", ["--set", "dev", "--analyze-only"])
        run_pilots_script("reproduction/MechAgent/experiments/pilots_v4/p31_confirm_core.py", ["--analyze-only"])

    print(
        f"[S10 OK] Post-cutoff 285 studies: Guided={m_eq['Guided']['recall'][0]}% recall / "
        f"{m_eq['Guided']['complete'][0]}% complete (equiv); "
        f"Strata Guided recall: prior={m_eq['strata']['direct']['Guided']}%, "
        f"inert={m_eq['strata']['inert']['Guided']}%, blind={m_eq['strata']['blind']['Guided']}%."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    verify(recompute=ap.parse_args().recompute)
