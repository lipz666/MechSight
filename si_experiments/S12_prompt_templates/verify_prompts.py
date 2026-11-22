"""Verify exact verbatim alignment across S12_*.txt, mechsight/prompts.py, and frozen reproduction scripts."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mechsight import prompts as P  # noqa: E402


def _extract_const(rel_script: str, var_name: str) -> str:
    src = (REPO_ROOT / rel_script).read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    return ast.literal_eval(node.value)
    raise KeyError(f"{var_name} not found in {rel_script}")


def main() -> None:
    checks = [
        ("S12_1_auto_enumerate.txt", P.AUTO_ENUMERATE_BLOCK, "reproduction/MechAgent/experiments/pilots_v4/p2_enumerate.py", "ENUM"),
        ("S12_2_auto_complete.txt", P.AUTO_COMPLETE_TEMPLATE, "reproduction/MechAgent/experiments/pilots_v4/p11_mechagent.py", "COMPLETE"),
        ("S12_3_auto_revise.txt", P.AUTO_REVISE_TEMPLATE, "reproduction/MechAgent/experiments/pilots_v4/p11_mechagent.py", "REVISE"),
        ("S12_4_guided_hypothesis.txt", P.GUIDED_HYPOTHESIS_TEMPLATE, "reproduction/MechAgent/experiments/pilots_v4/p28_lit172_v7.py", "HYP"),
        ("S12_5_guided_revise.txt", P.GUIDED_REVISE_TEMPLATE, "reproduction/MechAgent/experiments/pilots_v4/p31_guided71.py", "REVISE71"),
        ("S12_6_guided_insert.txt", P.GUIDED_INSERT_TEMPLATE, "reproduction/MechAgent/experiments/pilots_v4/p34_pilot36.py", "PLACE"),
        ("S12_8_pdf_extraction.txt", P.PDF_EXTRACTION_PROMPT, "reproduction/MechAgent/experiments/pilots_v4/p29_gold_rebuild.py", "PROMPT"),
    ]
    for fname, pkg_val, script, var_name in checks:
        txt = (HERE / fname).read_text()
        frozen_val = _extract_const(script, var_name)
        assert txt == pkg_val, f"Mismatch between {fname} and mechsight.prompts"
        assert pkg_val == frozen_val, f"Mismatch between mechsight.prompts and {script}:{var_name}"

    s12_7 = (HERE / "S12_7_baselines_and_controls.txt").read_text()
    refine = _extract_const("reproduction/MechAgent/experiments/pilots_v4/p12_controls.py", "REFINE")
    obs = _extract_const("reproduction/MechAgent/experiments/pilots_v4/p8_observations.py", "OBS")
    oracle = _extract_const("reproduction/MechAgent/experiments/pilots_v4/p12_controls.py", "ORACLE")
    relabel = _extract_const("reproduction/MechAgent/experiments/pilots_v4/p15_relabel.py", "PROMPT")

    assert P.SELF_REFINE_TEMPLATE == refine and refine in s12_7
    assert P.DIRECT_CLUES_BLOCK == obs and obs in s12_7
    assert P.ORACLE_INTERMEDIATES_BLOCK == oracle and oracle in s12_7
    assert P.STEP_LABEL_TEMPLATE == relabel and relabel in s12_7

    print("[S12 OK] All 8 SI prompt template files match mechsight/prompts.py and frozen reproduction scripts verbatim.")


if __name__ == "__main__":
    main()
