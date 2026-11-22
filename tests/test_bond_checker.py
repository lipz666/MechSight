"""Tests for Eq. S8 heavy-atom bond-change counter and BondChangeChecker."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from mechsight.bond_checker import BondChangeChecker, count_heavy_bond_changes
from mechsight.chemistry import build_step_reactions
from mechsight.schema import ReactionCase


class BondCheckerTests(unittest.TestCase):
    def test_count_heavy_bond_changes_eq_s8(self) -> None:
        # Proton transfer: Delta_bond = 0
        pt = "[CH3:1][C:2](=[O:3])[CH3:4]>>[CH2-:1][C:2](=[O:3])[CH3:4]"
        self.assertEqual(count_heavy_bond_changes(pt), 0)

        # Single C-Br cleavage with leaving Br: |B_leave| = 1 -> Delta_bond = 1
        alpha_elim = "[C:1]([C:2])[Br:4]>>[C:1]=[C:2]"
        self.assertEqual(count_heavy_bond_changes(alpha_elim), 1)

        # Concerted [4+2] Diels-Alder: 2 new sigma bonds formed -> Delta_bond = 2 (NOT flagged at >= 3)
        diels_alder = "[CH2:1]=[CH:2][CH:3]=[CH2:4].[CH2:5]=[CH2:6]>>[CH2:1]1[CH:2]=[CH:3][CH2:4][CH2:6][CH2:5]1"
        self.assertEqual(count_heavy_bond_changes(diels_alder), 2)

        # Concerted [3,3]-sigmatropic Cope rearrangement: 1 broken + 1 formed = 2 -> Delta_bond = 2
        cope = "[CH2:1]=[CH:2][CH2:3][CH2:4][CH:5]=[CH2:6]>>[CH2:3]=[CH:2][CH2:1][CH2:6][CH:5]=[CH2:4]"
        self.assertEqual(count_heavy_bond_changes(cope), 2)

        # Fritsch-Buttenberg-Wiechell 3-bond shortcut: C-Br leaving (1) + C1-Ph broken (1) + C2-Ph formed (1) = 3
        fbw_jump = (
            "[c:1]1[c:2][c:3][c:4][c:5][c:6]1[C:7]([c:8]2[c:9][c:10][c:11][c:12][c:13]2)"
            "=[CH:14][Br:15]>>[c:1]1[c:2][c:3][c:4][c:5][c:6]1[C:7]#[C:14][c:8]2[c:9][c:10][c:11][c:12][c:13]2"
        )
        self.assertEqual(count_heavy_bond_changes(fbw_jump), 3)

    def test_bond_change_checker_cache_and_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "stepmap_cache.json"
            case = ReactionCase(
                case_id="NR-073",
                reactants=["c1ccccc1C(=CBr)c2ccccc2"],
                products=["c1ccccc1C#Cc2ccccc2"],
            )
            steps = [
                {"step": 1, "subtype": "acid_base", "intermediate_smiles": "c1ccccc1C(=[C-]Br)c2ccccc2"},
                {"step": 2, "subtype": "rearrangement", "intermediate_smiles": "c1ccccc1C#Cc2ccccc2"},
                {"step": 3, "subtype": "invalid", "intermediate_smiles": "not_valid_smiles"},
            ]
            rxn1, rxn2, _ = build_step_reactions(case.reactants, steps)
            checker = BondChangeChecker(
                cache_path=cache_file,
                precomputed_cache={rxn1: 0, rxn2: 3},
            )
            flags = checker.detect_gaps(case, steps)
            self.assertEqual([(f.step_index, f.delta_bond) for f in flags], [(1, 3), (2, None)])

            rendered, all_flags = checker.render_annotated_candidates(case, [steps])
            self.assertIn("[FLAGGED: several heavy-atom bonds change in this one step]", rendered)
            self.assertIn("[FLAGGED: species does not parse]", rendered)
            self.assertEqual(len(all_flags[0]), 2)

    def test_bond_change_checker_live_rxnmapper_subprocess(self) -> None:
        rxn_py = os.environ.get("MECHSIGHT_RXNMAPPER_PYTHON")
        if not rxn_py or not Path(rxn_py).exists():
            self.skipTest("MECHSIGHT_RXNMAPPER_PYTHON not configured")

        with tempfile.TemporaryDirectory() as tmpdir:
            checker = BondChangeChecker(
                cache_path=Path(tmpdir) / "live_cache.json",
                rxnmapper_python=rxn_py,
            )
            da_case = ReactionCase(
                case_id="da",
                reactants=["C=CC=C", "C=C"],
                products=["C1=CCCCC1"],
            )
            da_flags = checker.detect_gaps(da_case, [{"step": 1, "intermediate_smiles": "C1=CCCCC1"}])
            self.assertEqual(da_flags, [])

            fbw_case = ReactionCase(
                case_id="fbw",
                reactants=["c1ccccc1C(=CBr)c2ccccc2"],
                products=["c1ccccc1C#Cc2ccccc2"],
            )
            fbw_flags = checker.detect_gaps(fbw_case, [{"step": 1, "intermediate_smiles": "c1ccccc1C#Cc2ccccc2"}])
            self.assertEqual(len(fbw_flags), 1)
            self.assertIsNotNone(fbw_flags[0].delta_bond)
            self.assertGreaterEqual(fbw_flags[0].delta_bond, 3)


if __name__ == "__main__":
    unittest.main()
