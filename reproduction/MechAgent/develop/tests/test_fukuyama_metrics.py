from __future__ import annotations

import json
import unittest
from pathlib import Path

from develop import fukuyama_metrics as fm
from develop import standalone_chemistry as chem

CLEAN = Path(__file__).resolve().parents[2] / "outputs" / "fukuyama" / "clean.jsonl"


def state(atoms, bonds):
    """atoms: (name, element, Hcount, charge); bonds: (a, b, order)."""
    kinds = {1.0: "single", 1.5: "aromatic", 2.0: "double", 3.0: "triple"}
    atomdict = {n: chem.Atom(e, Hcount=h, charge=c) for n, e, h, c in atoms}
    bondlist = [chem.Bond(a, b, kinds[o], order=o) for a, b, o in bonds]
    return chem.ReactionState([chem.Species("reactant", "stable", chem.Molecule("", [chem.Substructure("", atomdict, bondlist)]))])


CARBANION = state([("C1", "C", 2, -1), ("C2", "C", 1, 0), ("O3", "O", 0, 0)], [("C1", "C2", 1.0), ("C2", "O3", 2.0)])
ENOLATE = state([("C1", "C", 2, 0), ("C2", "C", 1, 0), ("O3", "O", 0, -1)], [("C1", "C2", 2.0), ("C2", "O3", 1.0)])
ENOL = state([("C1", "C", 2, 0), ("C2", "C", 1, 0), ("O3", "O", 1, 0)], [("C1", "C2", 2.0), ("C2", "O3", 1.0)])
ALDEHYDE = state([("C1", "C", 3, 0), ("C2", "C", 1, 0), ("O3", "O", 0, 0)], [("C1", "C2", 1.0), ("C2", "O3", 2.0)])


def view(s):
    return fm.state_view(s)


class StateComparisonTests(unittest.TestCase):
    def test_resonance_forms_match_only_when_normalized(self):
        self.assertTrue(fm.same_state(view(ENOLATE), view(CARBANION)))
        self.assertFalse(fm.same_state(view(ENOLATE), view(CARBANION), strict=True))

    def test_tautomers_differ(self):
        """A proton on another atom is a real event, never resonance."""
        self.assertFalse(fm.same_state(view(ENOL), view(ALDEHYDE)))

    def test_net_charge_differs(self):
        self.assertFalse(fm.same_state(view(ENOL), view(ENOLATE)))

    def test_explicit_and_recorded_hydrogens_count_alike(self):
        explicit = state([("C1", "C", 2, 0), ("C2", "C", 1, 0), ("O3", "O", 0, 0), ("H9", "H", 0, 0)],
                         [("C1", "C2", 2.0), ("C2", "O3", 1.0), ("O3", "H9", 1.0)])
        self.assertTrue(fm.same_state(view(explicit), view(ENOL), strict=True))

    def test_molecules_only_in_the_prediction_are_ignored(self):
        with_reagent = state([("C1", "C", 2, 0), ("C2", "C", 1, 0), ("O3", "O", 1, 0), ("Cl8", "Cl", 0, -1)],
                             [("C1", "C2", 2.0), ("C2", "O3", 1.0)])
        self.assertTrue(fm.same_state(view(with_reagent), view(ENOL)))
        self.assertFalse(fm.same_state(view(ENOL), view(with_reagent)))

    def test_a_finer_path_hits_every_intermediate(self):
        gold = [ALDEHYDE, ENOL]
        finer = [ENOLATE, ENOL]
        self.assertEqual(fm.intermediates_in_order(finer, gold), (True, 1))
        self.assertEqual(fm.intermediates_in_order([ENOLATE], gold), (False, 0))


class SigmaTests(unittest.TestCase):
    A = {"form": [("C1", "O2")], "break": []}
    B = {"form": [], "break": [("C1", "N3")]}
    C = {"form": [("H900001", "O2")], "break": [("H900001", "N4")]}

    def test_finer_path_passes_reorder_fails(self):
        gold = [{"form": self.A["form"], "break": self.B["break"]}, self.C]
        self.assertTrue(fm.sigma_scores([self.A, self.B, self.C], gold)["sigma_L2"])
        reordered = fm.sigma_scores([self.C, self.A, self.B], gold)
        self.assertTrue(reordered["sigma_L1"])
        self.assertFalse(reordered["sigma_L2"])

    def test_hydrogen_labels_are_anonymized(self):
        renamed = {"form": [("H7", "O2")], "break": [("H7", "N4")]}
        self.assertTrue(fm.sigma_scores([renamed], [self.C])["sigma_L1"])


@unittest.skipUnless(CLEAN.exists(), "converted FukuyamaBench not present")
class AnnotationConsistencyTests(unittest.TestCase):
    def test_initial_contains_first_annotated_state(self):
        for line in CLEAN.read_text().splitlines():
            case = json.loads(line)
            initial = chem.parse_pgl_literal(case["initial_state_pgl"])
            first = chem.parse_pgl_literal(case["states_pgl"][0])
            self.assertTrue(fm.same_state(view(initial), view(first), strict=True), case["case_id"])


if __name__ == "__main__":
    unittest.main()
