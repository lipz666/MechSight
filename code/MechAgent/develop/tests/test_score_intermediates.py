from __future__ import annotations

import unittest

from develop import score_intermediates as si


def sig(atoms, bonds):
    """atoms: (name, element, Hcount, charge); bonds: (a, b, order)."""

    return {
        "atoms": {name: {"element": element, "Hcount": hcount, "charge": charge} for name, element, hcount, charge in atoms},
        "bonds": {f"{a}|{b}": {"kind": "x", "order": order} for a, b, order in bonds},
    }


RING = [(f"C{i}", "C", 1, 0) for i in range(6)]


def ring(shift):
    return [(f"C{i}", f"C{(i + 1) % 6}", 2.0 if (i + shift) % 2 == 0 else 1.0) for i in range(6)]


class SameIntermediateTests(unittest.TestCase):
    def test_kekule_redraw_is_the_same_intermediate(self):
        self.assertTrue(si.same_intermediate(sig(RING, ring(0)), sig(RING, ring(1))))

    def test_interchanged_proton_labels_are_the_same_intermediate(self):
        atoms = [("O1", "O", 0, 0), ("H901", "H", 0, 0), ("H902", "H", 0, 0)]
        first = sig(atoms, [("O1", "H901", 1.0)])
        second = sig(atoms, [("O1", "H902", 1.0)])
        self.assertTrue(si.same_intermediate(first, second))

    def test_proton_on_a_different_atom_is_different(self):
        atoms = [("O1", "O", 0, 0), ("O2", "O", 0, 0), ("H901", "H", 0, 0)]
        self.assertFalse(si.same_intermediate(sig(atoms, [("O1", "H901", 1.0)]), sig(atoms, [("O2", "H901", 1.0)])))

    def test_charge_on_a_different_atom_is_different(self):
        """Carbanion versus enolate: the program does not decide resonance."""

        carbanion = sig([("C1", "C", 1, -1), ("C2", "C", 0, 0), ("O3", "O", 0, 0)], [("C1", "C2", 1.0), ("C2", "O3", 2.0)])
        enolate = sig([("C1", "C", 1, 0), ("C2", "C", 0, 0), ("O3", "O", 0, -1)], [("C1", "C2", 2.0), ("C2", "O3", 1.0)])
        self.assertFalse(si.same_intermediate(carbanion, enolate))

    def test_aromatic_bond_localization_alone_is_the_same_intermediate(self):
        """A038: the runtime keeps 1.5 bonds the annotation writes as singles."""

        atoms = [("C5", "C", 1, -1), ("C3", "C", 1, 0), ("C6", "C", 1, 0)]
        runtime = sig(atoms, [("C3", "C5", 1.5), ("C5", "C6", 1.5)])
        annotated = sig(atoms, [("C3", "C5", 1.0), ("C5", "C6", 1.0)])
        self.assertTrue(si.same_intermediate(runtime, annotated))
        neutral = sig([("C5", "C", 1, 0), ("C3", "C", 1, 0), ("C6", "C", 1, 0)], [("C3", "C5", 1.0), ("C5", "C6", 1.0)])
        self.assertFalse(si.same_intermediate(runtime, neutral))

    def test_species_outside_one_boundary_is_ignored(self):
        core = [("C1", "C", 3, 0), ("O2", "O", 1, 0)]
        with_spectator = sig(core + [("Cl9", "Cl", 0, -1)], [("C1", "O2", 1.0)])
        without = sig(core, [("C1", "O2", 1.0)])
        self.assertTrue(si.same_intermediate(with_spectator, without))


class SequenceTests(unittest.TestCase):
    def setUp(self):
        base = [("C1", "C", 0, 0), ("O2", "O", 0, 0), ("N3", "N", 0, 0)]
        self.a = sig(base, [("C1", "O2", 1.0)])
        self.b = sig(base, [("C1", "O2", 1.0), ("C1", "N3", 1.0)])
        self.c = sig(base, [("C1", "N3", 1.0)])
        self.extra = sig(base, [("C1", "N3", 2.0)])

    def test_identical_sequences_pass_every_level(self):
        result = si.compare_sequences([self.a, self.b, self.c], [self.a, self.b, self.c])
        self.assertTrue(result["strict_sequence"] and result["ordered_all"] and result["final_match"])

    def test_an_extra_intermediate_fails_strict_but_keeps_order(self):
        result = si.compare_sequences([self.a, self.extra, self.b, self.c], [self.a, self.b, self.c])
        self.assertFalse(result["strict_sequence"])
        self.assertTrue(result["ordered_all"])
        self.assertEqual(result["matched_in_order"], 3)

    def test_a_skipped_intermediate_fails_both(self):
        result = si.compare_sequences([self.a, self.c], [self.a, self.b, self.c])
        self.assertFalse(result["strict_sequence"])
        self.assertFalse(result["ordered_all"])
        self.assertTrue(result["final_match"])
        self.assertEqual(result["matched_in_order"], 2)

    def test_empty_prediction_matches_nothing(self):
        result = si.compare_sequences([], [self.a])
        self.assertFalse(result["ordered_all"] or result["final_match"])


class CleanDerivationTests(unittest.TestCase):
    def test_changes_are_applied_cumulatively(self):
        case = {
            "initial_state_pgl": "ReactionState(species=[Species(role='reactant', state='stable', molecule=Molecule(name='', substructures=[Substructure(name='main', atomdict={'C1': Atom('C'), 'O2': Atom('O', charge=-1), 'Br3': Atom('Br')}, bondlist=[Bond('C1', 'Br3', 'single')])]))])",
            "gt_step_changes": [
                {"bonds_formed": [["C1", "O2", "single"]], "bonds_broken": [], "bond_order_changes": [], "charge_changes": [["O2", -1, 0]]},
                {"bonds_formed": [], "bonds_broken": [["C1", "Br3", "single"]], "bond_order_changes": [], "charge_changes": [["Br3", 0, -1]]},
            ],
        }
        states = si.gold_states_from_clean(case)
        self.assertEqual(len(states), 2)
        self.assertEqual(set(states[0]["bonds"]), {"Br3|C1", "C1|O2"} if "Br3|C1" in states[0]["bonds"] else {"C1|Br3", "C1|O2"})
        self.assertNotIn("C1|Br3", states[1]["bonds"])
        self.assertNotIn("Br3|C1", states[1]["bonds"])
        self.assertEqual(states[1]["atoms"]["Br3"]["charge"], -1)


if __name__ == "__main__":
    unittest.main()
