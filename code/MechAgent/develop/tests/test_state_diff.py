from __future__ import annotations

import unittest

from develop import state_diff


def signature(atoms, bonds):
    return {
        "atoms": {
            name: {"element": element, "Hcount": hcount, "charge": charge}
            for name, element, hcount, charge in atoms
        },
        "bonds": {key: {"kind": kind, "order": order} for key, kind, order in bonds},
    }


class DiffTests(unittest.TestCase):
    def test_reports_bond_charge_and_hcount_changes(self):
        current = signature(
            [("C1", "C", 1, 0), ("O2", "O", 0, -1), ("Br3", "Br", 0, 0)],
            [("C1|Br3", "single", 1.0)],
        )
        target = signature(
            [("C1", "C", 1, 0), ("O2", "O", 0, 0), ("Br3", "Br", 0, -1)],
            [("C1|O2", "single", 1.0)],
        )
        diff = state_diff.diff_signatures(current, target)
        self.assertEqual([entry["atoms"] for entry in diff.bonds_to_form], [["C1", "O2"]])
        self.assertEqual([entry["atoms"] for entry in diff.bonds_to_break], [["C1", "Br3"]])
        self.assertEqual({entry["atom"] for entry in diff.charge_changes}, {"O2", "Br3"})
        self.assertEqual(diff.hcount_changes, [])

    def test_bond_order_change_is_not_a_form_or_break(self):
        current = signature([("C1", "C", 0, 0), ("O2", "O", 0, 0)], [("C1|O2", "double", 2.0)])
        target = signature([("C1", "C", 0, 0), ("O2", "O", 0, 0)], [("C1|O2", "single", 1.0)])
        diff = state_diff.diff_signatures(current, target)
        self.assertEqual(diff.bonds_to_form, [])
        self.assertEqual(diff.bonds_to_break, [])
        self.assertEqual(len(diff.bond_order_changes), 1)

    def test_atoms_absent_from_the_target_are_not_counted_as_changes(self):
        """The annotated target legitimately omits byproducts such as H2."""

        current = signature(
            [("C1", "C", 0, 0), ("O2", "O", 0, 0), ("H9", "H", 0, 0), ("H10", "H", 0, 0)],
            [("C1|O2", "single", 1.0), ("H9|H10", "single", 1.0)],
        )
        target = signature([("C1", "C", 0, 0), ("O2", "O", 0, 0)], [("C1|O2", "single", 1.0)])
        diff = state_diff.diff_signatures(current, target)
        self.assertTrue(diff.is_empty())
        self.assertEqual(diff.atoms_only_in_current, ["H10", "H9"])


class ResonanceTests(unittest.TestCase):
    @staticmethod
    def _ring(orders):
        atoms = [(f"C{i}", "C", 0, 0) for i in range(6)]
        bonds = [
            (f"C{i}|C{(i + 1) % 6}", "double" if order == 2.0 else "single", order)
            for i, order in enumerate(orders)
        ]
        return signature(atoms, bonds)

    def test_kekule_alternation_is_not_an_outstanding_event(self):
        current = self._ring([2.0, 1.0, 2.0, 1.0, 2.0, 1.0])
        target = self._ring([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
        diff = state_diff.diff_signatures(current, target)
        self.assertEqual(diff.bond_order_changes, [])
        self.assertEqual(len(diff.resonance_bond_order_changes), 6)
        self.assertTrue(diff.is_empty())
        allowed, _ = state_diff.boundary_completion_allowed(diff)
        self.assertTrue(allowed)

    def test_a_real_bond_order_change_survives_the_split(self):
        """Carbonyl reduced to alcohol: C and O both lose total bond order."""

        current = signature([("C1", "C", 0, 0), ("O2", "O", 0, 0)], [("C1|O2", "double", 2.0)])
        target = signature([("C1", "C", 0, 0), ("O2", "O", 0, 0)], [("C1|O2", "single", 1.0)])
        diff = state_diff.diff_signatures(current, target)
        self.assertEqual(len(diff.bond_order_changes), 1)
        self.assertEqual(diff.resonance_bond_order_changes, [])
        self.assertFalse(diff.is_empty())

    def test_resonance_and_real_changes_are_separated_per_component(self):
        atoms = [(f"C{i}", "C", 0, 0) for i in range(6)] + [("C9", "C", 0, 0), ("O9", "O", 0, 0)]
        current = signature(
            atoms,
            [(f"C{i}|C{(i + 1) % 6}", "x", 2.0 if i % 2 == 0 else 1.0) for i in range(6)]
            + [("C9|O9", "double", 2.0)],
        )
        target = signature(
            atoms,
            [(f"C{i}|C{(i + 1) % 6}", "x", 1.0 if i % 2 == 0 else 2.0) for i in range(6)]
            + [("C9|O9", "single", 1.0)],
        )
        diff = state_diff.diff_signatures(current, target)
        self.assertEqual(len(diff.resonance_bond_order_changes), 6)
        self.assertEqual([entry["atoms"] for entry in diff.bond_order_changes], [["C9", "O9"]])


class BoundaryGateTests(unittest.TestCase):
    def test_omitted_byproduct_may_complete(self):
        current = signature(
            [("C1", "C", 0, 0), ("H9", "H", 0, 0), ("H10", "H", 0, 0)],
            [("H9|H10", "single", 1.0)],
        )
        target = signature([("C1", "C", 0, 0)], [])
        allowed, reason = state_diff.boundary_completion_allowed(
            state_diff.diff_signatures(current, target)
        )
        self.assertTrue(allowed, reason)

    def test_unformed_bond_on_the_product_may_not_complete(self):
        current = signature([("C1", "C", 0, 0), ("O2", "O", 0, 0)], [])
        target = signature([("C1", "C", 0, 0), ("O2", "O", 0, 0)], [("C1|O2", "single", 1.0)])
        allowed, reason = state_diff.boundary_completion_allowed(
            state_diff.diff_signatures(current, target)
        )
        self.assertFalse(allowed)
        self.assertIn("missing reaction event", reason)


class ProposalRelevanceTests(unittest.TestCase):
    def _diff(self):
        current = signature([("C1", "C", 0, 0), ("O2", "O", 0, 0), ("N3", "N", 0, 0)], [])
        target = signature(
            [("C1", "C", 0, 0), ("O2", "O", 0, 0), ("N3", "N", 0, 0)],
            [("C1|O2", "single", 1.0)],
        )
        return state_diff.diff_signatures(current, target)

    def test_action_touching_a_required_bond_is_reported(self):
        action = {
            "electron_transfers": [
                {"source": ("O2", "lone_pair"), "target": ("O2", "C1", "new_sigma_bond")}
            ]
        }
        evidence = state_diff.proposal_relevance(action, self._diff())
        self.assertTrue(evidence["addresses_any_required_change"])
        self.assertEqual(evidence["required_changes_addressed"], ["form C1-O2"])

    def test_action_touching_nothing_required_is_flagged_but_not_condemned(self):
        action = {
            "electron_transfers": [
                {"source": ("N3", "lone_pair"), "target": ("N3", "lone_pair")}
            ]
        }
        evidence = state_diff.proposal_relevance(action, self._diff())
        self.assertFalse(evidence["addresses_any_required_change"])
        self.assertIn("not by itself an error", evidence["note"])


class RenderTests(unittest.TestCase):
    def test_empty_diff_renders_a_single_line(self):
        state = signature([("C1", "C", 0, 0)], [])
        rendered = state_diff.render_diff(state_diff.diff_signatures(state, state))
        self.assertIn("none;", rendered)

    def test_rendered_diff_names_every_required_change(self):
        current = signature([("C1", "C", 0, 0), ("O2", "O", 0, -1)], [])
        target = signature(
            [("C1", "C", 0, 0), ("O2", "O", 0, 0)], [("C1|O2", "single", 1.0)]
        )
        rendered = state_diff.render_diff(state_diff.diff_signatures(current, target))
        self.assertIn("C1-O2", rendered)
        self.assertIn("O2 -1 -> +0", rendered)
        self.assertIn("2 change(s) on shared atoms", rendered)

class HydrogenRelabelingTests(unittest.TestCase):
    def test_hcl_formed_with_a_different_proton_is_not_a_missing_bond(self):
        """Observed in double_SN2::701: the gate refused a completed mechanism."""

        current = signature(
            [("Cl102", "Cl", 0, 0), ("H900000", "H", 0, 0), ("H900001", "H", 0, 0)],
            [("Cl102|H900001", "single", 1.0)],
        )
        target = signature(
            [("Cl102", "Cl", 0, 0), ("H900000", "H", 0, 0)],
            [("Cl102|H900000", "single", 1.0)],
        )
        diff = state_diff.diff_signatures(current, target)
        self.assertEqual(diff.bonds_to_form, [])
        self.assertEqual(len(diff.hydrogen_relabelings), 1)
        self.assertTrue(diff.is_empty())
        allowed, _ = state_diff.boundary_completion_allowed(diff)
        self.assertTrue(allowed)

    def test_a_genuinely_absent_proton_still_counts(self):
        """The heavy atom carries fewer hydrogens than the target requires."""

        current = signature(
            [("O1", "O", 0, -1), ("H900000", "H", 0, 0), ("C2", "C", 0, 0)],
            [("C2|H900000", "single", 1.0)],
        )
        target = signature(
            [("O1", "O", 0, 0), ("H900000", "H", 0, 0), ("C2", "C", 0, 0)],
            [("O1|H900000", "single", 1.0)],
        )
        diff = state_diff.diff_signatures(current, target)
        self.assertEqual([e["atoms"] for e in diff.bonds_to_form], [["O1", "H900000"]])
        self.assertEqual([e["atoms"] for e in diff.bonds_to_break], [["C2", "H900000"]])
        self.assertEqual(diff.hydrogen_relabelings, [])
        self.assertFalse(diff.is_empty())

if __name__ == "__main__":
    unittest.main()
