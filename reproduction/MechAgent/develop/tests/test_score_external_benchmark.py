from __future__ import annotations

import unittest

from develop import metrics
from develop import score_external_benchmark as seb


def step(form=(), brk=(), order=(), charge=(), hcount=()):
    return {"form": list(form), "break": list(brk), "order": list(order), "charge": list(charge), "hcount": list(hcount)}


class NormalizationTests(unittest.TestCase):
    def test_kekule_alternation_is_a_redraw(self):
        ring = [(f"C{i}", f"C{(i + 1) % 6}", 2.0 if i % 2 else 1.0, 1.0 if i % 2 else 2.0) for i in range(6)]
        self.assertEqual(len(seb._kekule_bonds(ring)), 6)
        self.assertEqual(seb.to_counter(step(order=ring), normalized=True), {})
        self.assertEqual(sum(seb.to_counter(step(order=ring), normalized=False).values()), 6)

    def test_a_real_bond_order_change_survives(self):
        carbonyl = [("C1", "O2", 2.0, 1.0)]
        self.assertEqual(seb._kekule_bonds(carbonyl), set())
        self.assertIn(("order", "C1", "O2"), seb.to_counter(step(order=carbonyl), normalized=True))

    def test_relay_with_a_changing_atom_is_kept(self):
        """Conjugate addition: the pi relay moves charge, so it is chemistry."""

        relay = [("C1", "C2", 2.0, 1.0), ("C2", "C3", 1.0, 2.0), ("C3", "O4", 2.0, 1.0)]
        self.assertEqual(seb._kekule_bonds(relay), set())

    def test_proton_charge_and_hcount_are_dropped_only_when_normalized(self):
        s = step(form=[("O1", "H900001")], charge=["H900001", "O1"], hcount=["C5"])
        normalized = seb.to_counter(s, normalized=True)
        raw = seb.to_counter(s, normalized=False)
        self.assertNotIn(("charge", "H*"), normalized)
        self.assertIn(("charge", "O1"), normalized)
        self.assertIn(("form", "H*", "O1"), normalized)
        self.assertIn(("charge", "H*"), raw)
        self.assertIn(("hcount", "C5"), raw)


class LayeredMetricTests(unittest.TestCase):
    def test_same_changes_grouped_differently_pass_l1_and_fail_l3(self):
        gold = [seb.to_counter(step(form=[("C1", "O2")], brk=[("C1", "Br3")]), normalized=True)]
        predicted = [
            seb.to_counter(step(brk=[("C1", "Br3")]), normalized=True),
            seb.to_counter(step(form=[("C1", "O2")]), normalized=True),
        ]
        comparison = metrics.compare_path(seb.as_actions(predicted), seb.as_actions(gold))
        self.assertTrue(comparison.transfer_set_match)
        self.assertFalse(comparison.step_partition_match)
        self.assertEqual(comparison.boundary_diagnosis, "split_steps")

    def test_bonds_only_ignores_charge(self):
        gold = [seb.to_counter(step(form=[("Li1", "O2")], charge=["Li1", "O2"]), normalized=True)]
        predicted = [seb.to_counter(step(form=[("Li1", "O2")]), normalized=True)]
        self.assertFalse(metrics.compare_path(seb.as_actions(predicted), seb.as_actions(gold)).transfer_set_match)
        self.assertTrue(
            metrics.compare_path(
                seb.as_actions(predicted, bonds_only=True), seb.as_actions(gold, bonds_only=True)
            ).transfer_set_match
        )

    def test_clean_gold_uses_kind_orders(self):
        clean = {
            "bonds_formed": [["C2", "C6", "single"]],
            "bonds_broken": [],
            "bond_order_changes": [["C2", "O1", "double", "single"]],
            "charge_changes": [["O1", 0, -1]],
        }
        converted = seb._step_from_clean(clean)
        self.assertEqual(converted["order"], [("C2", "O1", 2.0, 1.0)])
        self.assertEqual(seb.to_counter(converted, normalized=True)[("charge", "O1")], 1)


if __name__ == "__main__":
    unittest.main()
