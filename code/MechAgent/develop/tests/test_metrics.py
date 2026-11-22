from __future__ import annotations

import unittest

from develop import metrics


def action(*transfers):
    return {
        "electron_transfers": [
            {"source": source, "target": target} for source, target in transfers
        ]
    }


class TransferKeyTests(unittest.TestCase):
    def test_bare_and_tuple_endpoints_are_the_same_event(self):
        """Gold writes source='O205'; the Executor must write a structural tuple."""

        gold = {"source": "O205", "target": "H302"}
        predicted = {
            "source": ("O205", "lone_pair"),
            "target": ("O205", "H302", "new_sigma_bond"),
        }
        self.assertEqual(
            metrics.transfer_key(gold, anonymize_h=False),
            metrics.transfer_key(predicted, anonymize_h=False),
        )

    def test_hydrogen_labels_collapse_only_when_anonymizing(self):
        gold = {"source": "O301", "target": "H900005"}
        predicted = {"source": "O301", "target": "H900003"}
        self.assertNotEqual(
            metrics.transfer_key(gold, anonymize_h=False),
            metrics.transfer_key(predicted, anonymize_h=False),
        )
        self.assertEqual(
            metrics.transfer_key(gold, anonymize_h=True),
            metrics.transfer_key(predicted, anonymize_h=True),
        )

    def test_element_labels_starting_with_h_are_not_hydrogen(self):
        self.assertFalse(metrics._is_hydrogen("Hg301"))
        self.assertTrue(metrics._is_hydrogen("H900003"))


class ComparePathTests(unittest.TestCase):
    def test_merged_steps_pass_l1_and_fail_l3(self):
        """Friedel-Crafts: same five arrows, annotation merges them into two steps."""

        gold = [
            action((("Cl101", "lone_pair"), ("Al301", "atom")), (("C1", "Cl101", "single"), ("Cl101", "atom"))),
            action(
                (("C16", "C17", "pi_bond"), ("C1", "atom")),
                (("Al301", "Cl101", "single"), ("H900000", "atom")),
                (("C16", "H900000", "single"), ("C17", "atom")),
            ),
        ]
        predicted = [
            action((("Cl101", "lone_pair"), ("Al301", "atom"))),
            action((("C1", "Cl101", "single"), ("Cl101", "atom"))),
            action((("C16", "C17", "pi_bond"), ("C1", "atom"))),
            action(
                (("Al301", "Cl101", "single"), ("H900000", "atom")),
                (("C16", "H900000", "single"), ("C17", "atom")),
            ),
        ]
        comparison = metrics.compare_path(predicted, gold)
        self.assertTrue(comparison.transfer_set_match)
        self.assertFalse(comparison.step_partition_match)
        self.assertEqual(comparison.boundary_diagnosis, "split_steps")
        self.assertEqual(comparison.missing_transfers, [])
        self.assertEqual(comparison.extra_transfers, [])

    def test_levels_are_nested(self):
        gold = [action((("A1", "lone_pair"), ("B1", "atom"))), action((("B1", "lone_pair"), ("C1", "atom")))]
        for predicted in (gold, list(reversed(gold)), gold[:1]):
            comparison = metrics.compare_path(predicted, gold)
            if comparison.step_partition_match:
                self.assertTrue(comparison.transfer_sequence_match)
            if comparison.transfer_sequence_match:
                self.assertTrue(comparison.transfer_set_match)

    def test_within_step_order_does_not_affect_any_level(self):
        """Transfers inside one ReactionTransformation are concerted."""

        first = action((("A1", "lone_pair"), ("B1", "atom")), (("C1", "D1", "pi_bond"), ("E1", "atom")))
        second = action((("C1", "D1", "pi_bond"), ("E1", "atom")), (("A1", "lone_pair"), ("B1", "atom")))
        comparison = metrics.compare_path([second], [first])
        self.assertTrue(comparison.transfer_set_match)
        self.assertTrue(comparison.transfer_sequence_match)
        self.assertTrue(comparison.step_partition_match)

    def test_missing_event_is_reported_as_missing_not_mismatch(self):
        gold = [
            action((("A1", "lone_pair"), ("B1", "atom"))),
            action((("B1", "lone_pair"), ("C1", "atom"))),
        ]
        predicted = gold[:1]
        comparison = metrics.compare_path(predicted, gold)
        self.assertEqual(comparison.boundary_diagnosis, "missing_events")
        self.assertEqual(len(comparison.missing_transfers), 1)

    def test_h_label_only_flag(self):
        gold = [action((("O301", "lone_pair"), ("H900005", "atom")))]
        predicted = [action((("O301", "lone_pair"), ("H900003", "atom")))]
        comparison = metrics.compare_path(predicted, gold, anonymize_h=True)
        self.assertTrue(comparison.transfer_set_match)
        self.assertTrue(comparison.h_label_only)


class AggregateTests(unittest.TestCase):
    def _comparisons(self):
        gold = [action((("A1", "lone_pair"), ("B1", "atom")))]
        good = metrics.compare_path(gold, gold, reaction_id="rxn::1")
        bad = metrics.compare_path([], gold, reaction_id="rxn::2")
        return [good, good, good, bad, bad, bad]

    def test_repeats_of_one_reaction_do_not_inflate_the_sample(self):
        report = metrics.aggregate(self._comparisons())
        self.assertEqual(report["num_paths"], 6)
        self.assertEqual(report["num_reactions"], 2)

    def test_clustered_ci_reflects_reaction_count(self):
        """Three repeats of one reaction are one observation, not three."""

        values = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        clusters = ["a", "a", "a", "b", "b", "b"]
        low, high = metrics.clustered_bootstrap_ci(values, clusters, iterations=500)
        self.assertLessEqual(low, 0.0)
        self.assertGreaterEqual(high, 1.0)


if __name__ == "__main__":
    unittest.main()
