"""Tests for SI Section S3.1 formal evaluation metrics and Section S1.3 bootstrap/permutation."""
from __future__ import annotations

import unittest

from mechsight.evaluation import (
    bootstrap_ci,
    evaluate_pathway,
    evaluate_single_step,
    paired_permutation_pvalue,
)


class EvaluationTests(unittest.TestCase):
    def test_evaluate_pathway_strict_vs_equiv_and_order(self) -> None:
        reactants = ["CC(C)=O", "O", "c1ccc(N)cc1"]
        products = ["CC(O)(O)C", "c1ccc(N)cc1"]
        refs = ["CC(O)(C)[OH2+]", "c1ccc([NH3+])cc1"]

        # Prediction uses neutral hydrate (product) & neutral aniline (endpoint)
        pred = [
            {"step": 1, "intermediate_smiles": "CC(O)(O)C"},
            {"step": 2, "intermediate_smiles": "c1ccc(N)cc1"},
        ]
        res = evaluate_pathway(
            reactants=reactants,
            products=products,
            reference_intermediates=refs,
            predicted_steps=pred,
        )
        self.assertEqual(res["strict"]["complete_pathway"], 0.0)
        self.assertEqual(res["strict"]["product_reached"], 1.0)
        self.assertEqual(res["equiv"]["complete_pathway"], 1.0)
        self.assertEqual(res["equiv"]["intermediate_recall"], 1.0)
        self.assertEqual(res["equiv"]["in_order_pathway"], 1.0)

    def test_evaluate_pathway_precision_f1_and_padding_penalty(self) -> None:
        reactants = ["C=CC=C", "C=C"]
        products = ["C1=CCCCC1"]
        refs = ["C[CH]C=CC[CH2]"]
        pred_clean = [
            {"step": 1, "intermediate_smiles": "C[CH]C=CC[CH2]"},
            {"step": 2, "intermediate_smiles": "C1=CCCCC1"},
        ]
        pred_padded = [
            {"step": 1, "intermediate_smiles": "CCCC"},
            {"step": 2, "intermediate_smiles": "CCCCC"},
            {"step": 3, "intermediate_smiles": "C[CH]C=CC[CH2]"},
            {"step": 4, "intermediate_smiles": "C1=CCCCC1"},
        ]
        sc_clean = evaluate_pathway(
            reactants=reactants, products=products, reference_intermediates=refs, predicted_steps=pred_clean
        )
        sc_padded = evaluate_pathway(
            reactants=reactants, products=products, reference_intermediates=refs, predicted_steps=pred_padded
        )
        self.assertEqual(sc_clean["strict"]["precision"], 1.0)
        self.assertEqual(sc_clean["strict"]["f1"], 1.0)
        self.assertLess(sc_padded["strict"]["precision"], 0.5)
        self.assertLess(sc_padded["strict"]["f1"], sc_clean["strict"]["f1"])

    def test_evaluate_single_step_and_statistics(self) -> None:
        step_hit = evaluate_single_step(
            prev_state_smiles="CC(C)=O.O",
            ref_next_state_smiles="CC(O)(C)[OH2+]",
            pred_next_state_smiles="CC(O)(O)C",
        )
        self.assertFalse(step_hit["strict"])
        self.assertTrue(step_hit["equiv"])

        mean, lo, hi = bootstrap_ci(
            [1.0, 0.0, 1.0, 1.0, 0.5], clusters=["a", "a", "b", "c", "d"], n_resamples=500, seed=42
        )
        self.assertLessEqual(0.0, lo)
        self.assertLessEqual(lo, mean)
        self.assertLessEqual(mean, hi)
        self.assertLessEqual(hi, 1.0)

        p_sig = paired_permutation_pvalue([0.5, 0.4, 0.6, 0.3, 0.5, 0.7, 0.4, 0.6], n_permutations=2000, seed=0)
        self.assertLess(p_sig, 0.05)


if __name__ == "__main__":
    unittest.main()
