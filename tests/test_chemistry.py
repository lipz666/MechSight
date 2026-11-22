"""Unit tests for deterministic RDKit keys, partner covering, and state validation."""
from __future__ import annotations

import unittest
from collections import Counter

from mechsight.chemistry import (
    build_step_reactions,
    canonicalize_smiles,
    equiv_connectivity_keys,
    filter_proposed_states,
    fixed_h_inchi_keys,
    select_minimal_partners,
    sigma_graph_keys,
)


class ChemistryTests(unittest.TestCase):
    def test_sigma_graph_keys_resonance_and_protonation(self) -> None:
        # Resonance forms / bond-order depictions share the same sigma-graph key
        pyridone = "O=c1[nH]cccc1"
        hydroxypyridine = "Oc1ncccc1"
        self.assertEqual(sigma_graph_keys(pyridone), sigma_graph_keys(hydroxypyridine))

        # Different protonation states have distinct strict sigma-graph keys
        anilinium = "c1ccc([NH3+])cc1"
        aniline = "c1ccc(N)cc1"
        self.assertNotEqual(sigma_graph_keys(anilinium), sigma_graph_keys(aniline))
        self.assertNotEqual(sigma_graph_keys("CC(O)(C)[OH2+]"), sigma_graph_keys("CC(O)(O)C"))

        # Chemically equivalent 14-char InChIKey unifies protonation states & mobile-H tautomers
        self.assertEqual(equiv_connectivity_keys(anilinium), equiv_connectivity_keys(aniline))
        self.assertEqual(equiv_connectivity_keys("CC(O)(C)[OH2+]"), equiv_connectivity_keys("CC(O)(O)C"))
        self.assertEqual(equiv_connectivity_keys(pyridone), equiv_connectivity_keys(hydroxypyridine))

    def test_fixed_h_inchi_keys(self) -> None:
        self.assertTrue(fixed_h_inchi_keys("CCO"))
        self.assertNotEqual(fixed_h_inchi_keys("CC(O)=C"), fixed_h_inchi_keys("CC(=O)C"))
        self.assertEqual(fixed_h_inchi_keys("not_a_smiles"), set())

    def test_select_minimal_partners_and_step_reactions(self) -> None:
        partners = ["CC=O", "O", "c1ccccc1"]
        deficit = Counter({"C": 2, "O": 1})
        chosen = select_minimal_partners(deficit, partners)
        self.assertEqual(chosen, ["CC=O"])

        steps = [
            {"step": 1, "intermediate_smiles": "[CH2-]C=O"},
            {"step": 2, "intermediate_smiles": "CC([O-])CC=O"},
        ]
        rxns = build_step_reactions(["CC=O"], steps)
        self.assertEqual(len(rxns), 2)
        self.assertEqual(rxns[0], "CC=O>>[CH2-]C=O")
        self.assertEqual(rxns[1], "[CH2-]C=O.CC=O>>CC([O-])CC=O")

    def test_filter_proposed_states_enforces_valence_endpoints_and_budget(self) -> None:
        reactants = ["c1ccccc1C(=C(Br))c2ccccc2", "[OH-]"]
        products = ["c1ccccc1C#Cc2ccccc2", "[Br-]", "O"]
        raw = """
        <states>
        C1.S1: c1ccccc1C(=[C])c2ccccc2
        C1.S2: c1ccccc1C#Cc2ccccc2
        C1.S3: invalid_smiles_xxx
        C1.S4: c1ccccc1[Fe]Br
        C1.S5: C1=CC=CC=C1C(=[C])C1=CC=CC=C1
        </states>
        """
        kept = filter_proposed_states(reactants, products, raw)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0][0], "C1.S1")
        self.assertEqual(kept[0][1], canonicalize_smiles("c1ccccc1C(=[C])c2ccccc2"))


if __name__ == "__main__":
    unittest.main()
