from __future__ import annotations

import unittest
from types import SimpleNamespace

from develop.chemistry import ChemistryBridge


class ChemistryBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = ChemistryBridge()

    def _state(
        self,
        *,
        hcount: int = 0,
        charge: int = 0,
        bond_kind: str = "single",
        bond_order=None,
        explicit_h: bool = False,
    ):
        bridge = self.bridge
        atomdict = {
            "C1": bridge.Atom("C", Hcount=hcount, charge=charge),
            "O1": bridge.Atom("O"),
        }
        bonds = [bridge.Bond("C1", "O1", bond_kind, order=bond_order)]
        if explicit_h:
            atomdict["H1"] = bridge.Atom("H")
            bonds.append(bridge.Bond("O1", "H1", "single"))
        return bridge.ReactionState(
            species=[
                bridge.Species(
                    role="intermediate",
                    state="stable",
                    molecule=bridge.Molecule(
                        name="",
                        substructures=[
                            bridge.Substructure(
                                name="main",
                                atomdict=atomdict,
                                bondlist=bonds,
                            )
                        ],
                    ),
                )
            ]
        )

    def test_strict_match_checks_order_charge_hcount_and_explicit_h(self):
        baseline = self._state()
        self.assertTrue(
            self.bridge.strict_state_matches(
                baseline,
                self._state(bond_order=1.0),
            )
        )
        self.assertFalse(
            self.bridge.strict_state_matches(
                baseline,
                self._state(bond_kind="double"),
            )
        )
        self.assertFalse(self.bridge.strict_state_matches(baseline, self._state(charge=-1)))
        self.assertFalse(self.bridge.strict_state_matches(baseline, self._state(hcount=1)))
        self.assertFalse(self.bridge.strict_state_matches(baseline, self._state(explicit_h=True)))

    def test_pgl_omits_zero_hcount_and_keeps_nonzero_hcount(self):
        zero_text = self.bridge.state_to_pgl(self._state(hcount=0))
        nonzero_text = self.bridge.state_to_pgl(self._state(hcount=2))
        self.assertNotIn("Hcount=0", zero_text)
        self.assertIn("Atom('C', Hcount=2)", nonzero_text)

    def test_default_task_context_hides_reaction_metadata(self):
        initial = self._state()
        target = self._state(bond_kind="double")
        task = SimpleNamespace(
            mechanism_id="mechfinder::Swern_oxidation::123",
            mechanism_name="Swern oxidation example",
            reaction_class="Swern_oxidation",
            metadata={"conditions": "Swern_oxidation"},
            source_path="secret/path",
            split="val",
            path=SimpleNamespace(sequence=[initial, target]),
        )

        context = self.bridge.task_context(task)

        self.assertEqual(context.reaction_id, "structures_only::123")
        self.assertIsNone(context.conditions)
        self.assertNotIn("Swern", context.task_description)
        self.assertNotIn("oxidation", context.task_description.lower())
        self.assertEqual(context.reaction_metadata, {"input_mode": "structures_only"})


if __name__ == "__main__":
    unittest.main()
