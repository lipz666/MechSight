"""End-to-end offline tests of MechSightAuto and MechSightGuided with MockLLMClient."""
from __future__ import annotations

import json
import unittest
from dataclasses import dataclass, field

from mechsight.auto import MechSightAuto
from mechsight.bond_checker import BondChangeChecker
from mechsight.chemistry import build_step_reactions, canonicalize_smiles
from mechsight.guided import MechSightGuided
from mechsight.schema import ReactionCase


@dataclass
class MockLLMClient:
    """Deterministic scripted LLM responder recording all calls."""

    responses: dict[str, str]
    calls: list[dict[str, object]] = field(default_factory=list)

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float,
        tag: str = "",
        cache_key: str | None = None,
    ) -> str | None:
        self.calls.append({"tag": tag, "temperature": temperature, "cache_key": cache_key, "prompt": prompt})
        return self.responses.get(tag)


def _step(n: int, smi: str, subtype: str = "step") -> dict[str, object]:
    return {"step": n, "type": "reaction", "subtype": subtype, "intermediate_smiles": smi}


class PipelineMockTests(unittest.TestCase):
    def test_mechsight_auto_end_to_end_two_rounds(self) -> None:
        case = ReactionCase(
            case_id="NR-073",
            reactants=["c1ccccc1C(=CBr)c2ccccc2"],
            products=["c1ccccc1C#Cc2ccccc2"],
            conditions="LDA",
        )
        vinyl_anion = "c1ccccc1C(=[C-]Br)c2ccccc2"
        carbene = "c1ccccc1C(=[C])c2ccccc2"
        prod = "c1ccccc1C#Cc2ccccc2"

        shortcut_steps = [_step(1, vinyl_anion), _step(2, prod)]
        repaired_steps = [_step(1, vinyl_anion), _step(2, carbene), _step(3, prod)]
        shortcut_mech = json.dumps(shortcut_steps)
        repaired_mech = json.dumps(repaired_steps)

        rxn_r_va, rxn_va_prod = build_step_reactions(case.reactants, shortcut_steps)
        _, rxn_va_cb, rxn_cb_prod = build_step_reactions(case.reactants, repaired_steps)
        checker = BondChangeChecker(
            precomputed_cache={
                rxn_r_va: 0,
                rxn_va_prod: 3,
                rxn_va_cb: 1,
                rxn_cb_prod: 2,
            }
        )

        llm = MockLLMClient(
            responses={
                "base_default": shortcut_mech,
                "samp1": shortcut_mech,
                "samp2": shortcut_mech,
                "samp3": shortcut_mech,
                "samp4": repaired_mech,
                "enum3": f"[PATH1]{shortcut_mech}[/PATH1]\n[PATH2]{repaired_mech}[/PATH2]\n[PATH3]{shortcut_mech}[/PATH3]",
                "V5_comp1": f"<states>\nC1.S2: {carbene}\nC1.S2: invalid_smiles\n</states>",
                "V5_rev1": f"[ANSWER]{shortcut_mech}[/ANSWER]",
                "V5_comp2": f"<states>\nC1.S2: {carbene}\n</states>",
                "V5_rev2": f"[ANSWER]{repaired_mech}[/ANSWER]",
            }
        )

        res = MechSightAuto(llm=llm, bond_checker=checker).run(case)
        self.assertEqual(len(res.candidates), 2)
        self.assertEqual(res.round1_bridging_states, [("C1.S2", canonicalize_smiles(carbene))])
        self.assertTrue(res.round2_triggered)
        self.assertIsNotNone(res.final_pathway)
        self.assertEqual([s["intermediate_smiles"] for s in res.final_pathway], [vinyl_anion, carbene, prod])
        self.assertEqual(res.final_gaps, [])

        temps = {c["tag"]: c["temperature"] for c in llm.calls}
        self.assertEqual(temps["base_default"], 0.2)
        self.assertEqual(temps["samp1"], 0.7)
        self.assertEqual(temps["enum3"], 0.7)
        self.assertEqual(temps["V5_comp1"], 0.2)
        self.assertEqual(temps["V5_rev1"], 0.2)
        self.assertEqual(temps["V5_comp2"], 0.2)
        self.assertEqual(temps["V5_rev2"], 0.2)

    def test_mechsight_guided_consensus_and_conditional_insertion(self) -> None:
        case = ReactionCase(
            case_id="PMC13321810",
            reactants=["COC(=O)C(CC(=C(F)F)c1ccccc1)(C(=O)OC)c1cccs1"],
            products=["COC(=O)C1(C(=O)OC)CC2(c3ccccc3)C3C=CSC13C2(F)F"],
            conditions="Ir-F, 450 nm",
            observations="TEMPO and hexa-2,4-diene quench reaction; linear Stern-Volmer.",
        )
        diradical_1a = "COC(=O)C(C[C](c1ccccc1)[C](F)F)(C(=O)OC)c1cccs1"
        spiro_int = "COC(=O)C1(C(=O)OC)C[C](c2ccccc2)C(F)(F)C12SC=C[CH]2"
        prod = case.products[0]

        concerted_steps = [_step(1, prod)]
        spiro_steps = [_step(1, spiro_int), _step(2, prod)]
        full_steps = [_step(1, diradical_1a), _step(2, spiro_int), _step(3, prod)]
        concerted = json.dumps(concerted_steps)
        with_spiro_only = json.dumps(spiro_steps)
        full_guided = json.dumps(full_steps)

        (rxn_r_prod,) = build_step_reactions(case.reactants, concerted_steps)
        rxn_r_spiro, rxn_spiro_prod = build_step_reactions(case.reactants, spiro_steps)
        rxn_r_dirad, rxn_dirad_spiro, _ = build_step_reactions(case.reactants, full_steps)
        checker = BondChangeChecker(
            precomputed_cache={
                rxn_r_prod: 2,
                rxn_r_spiro: 2,
                rxn_spiro_prod: 1,
                rxn_r_dirad: 1,
                rxn_dirad_spiro: 1,
            }
        )

        llm = MockLLMClient(
            responses={
                "base_default": concerted,
                "samp1": concerted,
                "samp2": concerted,
                "samp3": concerted,
                "samp4": concerted,
                "enum3": f"[PATH1]{concerted}[/PATH1]",
                "V5_comp1": "<states></states>",
                "V5_rev1": f"[ANSWER]{concerted}[/ANSWER]",
                "G7L_hyp0": f"<states>\n{diradical_1a}\n{spiro_int}\n</states>",
                "G7L_hyp1": f"<states>\n{diradical_1a}\n{spiro_int}\n</states>",
                "G7L_hyp2": f"<states>\n{diradical_1a}\n</states>",
                "G7L_hyp3": f"<states>\n{diradical_1a}\n</states>",
                "O_llm": with_spiro_only,
                "G7L_clue1": with_spiro_only,
                "G7L_clue2": with_spiro_only,
                "G71_rev": f"[ANSWER]{with_spiro_only}[/ANSWER]",
                "A1_place": f"[ANSWER]{full_guided}[/ANSWER]",
            }
        )
        res = MechSightGuided(llm=llm, bond_checker=checker).run(case)
        c_diradical = canonicalize_smiles(diradical_1a)
        c_spiro = canonicalize_smiles(spiro_int)
        self.assertEqual(res.proposal_support[c_diradical], 4)
        self.assertEqual(res.proposal_support[c_spiro], 5)
        self.assertEqual(res.missing_anchors, [c_diradical])
        self.assertTrue(res.insertion_triggered)
        self.assertEqual([s["intermediate_smiles"] for s in res.final_pathway], [diradical_1a, spiro_int, prod])

        llm_no_insert = MockLLMClient(
            responses={**llm.responses, "G71_rev": f"[ANSWER]{full_guided}[/ANSWER]"}
        )
        res_b = MechSightGuided(llm=llm_no_insert, bond_checker=checker).run(case)
        self.assertEqual(res_b.missing_anchors, [])
        self.assertFalse(res_b.insertion_triggered)
        self.assertTrue(all(c["tag"] != "A1_place" for c in llm_no_insert.calls))


if __name__ == "__main__":
    unittest.main()
