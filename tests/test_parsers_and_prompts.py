"""Tests for response parsers and SI Section S12 prompt templates."""
from __future__ import annotations

import unittest

from mechsight.chemistry import deduplicate_candidate_pathways, parse_enumerated_paths, parse_mechanism_json
from mechsight.prompts import (
    AUTO_COMPLETE_TEMPLATE,
    AUTO_ENUMERATE_BLOCK,
    AUTO_REVISE_TEMPLATE,
    GUIDED_HYPOTHESIS_TEMPLATE,
    GUIDED_INSERT_TEMPLATE,
    GUIDED_REVISE_TEMPLATE,
    format_direct_clues_prompt,
    format_direct_prompt,
    format_enumerate_prompt,
)
from mechsight.schema import ReactionCase


class ParsersAndPromptsTests(unittest.TestCase):
    def test_parse_mechanism_json_variants(self) -> None:
        plain = '[{"step": 1, "type": "addition", "subtype": "nucleophilic_addition", "intermediate_smiles": "CCO"}]'
        self.assertEqual(len(parse_mechanism_json(plain) or []), 1)

        fenced = f"Reasoning here.\n[ANSWER]\n```json\n{plain}\n```\n[/ANSWER]"
        parsed = parse_mechanism_json(fenced)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed[0]["intermediate_smiles"], "CCO")

        self.assertIsNone(parse_mechanism_json("no json here"))
        self.assertIsNone(parse_mechanism_json("[ANSWER] [] [/ANSWER]"))

    def test_parse_enumerated_paths_and_dedup(self) -> None:
        raw = """
        [PATH1]
        [{"step": 1, "intermediate_smiles": "CC(=O)C"}]
        [/PATH1]
        [PATH2]
        [{"step": 1, "intermediate_smiles": "CC(=O)C"}]
        [/PATH2]
        [PATH3]
        [{"step": 1, "intermediate_smiles": "CC(O)=C"}]
        [/PATH3]
        """
        paths = parse_enumerated_paths(raw)
        self.assertEqual(len(paths), 3)
        dedup = deduplicate_candidate_pathways(paths)
        self.assertEqual(len(dedup), 2)

    def test_prompt_templates_match_si_s12(self) -> None:
        case = ReactionCase(
            case_id="demo",
            reactants=["CC=O"],
            products=["CC(O)CC=O"],
            conditions="Base",
            observations="Aldol addition product observed.",
        )
        direct = format_direct_prompt(case)
        self.assertIn('"reactants_smiles": ["CC=O"]', direct)
        self.assertIn('"products_smiles": ["CC(O)CC=O"]', direct)

        enum_p = format_enumerate_prompt(case)
        self.assertIn("[PATH1] and [/PATH1]", enum_p)
        self.assertNotIn("Return only valid JSON", enum_p)

        clue_p = format_direct_clues_prompt(case)
        self.assertIn("Experimental observation made while running this reaction", clue_p)
        self.assertIn("Aldol addition product observed.", clue_p)

        self.assertIn("C<candidate>.S<step>: SMILES", AUTO_COMPLETE_TEMPLATE)
        self.assertIn("discard those with unsound elementary steps", AUTO_REVISE_TEMPLATE)
        self.assertIn("Distinguish a trapped adduct from its precursor", GUIDED_HYPOTHESIS_TEMPLATE)
        self.assertIn("state the reason in one sentence before the answer", GUIDED_REVISE_TEMPLATE)
        self.assertIn("Do not delete or change the existing intermediates.", GUIDED_INSERT_TEMPLATE)
        self.assertIn("THREE mechanistically distinct candidate mechanisms", AUTO_ENUMERATE_BLOCK)


if __name__ == "__main__":
    unittest.main()
