"""Runnable example for MechSight-Auto (clue-free prior-scaffold elucidation).

Usage:
  # Dry run (no API key needed; inspects the generated prompts and RDKit/RXNMapper gap flagger)
  python examples/run_mechsight_auto_example.py --dry-run

  # Live run against an OpenAI-compatible LLM endpoint
  export OPENAI_API_KEY="your-key"
  export MECHSIGHT_MODEL="gemini-3.8-flash"
  export MECHSIGHT_RXNMAPPER_PYTHON="/path/to/env-map/bin/python"
  python examples/run_mechsight_auto_example.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mechsight import (
    BondChangeChecker,
    MechSightAuto,
    OpenAICompatibleClient,
    ReactionCase,
    evaluate_pathway,
)
from mechsight.prompts import format_direct_prompt, format_enumerate_prompt


def main() -> None:
    ap = argparse.ArgumentParser(description="Run MechSight-Auto on the Fritsch-Buttenberg-Wiechell example")
    ap.add_argument("--dry-run", action="store_true", help="Print prompts and offline gap-detection demo without API calls")
    ap.add_argument("--cache-dir", default=None, help="Optional LLM response cache directory")
    args = ap.parse_args()

    samples = json.loads((Path(__file__).resolve().parent / "sample_reactions.json").read_text())
    case = ReactionCase.from_dict(samples["fbw_nr073"])

    if args.dry_run:
        print("=== MechSight-Auto Dry Run ===")
        print(f"Case: {case.case_id} ({samples['fbw_nr073']['title']})")
        print("\n--- Direct Prompt (excerpt) ---")
        print(format_direct_prompt(case)[:600] + "\n...")
        print("\n--- Hidden-Intermediate Enumeration Block ---")
        print(format_enumerate_prompt(case)[-700:])

        # Demonstrate offline Delta_bond >= 3 gap detection with precomputed step counts
        from mechsight.chemistry import build_step_reactions

        shortcut_candidate = [
            {"step": 1, "type": "proton_transfer", "subtype": "acid_base", "intermediate_smiles": "Br[C-]=C(c1ccccc1)c1ccccc1"},
            {"step": 2, "type": "rearrangement", "subtype": "1,2-shift", "intermediate_smiles": "c1ccccc1C#Cc2ccccc2"},
        ]
        rxn1, rxn2 = build_step_reactions(case.reactants, shortcut_candidate)
        checker = BondChangeChecker(precomputed_cache={rxn1: 0, rxn2: 3})
        rendered, flags = checker.render_annotated_candidates(case, [shortcut_candidate])
        print("\n--- Annotated Candidate with Delta_bond >= 3 Flag ---")
        print(rendered)
        print("Detected gaps:", [f.to_dict() for f in flags[0]])
        return

    llm = OpenAICompatibleClient(cache_dir=args.cache_dir)
    checker = BondChangeChecker()
    runner = MechSightAuto(llm=llm, bond_checker=checker)
    result = runner.run(case)

    print(json.dumps(result.to_dict(), indent=2))
    if case.reference_intermediates and result.final_pathway:
        scores = evaluate_pathway(
            reactants=case.reactants,
            products=case.products,
            reference_intermediates=case.reference_intermediates,
            predicted_steps=result.final_pathway,
        )
        print("\nEvaluation (SI S3.1):", json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
