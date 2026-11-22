"""Runnable example for MechSight-Guided (consensus experimental-evidence anchoring).

Usage:
  # Dry run (no API key needed; inspects the Guided prompts and consensus scoring logic)
  python examples/run_mechsight_guided_example.py --dry-run

  # Live run against an OpenAI-compatible LLM endpoint
  export OPENAI_API_KEY="your-key"
  export MECHSIGHT_MODEL="gemini-3.8-flash"
  export MECHSIGHT_RXNMAPPER_PYTHON="/path/to/env-map/bin/python"
  python examples/run_mechsight_guided_example.py --case wickens_jacs_2026
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mechsight import (
    BondChangeChecker,
    MechSightGuided,
    OpenAICompatibleClient,
    ReactionCase,
    evaluate_pathway,
)
from mechsight.prompts import format_direct_clues_prompt, format_guided_hypothesis_prompt


def main() -> None:
    ap = argparse.ArgumentParser(description="Run MechSight-Guided on a 2026 wet-lab literature example")
    ap.add_argument(
        "--case",
        choices=["wickens_jacs_2026", "chen_chem_sci_2026"],
        default="wickens_jacs_2026",
        help="Sample case key from sample_reactions.json",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print prompts without calling an LLM API")
    ap.add_argument("--cache-dir", default=None, help="Optional LLM response cache directory")
    args = ap.parse_args()

    samples = json.loads((Path(__file__).resolve().parent / "sample_reactions.json").read_text())
    case = ReactionCase.from_dict(samples[args.case])

    if args.dry_run:
        print("=== MechSight-Guided Dry Run ===")
        print(f"Case: {case.case_id} ({samples[args.case]['title']})")
        print("\n--- Observation-Driven Hypothesis Prompt (4 calls, T=0.7) ---")
        print(format_guided_hypothesis_prompt(case))
        print("\n--- Direct + Clues Prompt (excerpt) ---")
        print(format_direct_clues_prompt(case)[-900:])
        return

    llm = OpenAICompatibleClient(cache_dir=args.cache_dir)
    checker = BondChangeChecker()
    runner = MechSightGuided(llm=llm, bond_checker=checker)
    result = runner.run(case)

    print("Consensus anchors (>=3/7):", result.consensus_anchors)
    print("Insertion triggered:", result.insertion_triggered)
    print(json.dumps(result.final_pathway, indent=2))
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
