"""Command-line interface for running MechSight-Auto, MechSight-Guided, gap checks, and evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .auto import MechSightAuto
from .bond_checker import BondChangeChecker
from .chemistry import parse_mechanism_json
from .evaluation import evaluate_pathway
from .guided import MechSightGuided
from .llm import OpenAICompatibleClient
from .schema import ReactionCase


def _load_case(path: str) -> ReactionCase:
    return ReactionCase.from_dict(json.loads(Path(path).read_text()))


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mechsight", description="MechSight reaction mechanism elucidation CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("auto", "guided"):
        p = sub.add_parser(name, help=f"Run MechSight-{name.capitalize()} on a reaction JSON file")
        p.add_argument("--input", required=True, help="Path to reaction JSON file")
        p.add_argument("--output", help="Optional path to write JSON result")
        p.add_argument("--model", default=None, help="OpenAI-compatible model name (default: MECHSIGHT_MODEL)")
        p.add_argument("--cache-dir", default=None, help="Optional directory for caching LLM calls")
        p.add_argument("--rxnmapper-cache", default=None, help="Optional JSON cache for RXNMapper bond counts")
        p.add_argument("--rxnmapper-python", default=None, help="Python executable with rxnmapper installed")

    p_gap = sub.add_parser("check-gaps", help="Flag Delta_bond >= 3 steps in a predicted mechanism JSON")
    p_gap.add_argument("--input", required=True, help="Path to reaction JSON file")
    p_gap.add_argument("--mechanism", required=True, help="Path to mechanism JSON (step list or answer file)")
    p_gap.add_argument("--rxnmapper-cache", default=None)
    p_gap.add_argument("--rxnmapper-python", default=None)

    p_ev = sub.add_parser("evaluate", help="Evaluate a predicted mechanism against reference intermediates (SI S3.1)")
    p_ev.add_argument("--input", required=True, help="Path to reaction JSON with reference_intermediates")
    p_ev.add_argument("--mechanism", required=True, help="Path to predicted mechanism JSON")
    p_ev.add_argument("--strict-mode", choices=["sigma", "fixed_h"], default="sigma")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd in ("auto", "guided"):
        case = _load_case(args.input)
        llm = OpenAICompatibleClient(model=args.model, cache_dir=args.cache_dir)
        checker = BondChangeChecker(cache_path=args.rxnmapper_cache, rxnmapper_python=args.rxnmapper_python)
        if args.cmd == "auto":
            res = MechSightAuto(llm=llm, bond_checker=checker).run(case).to_dict()
        else:
            res = MechSightGuided(llm=llm, bond_checker=checker).run(case).to_dict()
        payload = json.dumps(res, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(payload)
        else:
            print(payload)
        return 0

    if args.cmd == "check-gaps":
        case = _load_case(args.input)
        raw = Path(args.mechanism).read_text()
        steps = parse_mechanism_json(raw) or json.loads(raw)
        checker = BondChangeChecker(cache_path=args.rxnmapper_cache, rxnmapper_python=args.rxnmapper_python)
        flags = [f.to_dict() for f in checker.detect_gaps(case, steps)]
        print(json.dumps({"case_id": case.case_id, "gaps": flags}, indent=2))
        return 0

    if args.cmd == "evaluate":
        case = _load_case(args.input)
        raw = Path(args.mechanism).read_text()
        steps = parse_mechanism_json(raw) or json.loads(raw)
        metrics = evaluate_pathway(
            reactants=case.reactants,
            products=case.products,
            reference_intermediates=case.reference_intermediates,
            predicted_steps=steps,
            strict_mode=args.strict_mode,
        )
        print(json.dumps(metrics, indent=2))
        return 0

    return 1
