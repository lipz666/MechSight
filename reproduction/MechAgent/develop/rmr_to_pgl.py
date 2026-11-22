"""Convert the remaining ReactionMechanismReasoning sources into PGL tasks.

`data_prep/sources/` carries two mechanism sets beyond FukuyamaBench that are
worth having:

- `oMe_Combined_mapped.json` - 2689 cases, mean 4.2 steps, with an easy/medium/
  hard `level` and a gold/silver `source` grade.
- `350_HumanBenchmark_mapped.json` - 350 cases, mean 2.9 steps, each citing a
  textbook page or a paper, and the shortest of the three sets.

Both store a step as per-species `mapped_smiles` rather than FukuyamaBench's
single `mapped_rxn` string, so only the reader differs; stitching, state
accumulation and the per-step bond-change ground truth are shared with
:mod:`develop.fukuyama_to_pgl`.

Two sources in that directory are deliberately not converted.
`art_augmented_reactions.json` is 1083 rows enumerating substituents over 19
parent reactions, so it would inflate any score by asking the same question
many times, and it carries no atom maps. `manually_curated_train.csv` is
single-step SMIRKS with an orbital-pair label, not a multi-step pathway.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .fukuyama_to_pgl import convert_steps, write_core_named

SOURCES = {
    "ome": "oMe_Combined_mapped.json",
    "human": "350_HumanBenchmark_mapped.json",
}


def _rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and "data" in payload:
        return list(payload["data"])
    return list(payload)


def _side(entries: List[Dict[str, Any]]) -> str:
    """Join one side of a step, preferring the mapped form."""

    return ".".join(
        str(entry.get("mapped_smiles") or entry.get("smiles") or "")
        for entry in entries
        if entry.get("mapped_smiles") or entry.get("smiles")
    )


def case_steps(case: Dict[str, Any]) -> Tuple[List[Tuple[str, str]], List[float]]:
    steps: List[Tuple[str, str]] = []
    confidences: List[float] = []
    for step in case.get("mechanism") or []:
        left = _side(step.get("reactants", []))
        right = _side(step.get("products", []))
        if not left or not right:
            continue
        steps.append((left, right))
        confidence = step.get("mapping_confidence")
        if confidence is not None:
            confidences.append(float(confidence))
    return steps, confidences


def _extra(case: Dict[str, Any]) -> Dict[str, Any]:
    """Carry through the per-source metadata worth filtering on later."""

    extra: Dict[str, Any] = {}
    for key in ("name", "level", "source", "citation", "product_formula"):
        if case.get(key) is not None:
            extra[key] = case[key]
    types = [
        step.get("type")
        for step in case.get("mechanism") or []
        if step.get("type")
    ]
    if types:
        extra["step_types"] = types
    return extra


def convert(path: Path, prefix: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: List[Dict[str, Any]] = []
    failures: Dict[str, int] = defaultdict(int)
    for case in _rows(payload):
        steps, confidences = case_steps(case)
        if not steps:
            failures["no mechanism steps"] += 1
            continue
        case_id = str(case.get("id") or case.get("name") or len(records))
        try:
            record = convert_steps(
                steps,
                case_id=case_id,
                partition=prefix,
                confidences=confidences,
                extra=_extra(case),
            )
        except Exception as exc:
            failures[str(exc).split(" has ")[-1] or repr(exc)] += 1
            continue
        records.append(record)
    return records, dict(failures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources_dir", type=Path, help="RMR data_prep/sources")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--source",
        choices=sorted(SOURCES) + ["all"],
        default="all",
    )
    parser.add_argument("--runnable-only", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument(
        "--core-dir",
        type=Path,
        default=None,
        help="Also write a core named-reaction file combining the kept records.",
    )
    args = parser.parse_args()

    chosen = sorted(SOURCES) if args.source == "all" else [args.source]
    kept: List[Dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in chosen:
        path = args.sources_dir / SOURCES[name]
        if not path.exists():
            print(f"{name}: missing {path}")
            continue
        records, failures = convert(path, name)
        total = len(records)
        if args.runnable_only:
            records = [r for r in records if r["quality"]["initial_state_complete"]]
        if args.min_confidence is not None:
            records = [
                r
                for r in records
                if (r["quality"]["mapping_confidence_min"] or 0.0) >= args.min_confidence
            ]
        out = args.output_dir / f"{name}.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        steps = sum(r["num_steps"] for r in records)
        print(
            f"{name}: converted {total}, kept {len(records)} ({steps} steps) -> {out}"
        )
        if failures:
            print(f"  skipped: {failures}")
        if records and records[0].get("level"):
            print(f"  level: {dict(Counter(r.get('level') for r in records))}")
        if records and records[0].get("source"):
            print(f"  grade: {dict(Counter(r.get('source') for r in records))}")
        kept.extend(records)

    if args.core_dir and kept:
        print(f"core dataset -> {write_core_named(kept, args.core_dir)}  ({len(kept)} cases)")


if __name__ == "__main__":
    main()
