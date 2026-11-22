from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Planner skill retrieval and trigger verification without planning."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="gemma4-31b")
    parser.add_argument("--skills-dir", default="develop-2/planner_skills")
    parser.add_argument(
        "--expectations",
        default="develop-2/planner_skills/trigger_expectations.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-planner-skills", type=int, default=2)
    parser.add_argument("--skill-catalog-batch-size", type=int, default=6)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--skillrl-root", default=None)
    parser.add_argument("--reaction-id", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser


def _load_expectations(path: str) -> Dict[str, List[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("expectations must be a JSON object")
    return {
        str(reaction_id): [str(name) for name in names]
        for reaction_id, names in data.items()
    }


def main() -> None:
    args = build_parser().parse_args()
    chemistry = importlib.import_module("develop-2.chemistry")
    client = importlib.import_module("develop-2.client")
    agents = importlib.import_module("develop-2.agents")
    skills = importlib.import_module("develop-2.planner_skills")

    bridge_kwargs: Dict[str, Any] = {}
    if args.skillrl_root:
        bridge_kwargs["skillrl_root"] = args.skillrl_root
    bridge = chemistry.ChemistryBridge(**bridge_kwargs)
    tasks = bridge.load_tasks(
        data_source="core",
        split="all",
        extracted_dir="data/chem/extracted_paths",
        core_dir="data/chem/sft2_mechfindermerge_alignschema/core",
        core_subset="mechfinder",
        max_tasks=60,
        reaction_id=args.reaction_id,
        mechfinder_num_classes=60,
        mechfinder_per_class=1,
    )
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least one")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    tasks = tasks[args.shard_index :: args.num_shards]
    expectations = _load_expectations(args.expectations)
    library = skills.PlannerSkillLibrary.from_directory(args.skills_dir)
    model = client.OpenAIChatModel(
        client.OpenAIChatConfig(
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            enable_thinking=args.enable_thinking,
            timeout=args.timeout,
            trace_path=args.trace,
        )
    )
    planner = agents.PlannerAgent(
        model,
        skill_library=library,
        max_planner_skills=args.max_planner_skills,
        skill_catalog_batch_size=args.skill_catalog_batch_size,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    records: List[Dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        reaction_id = str(task.mechanism_id)
        print(f"[trigger {index}/{len(tasks)}] {reaction_id}", flush=True)
        try:
            selection = planner._select_skills(
                bridge.task_context(task, input_mode="structures_only")
            )
            candidates = [
                item["name"] for item in selection.get("candidate_skills", [])
            ]
            selected = [
                item["name"] for item in selection.get("selected_skills", [])
            ]
            error = None
        except Exception as exc:
            selection = {}
            candidates = []
            selected = []
            error = repr(exc)
        expected = expectations.get(reaction_id, [])
        record = {
            "reaction_id": reaction_id,
            "expected_skills": expected,
            "candidate_skills": candidates,
            "selected_skills": selected,
            "candidate_recall": all(name in candidates for name in expected),
            "trigger_exact_match": sorted(selected) == sorted(expected),
            "selection": selection,
            "error": error,
        }
        records.append(record)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    positives = [record for record in records if record["expected_skills"]]
    negatives = [record for record in records if not record["expected_skills"]]
    summary = {
        "num_tasks": len(records),
        "positive_count": len(positives),
        "positive_candidate_recall": sum(
            bool(record["candidate_recall"]) for record in positives
        ),
        "positive_exact_trigger": sum(
            bool(record["trigger_exact_match"]) for record in positives
        ),
        "negative_count": len(negatives),
        "negative_no_trigger": sum(
            bool(record["trigger_exact_match"]) for record in negatives
        ),
        "error_count": sum(record["error"] is not None for record in records),
        "output": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
