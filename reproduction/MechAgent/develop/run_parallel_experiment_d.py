from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .chemistry import ChemistryBridge


@dataclass(frozen=True)
class RolloutJob:
    task_index: int
    reaction_id: str
    repeat_index: int

    @property
    def stem(self) -> str:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.reaction_id).strip("_")
        return f"task{self.task_index:03d}_repeat{self.repeat_index}_{safe_id}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run experiment D as resumable parallel shards.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--core-dir", default="../core")
    parser.add_argument("--output-root", default="outputs/gemini37_experiment_d/full_parallel")
    parser.add_argument("--merged-output", default="outputs/gemini37_experiment_d/full.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8317/v1")
    parser.add_argument("--model", default="gemini-3.7-flash-high")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--api-retries", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-env-calls", type=int, default=12)
    parser.add_argument("--max-planner-review-rounds", type=int, default=3)
    parser.add_argument(
        "--core-subset",
        choices=["mechfinder", "named_train", "named_val", "all"],
        default="mechfinder",
        help="Which core file to draw tasks from.",
    )
    parser.add_argument(
        "--disable-gt-assessment",
        action="store_true",
        help=(
            "Skip the post-hoc LLM evaluator. Required for a dataset whose "
            "ground truth carries states but no electron-transfer arrows."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Passed through to the child, e.g. minimal to turn thinking off.",
    )
    parser.add_argument(
        "--task-input-mode",
        choices=["structures_only", "metadata", "forward"],
        default="structures_only",
    )
    parser.add_argument(
        "--disable-planner-skills",
        action="store_true",
        help="develop-2 only. Required with --task-input-mode forward.",
    )
    parser.add_argument(
        "--agent-module",
        default="develop.run_agent",
        help="Runnable module for one rollout, e.g. develop-2.run_agent.",
    )
    parser.add_argument(
        "--oracle-step-count",
        action="store_true",
        help=(
            "Ablation only. Give the Planner the annotated step count so the "
            "step-convention component of the score can be separated from chemistry."
        ),
    )
    return parser


def _read_one_record(path: Path) -> Dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"expected one analysis record in {path}, found {len(lines)}")
    record = json.loads(lines[0])
    if not isinstance(record, dict) or "reaction" not in record:
        raise ValueError(f"invalid analysis record in {path}")
    return record


def _has_complete_assessment(record: Dict[str, Any]) -> bool:
    assessment = (record.get("gt") or {}).get("evaluator_assessment")
    required = {
        "planner_global_mechanism_judgment",
        "planner_elementary_steps_judgment",
        "predicted_events_judgment",
    }
    return isinstance(assessment, dict) and required.issubset(assessment)


def _normalize_record(record: Dict[str, Any], job: RolloutJob, repeats: int) -> Dict[str, Any]:
    normalized = dict(record)
    reaction = dict(normalized["reaction"])
    reaction["task_index"] = job.task_index
    reaction["repeat_index"] = job.repeat_index
    reaction["repeats_per_task"] = repeats
    reaction["reaction_id"] = job.reaction_id
    normalized["reaction"] = reaction
    return normalized


def _job_paths(root: Path, job: RolloutJob) -> Dict[str, Path]:
    return {
        "analysis": root / "raw" / f"{job.stem}.jsonl",
        "result": root / "results" / f"{job.stem}.jsonl",
        "planner": root / "traces" / f"{job.stem}.planner.jsonl",
        "executor": root / "traces" / f"{job.stem}.executor.jsonl",
        "evaluator": root / "traces" / f"{job.stem}.evaluator.jsonl",
        "console": root / "logs" / f"{job.stem}.log",
    }


def _run_job(
    *,
    job: RolloutJob,
    args: argparse.Namespace,
    repo_root: Path,
    output_root: Path,
    env: Dict[str, str],
) -> Tuple[RolloutJob, str]:
    paths = _job_paths(output_root, job)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    if paths["result"].exists():
        record = _read_one_record(paths["result"])
        if (
            str(record["reaction"].get("reaction_id")) == job.reaction_id
            and int(record["reaction"].get("repeat_index")) == job.repeat_index
            and (args.disable_gt_assessment or _has_complete_assessment(record))
        ):
            return job, "skipped"

    for key in ("analysis", "planner", "executor", "evaluator"):
        if paths[key].exists():
            paths[key].unlink()

    command = [
        sys.executable,
        "-m",
        args.agent_module,
        "--skillrl-root",
        str(repo_root),
        "--data-source",
        "core",
        "--split",
        "all",
        "--core-dir",
        str(Path(args.core_dir).expanduser().resolve()),
        "--core-subset",
        args.core_subset,
        "--mechfinder-num-classes",
        str(args.num_classes),
        "--mechfinder-per-class",
        "1",
        "--reaction-id",
        job.reaction_id,
        "--num-runs",
        "1",
        "--repeats-per-task",
        "1",
        "--task-input-mode",
        args.task_input_mode,
        "--planner-base-url",
        args.base_url,
        "--executor-base-url",
        args.base_url,
        "--gt-evaluator-base-url",
        args.base_url,
        "--planner-model",
        args.model,
        "--executor-model",
        args.model,
        "--gt-evaluator-model",
        args.model,
        "--planner-temperature",
        "0.2",
        "--executor-temperature",
        "0.2",
        "--gt-evaluator-temperature",
        "0.0",
        "--planner-max-tokens",
        str(args.max_tokens),
        "--executor-max-tokens",
        str(args.max_tokens),
        "--gt-evaluator-max-tokens",
        str(args.max_tokens),
        "--timeout",
        str(args.timeout),
        "--api-retries",
        str(args.api_retries),
        "--max-steps",
        str(args.max_steps),
        "--max-env-calls",
        str(args.max_env_calls),
        "--max-planner-review-rounds",
        str(args.max_planner_review_rounds),
        "--omit-thinking-extra-body",
        "--analysis-output",
        str(paths["analysis"]),
        "--planner-trace",
        str(paths["planner"]),
        "--executor-trace",
        str(paths["executor"]),
        "--gt-evaluator-trace",
        str(paths["evaluator"]),
    ]
    if getattr(args, "oracle_step_count", False):
        command.append("--oracle-step-count")
    if getattr(args, "disable_gt_assessment", False):
        command.append("--disable-gt-assessment")
    if getattr(args, "disable_planner_skills", False):
        command.append("--disable-planner-skills")
    if getattr(args, "reasoning_effort", None):
        command += ["--reasoning-effort", str(args.reasoning_effort)]
    with paths["console"].open("w", encoding="utf-8") as console:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            stdout=console,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        return job, f"failed:{completed.returncode}"

    record = _normalize_record(_read_one_record(paths["analysis"]), job, args.repeats)
    paths["result"].write_text(
        json.dumps(record, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return job, "completed"


def _merge_results(
    output_root: Path,
    jobs: List[RolloutJob],
    target: Path,
    *,
    require_assessment: bool = True,
) -> int:
    records: List[str] = []
    for job in jobs:
        path = _job_paths(output_root, job)["result"]
        if path.exists():
            record = _read_one_record(path)
            if not require_assessment or _has_complete_assessment(record):
                records.append(json.dumps(record, ensure_ascii=False, default=str))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text("\n".join(records) + ("\n" if records else ""), encoding="utf-8")
    temporary.replace(target)
    return len(records)


def main() -> None:
    args = build_parser().parse_args()
    if args.workers < 1 or args.repeats < 1:
        raise ValueError("--workers and --repeats must be positive")
    api_key = os.environ.get("MECH_API_KEY")
    if not api_key:
        raise ValueError("MECH_API_KEY must be set")

    repo_root = Path(__file__).resolve().parents[1]
    output_root = (repo_root / args.output_root).resolve()
    merged_output = (repo_root / args.merged_output).resolve()
    core_dir = Path(args.core_dir).expanduser()
    if not core_dir.is_absolute():
        core_dir = (repo_root / core_dir).resolve()
    args.core_dir = str(core_dir)

    bridge = ChemistryBridge(str(repo_root))
    tasks = bridge.load_tasks(
        data_source="core",
        split="all",
        extracted_dir="unused",
        core_dir=str(core_dir),
        core_subset=args.core_subset,
        max_tasks=None,
        reaction_id=None,
        mechfinder_num_classes=args.num_classes,
        mechfinder_per_class=1,
    )
    if args.core_subset == "mechfinder" and len(tasks) != args.num_classes:
        raise ValueError(f"expected {args.num_classes} tasks, loaded {len(tasks)}")
    if args.core_subset != "mechfinder":
        tasks = tasks[: args.num_classes] if args.num_classes else tasks
    jobs = [
        RolloutJob(task_index=index, reaction_id=str(task.mechanism_id), repeat_index=repeat)
        for index, task in enumerate(tasks, start=1)
        for repeat in range(1, args.repeats + 1)
    ]

    env = dict(os.environ)
    no_proxy = "localhost,127.0.0.1," + env.get("NO_PROXY", "")
    env.update(
        {
            "NO_PROXY": no_proxy,
            "no_proxy": no_proxy,
            "PLANNER_API_KEY": api_key,
            "EXECUTOR_API_KEY": api_key,
            "GT_EVALUATOR_API_KEY": api_key,
        }
    )

    lock = threading.Lock()
    counters = {"completed": 0, "skipped": 0, "failed": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _run_job,
                job=job,
                args=args,
                repo_root=repo_root,
                output_root=output_root,
                env=env,
            )
            for job in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            job, status = future.result()
            category = status.split(":", 1)[0]
            with lock:
                counters[category] = counters.get(category, 0) + 1
                done = sum(counters.values())
                merged = _merge_results(output_root, jobs, merged_output, require_assessment=not args.disable_gt_assessment)
                print(
                    f"[{done}/{len(jobs)}] task={job.task_index} repeat={job.repeat_index} "
                    f"status={status} merged={merged} counters={counters}",
                    flush=True,
                )

    merged = _merge_results(output_root, jobs, merged_output, require_assessment=not args.disable_gt_assessment)
    print(json.dumps({"jobs": len(jobs), "merged": merged, **counters}, indent=2), flush=True)
    if counters.get("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
