"""Shared path resolution, JSON loading, and subprocess helpers for si_experiments."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = REPO_ROOT / "reproduction"
MECHAGENT_ROOT = REPRO_ROOT / "MechAgent"
PILOTS_V4_DIR = MECHAGENT_ROOT / "experiments" / "pilots_v4"
RESULTS_ROOT = REPO_ROOT / "results"
CANONICAL_ROOT = RESULTS_ROOT / "canonical_analyses"
PAPER_SUMMARIES_ROOT = RESULTS_ROOT / "paper_summaries"
RAW_OUTPUTS_V4 = RESULTS_ROOT / "raw_outputs" / "pilots_v4_20261103"
RAW_STORYLINE = RESULTS_ROOT / "raw_outputs" / "03_early_exploration_storyline_package"


def load_json(rel_path: str | Path) -> Any:
    """Load a JSON file relative to ``REPO_ROOT`` (or absolute path)."""
    p = Path(rel_path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return json.loads(p.read_text())


def raw_outputs_ready() -> bool:
    """Return True if `./unpack_raw_outputs.sh` has restored raw outputs and symlinks."""
    return (RAW_OUTPUTS_V4 / "T3_confirm").exists() and (
        MECHAGENT_ROOT / "outputs" / "benchmarks_20261103" / "splits.json"
    ).exists()


def require_raw_outputs() -> None:
    """Raise an informative error if `--recompute` is run before `./unpack_raw_outputs.sh`."""
    if not raw_outputs_ready():
        raise RuntimeError(
            "Unpacked raw outputs not found. Run `./unpack_raw_outputs.sh` from the repository root first."
        )


def run_pilots_script(
    script_path: str | Path,
    args: list[str] | None = None,
    *,
    extra_env: dict[str, str] | None = None,
) -> None:
    """Execute a frozen analysis script with cwd=``reproduction/MechAgent/experiments/pilots_v4``."""
    p = Path(script_path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    subprocess.run(
        [sys.executable, str(p), *(args or [])],
        cwd=str(PILOTS_V4_DIR),
        env=env,
        check=True,
    )
