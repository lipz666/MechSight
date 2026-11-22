"""Verify SI Section S12 prompt templates."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_prompts  # noqa: E402


def verify(*, recompute: bool = False) -> None:
    verify_prompts.main()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    verify(recompute=ap.parse_args().recompute)
