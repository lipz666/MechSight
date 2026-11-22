"""Master offline verification runner across all Supporting Information sections (S01-S12).

Usage:
  python si_experiments/verify_all.py           # fast summary + prompt parity checks
  python si_experiments/verify_all.py --full    # also recompute from unpacked raw outputs
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify SI Sections S01-S12")
    ap.add_argument("--full", action="store_true", help="Recompute from unpacked raw outputs (--recompute)")
    args = ap.parse_args()

    sections = sorted(p for p in HERE.iterdir() if p.is_dir() and p.name.startswith("S") and (p / "reproduce.py").exists())
    assert len(sections) == 12, f"Expected 12 SI sections, found {len(sections)}"

    for sec in sections:
        spec = importlib.util.spec_from_file_location(f"si_{sec.name}", sec / "reproduce.py")
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.verify(recompute=args.full)

    print(f"\nALL {len(sections)} SI SECTIONS VERIFIED SUCCESSFULLY" + (" (FULL OFFLINE RECOMPUTE)" if args.full else " (FROZEN SUMMARIES)"))


if __name__ == "__main__":
    main()
