#!/usr/bin/env python3
"""Run cross-session fixed-memory 2-target linear decoding baselines."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python"))

from ppc_direction_decoding.cross_session import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
