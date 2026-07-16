#!/usr/bin/env python3
"""Command-line wrapper for the within-session decoding benchmark."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python"))

from ppc_direction_decoding.within_session_benchmark import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
