#!/usr/bin/env python3
"""Current-image entry; frozen original assertions remain in the source fixture."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
entry = ROOT / "tests/current_runtime_historical/run.py"
sys.argv = [str(entry), "--suite", 'bool27', *sys.argv[1:]]
runpy.run_path(str(entry), run_name="__main__")
