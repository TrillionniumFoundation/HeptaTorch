#!/usr/bin/env python3
"""Recreate and verify the twelve frozen module sources without running native tools."""
import sys
sys.dont_write_bytecode = True
from rebuild import main

if __name__ == "__main__":
    raise SystemExit(main(check_only=True))
