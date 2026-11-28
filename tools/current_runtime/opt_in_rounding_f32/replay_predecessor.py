#!/usr/bin/env python3
"""Replay the exact #93 qualification script; only substitute the new ELF pin."""
import argparse
import hashlib
from pathlib import Path
import sys

SCRIPT_PIN = "2a0b6892571c364be01318819b086ccbb150d5d28e35d72383921d39e396dabc"
ELF = "c5f542e32675e3e058ca409a5bda57f092f344b8d0a58903797731937f6652c6"


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--previous-verifier", required=True, type=Path)
    parsed, remaining = parser.parse_known_args()
    path = parsed.previous_verifier.resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SCRIPT_PIN:
        raise ValueError("predecessor verifier byte identity is not exact #93")
    namespace = {"__name__": "pinned_previous_verifier", "__file__": str(path)}
    exec(compile(raw, str(path), "exec"), namespace)
    namespace["ELF"] = ELF
    saved = sys.argv
    try:
        sys.argv = [str(path), *remaining]
        namespace["main"]()
    finally:
        sys.argv = saved


if __name__ == "__main__":
    main()
