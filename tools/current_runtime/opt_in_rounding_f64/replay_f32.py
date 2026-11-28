#!/usr/bin/env python3
"""Replay exact #97 F32 checks, retaining a now-correct nonfloating guard."""
import argparse
import hashlib
from pathlib import Path
import sys

SCRIPT = "bbb6053445f2a4293347d12f7fe840611b3fa1db6a81dd632ea7a650c30d7970"
ELF = "949c8435546617dcf3e6298b5393b76b2dac2cf2b8b183fd82cb7d9bf3299eb8"
F64_GUARD = b'(97, 4, wire.Payload(struct.pack("<9Q", 1, 0, 1, 0, 0, 0, 160, 8, 0x3FF0000000000000)))'
I64_GUARD = b'(97, 2, wire.Payload(struct.pack("<9Q", 1, 0, 1, 0, 0, 0, 160, 8, 1)))'


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--f32-verifier", type=Path, required=True)
    parsed, forwarded = parser.parse_known_args()
    path = parsed.f32_verifier.resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SCRIPT or raw.count(F64_GUARD) != 1:
        raise ValueError("predecessor F32 verifier is not the exact #97 script")
    # Only the dtype-boundary fixture changes: the candidate now admits F64.
    namespace = {"__name__": "pinned_f32_verifier", "__file__": str(path)}
    exec(compile(raw.replace(F64_GUARD, I64_GUARD, 1), str(path), "exec"), namespace)
    namespace["CANDIDATE"] = ELF
    saved = sys.argv
    try:
        sys.argv = [str(path), *forwarded]
        namespace["main"]()
    finally:
        sys.argv = saved


if __name__ == "__main__":
    main()
