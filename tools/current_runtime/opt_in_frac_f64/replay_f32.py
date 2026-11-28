#!/usr/bin/env python3
"""Replay exact #109 F32 frac checks, retaining a now-correct nonfloating guard."""
import argparse
import hashlib
from pathlib import Path
import sys

SCRIPT = "19f7d91ab895c33684211aaa1d7b8ff7fbb8e91750ced4d5b4485bf420770d4f"
ELF = "782280331abbfd3904b2857371288aa4a0af8386ce71d54f3be50508bc3a34f5"
F64_GUARD = b'(97, 4, wire.Payload(struct.pack("<9Q", 1, 0, 1, 0, 0, 0, 160, 8, 0x3FF0000000000000)))'
I64_GUARD = b'(97, 2, wire.Payload(struct.pack("<9Q", 1, 0, 1, 0, 0, 0, 160, 8, 1)))'


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--f32-verifier", type=Path, required=True)
    parsed, forwarded = parser.parse_known_args()
    path = parsed.f32_verifier.resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SCRIPT or raw.count(F64_GUARD) != 1:
        raise ValueError("predecessor F32 verifier is not the exact #109 script")
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
