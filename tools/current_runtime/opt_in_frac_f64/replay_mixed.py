#!/usr/bin/env python3
"""Extend the pinned #107 mixed session with F32 and F64 frac."""
import argparse
import hashlib
from pathlib import Path
import sys

SCRIPT = "f06033f6e97379c31eeef758f27290361e6b3dfbcaac7d1d8b8e9fb841ffa9ee"
EDITS = {
    b"[((), (0xBFF8000000000000,))], 19)": b"[((), (0xBFF8000000000000,))], 22)",
    b"tail = torch.ops.aten.trunc.default(middle)": b"tail = torch.ops.aten.frac.default(copy)",
    b"terminals = session.run([(3, 19, ids[width][3])": b"terminals = session.run([(3, 22, ids[width][2])",
}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mixed-verifier", type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    path = args.mixed_verifier.resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SCRIPT:
        raise ValueError("mixed verifier is not exact #107")
    for old, new in EDITS.items():
        if raw.count(old) != 1:
            raise ValueError("unexpected mixed verifier source")
        raw = raw.replace(old, new, 1)
    # The verifier imports the new local F64 constructor/oracle and ELF pin.
    namespace = {"__name__": "frac_mixed_verifier", "__file__": __file__}
    exec(compile(raw, str(path), "exec"), namespace)
    saved = sys.argv
    try:
        sys.argv = [str(path), *forwarded]
        namespace["main"]()
    finally:
        sys.argv = saved


if __name__ == "__main__":
    main()
