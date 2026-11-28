#!/usr/bin/env python3
"""Compose the pinned F32 rounding aggregate with the single F64 broadcast edit."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

ROUND = "32f1d664826a1ccf524d8fc049889f439f164c7c6cb6a28eae3054691af97d82"
BASE = "7ae4ad04e35ac014737b38979d66173af1b0b7c8611c84740525f4f1f0aa5070"
BROADCAST = "3bf860eafefd7cecfc34b9b3024b9583a8c3dd494c3012c01392500356446efe"


def exact(path, digest):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"unexpected SHA-256: {path}")
    return raw


def split(raw):
    marker = b"\nfn opt_in_copysign_f64("
    if raw.count(marker) != 1:
        raise ValueError("expected exactly one F64 copysign definition")
    start = raw.index(marker) + 1
    end = raw.index(b"\nend\n", start) + 5
    return raw[:start], raw[start:end], raw[end:]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime-tools", "rounding-source", "copysign-base", "broadcast-source", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--expected-build", type=Path)
    a = p.parse_args()
    pre, old, post = split(exact(a.copysign_base, BASE))
    bpre, new, bpost = split(exact(a.broadcast_source, BROADCAST))
    rpre, current, rpost = split(exact(a.rounding_source, ROUND))
    if (pre, post) != (bpre, bpost) or current != old or current == new:
        raise ValueError("predecessor delta overlaps another function")
    with tempfile.TemporaryDirectory(prefix="forward-integrated-") as tmp:
        src = Path(tmp) / "runtime.hepta"
        src.write_bytes(rpre + new + rpost)
        command = [sys.executable, "-I", "-B", str(a.runtime_tools / "partition.py"),
                   "--runtime", str(src), "--name", "typed-relu775-opt-in-rounding-and-f64-broadcast",
                   "--prefix", "forward782", "--output", str(a.output)]
        if a.expected_build:
            command += ["--expected-build", str(a.expected_build)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
