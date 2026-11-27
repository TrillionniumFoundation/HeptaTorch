#!/usr/bin/env python3
"""Combine the exact prepared three-unary and two-dtype copysign aggregates."""
import argparse
from pathlib import Path
import hashlib
import subprocess
import sys
import tempfile

BASE = "0b000b51288af20490162efd8e525b86359a32cdd6d2521df97ecd9cddbd7b43"
UNARY = "ccb36fcb59931bc92f65f89e71c919083f5f328518e0c19139c07bdd6c4ba776"
COPY = "7ae4ad04e35ac014737b38979d66173af1b0b7c8611c84740525f4f1f0aa5070"


def pinned(path, sha):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError(f"unexpected input SHA-256: {path.name}")
    return raw


def extent(raw, name):
    start = raw.index(b'\nfn ' + name + b'(') + 1
    end = raw.index(b'\nend\n', start) + len(b'\nend\n')
    return start, end


def body(raw, name):
    start, end = extent(raw, name)
    return raw[start:end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-tools", type=Path, default=Path(__file__).resolve().parents[1])
    for name in ("unary-source", "copysign-source", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-build", type=Path)
    a = parser.parse_args()
    base = pinned(a.runtime_tools / "source/runtime.hepta", BASE)
    unary = pinned(a.unary_source, UNARY)
    copy = pinned(a.copysign_source, COPY)
    start, end = extent(base, b"unary")
    replacement = body(unary,b"unary")
    old = base[start:end]
    # The unary predecessor is exactly one changed function and three
    # appended functions. The copysign predecessor still has the old unary.
    prefix = base[:start] + replacement + base[end:]
    if not unary.startswith(prefix) or body(copy,b"unary") != old:
        raise ValueError("predecessor edits overlap")
    extra = unary[len(prefix):]
    copy_start, copy_end = extent(copy, b"unary")
    combined = copy[:copy_start] + replacement + copy[copy_end:] + extra
    with tempfile.TemporaryDirectory(prefix="heptatorch-mix-") as temp:
        src = Path(temp) / "runtime.hepta"
        src.write_bytes(combined)
        command = [sys.executable, "-I", "-B", str(a.runtime_tools.resolve() / "partition.py"),
                   "--runtime", str(src), "--name", "typed-relu775-opt-in-unary-and-copysign",
                   "--prefix", "mix780", "--output", str(a.output)]
        if a.expected_build:
            command += ["--expected-build", str(a.expected_build)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
