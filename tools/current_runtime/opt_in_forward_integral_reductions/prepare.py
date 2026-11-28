#!/usr/bin/env python3
"""Compose exact #103 forward operations with the two #105 reduction helpers."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

FORWARD = "4580f9cdc0cf559c9f8595d6e88d71ed3d1f596182a9c0a1c5a4c5727d53ca08"
CANONICAL = "0b000b51288af20490162efd8e525b86359a32cdd6d2521df97ecd9cddbd7b43"
REDUCTIONS = "a4371f0831d02e764bf27710abbbff99c58b256e39daf578c8367e93a8731396"
OLD = b"  branch failed failure() route(s,operation,input)\n"
FORWARD_ENTRY = b"  branch failed failure() rounding_route(s,operation,input)\n"
REDUCTION_ENTRY = b"  branch failed failure() extrema_route(s,operation,input)\n"


def exact(path, digest):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"unexpected SHA-256: {path}")
    return raw


def split(raw):
    marker = b"\nfn unary("
    if raw.count(marker) != 1:
        raise ValueError("expected exactly one unary definition")
    start = raw.index(marker) + 1
    end = raw.index(b"\nend\n", start) + 5
    return raw[:start], raw[start:end], raw[end:]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime-tools", "forward-source", "reduction-source", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--expected-build", type=Path)
    a = p.parse_args()
    pre, old, post = split(exact(a.runtime_tools/"source/runtime.hepta", CANONICAL))
    rpre, red, rpost = split(exact(a.reduction_source, REDUCTIONS))
    fpre, forward, fpost = split(exact(a.forward_source, FORWARD))
    if pre != rpre or not rpost.startswith(post) or old.count(OLD) != 1:
        raise ValueError("reduction edits overlap canonical source")
    start, end = old.split(OLD)
    if not (red.startswith(start + REDUCTION_ENTRY) and red.endswith(end)):
        raise ValueError("unexpected reduction dispatcher extent")
    route = red[len(start)+len(REDUCTION_ENTRY):-len(end)]
    if route.count(b") route(") != 2 or forward.count(FORWARD_ENTRY) != 1:
        raise ValueError("unexpected fallback edges")
    route = route.replace(b") route(", b") rounding_route(")
    extra = rpost[len(post):]
    if not extra.startswith(b"fn opt_in_integral_prod(") or extra.count(b"\nfn opt_in_integral_extremum(") != 1:
        raise ValueError("unexpected reduction helper extent")
    merged = forward.replace(FORWARD_ENTRY, REDUCTION_ENTRY + route, 1)
    with tempfile.TemporaryDirectory(prefix="forward-reductions-") as tmp:
        src = Path(tmp) / "runtime.hepta"
        src.write_bytes(fpre + merged + fpost + extra)
        command = [sys.executable, "-I", "-B", str(a.runtime_tools / "partition.py"),
                   "--runtime", str(src), "--name", "typed-relu775-opt-in-forward-reductions",
                   "--prefix", "fwreduce784", "--output", str(a.output)]
        if a.expected_build:
            command += ["--expected-build", str(a.expected_build)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
