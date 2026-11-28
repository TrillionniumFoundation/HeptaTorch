#!/usr/bin/env python3
"""Build the source projection against the exact native F64 copysign parent."""
import argparse
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PARENT = "7ae4ad04e35ac014737b38979d66173af1b0b7c8611c84740525f4f1f0aa5070"
OLD_BODY = "ea96eeedaaf8c4cdab3e33ec7382eefcd6eb4531a93d86e612b6089eba8bc56e"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-tools", type=Path, default=HERE.parent)
    p.add_argument("--parent-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-build", type=Path)
    a = p.parse_args()
    parent = a.parent_source.read_bytes()
    old = (HERE.parent/"opt_in_copysign_f64/copysign.hepta").read_bytes()
    if (sha256(parent).hexdigest() != PARENT or sha256(old).hexdigest() != OLD_BODY
            or parent.count(old) != 1 or not parent.endswith(old)):
        raise ValueError("expected exact #90 F64 copysign source projection")
    new = (HERE/"copysign.hepta").read_bytes()
    if not new.startswith(b"fn opt_in_copysign_f64(") or not new.endswith(b"end\n"):
        raise ValueError("expected one replacement body")
    with tempfile.TemporaryDirectory(prefix="heptatorch-copysign-broadcast-") as scratch:
        path = Path(scratch)/"runtime.hepta"
        path.write_bytes(parent[:-len(old)] + new)
        command = [sys.executable, "-I", "-B", str(a.runtime_tools.resolve()/"partition.py"),
                   "--runtime", str(path), "--name", "typed-relu775-opt-in-f64-copysign-broadcast",
                   "--prefix", "copysign777", "--output", str(a.output)]
        if a.expected_build:
            command.extend(["--expected-build", str(a.expected_build)])
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
