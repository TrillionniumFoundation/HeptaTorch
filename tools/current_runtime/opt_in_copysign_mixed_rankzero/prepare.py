#!/usr/bin/env python3
"""Build the source projection against the exact mixed copysign parent."""
import argparse
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PARENT = "cd40ba41425f1c65f4c6d2463134082c3c64756fb322e63ec93c64c4c8838b1a"
OLD_BODY = "397b60ea2420d7aa1bb331de139e82280f9cbe2a2bca1acc09ff7144ca5c709f"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-tools", type=Path, default=HERE.parent)
    p.add_argument("--parent-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-build", type=Path)
    a = p.parse_args()
    parent = a.parent_source.read_bytes()
    old = (HERE.parent/"opt_in_copysign_mixed_broadcast/copysign.hepta").read_bytes()
    if (sha256(parent).hexdigest() != PARENT or sha256(old).hexdigest() != OLD_BODY
            or parent.count(old) != 1 or not parent.endswith(old)):
        raise ValueError("expected exact #104 mixed source projection")
    new = (HERE/"copysign.hepta").read_bytes()
    if not new.startswith(b"fn opt_in_copysign_mixed(") or not new.endswith(b"end\n"):
        raise ValueError("expected one replacement body")
    with tempfile.TemporaryDirectory(prefix="heptatorch-copysign-broadcast-") as scratch:
        path = Path(scratch)/"runtime.hepta"
        path.write_bytes(parent[:-len(old)] + new)
        command = [sys.executable, "-I", "-B", str(a.runtime_tools.resolve()/"partition.py"),
                   "--runtime", str(path), "--name", "typed-relu775-opt-in-mixed-copysign-rankzero",
                   "--prefix", "copysign778", "--output", str(a.output)]
        if a.expected_build:
            command.extend(["--expected-build", str(a.expected_build)])
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
