#!/usr/bin/env python3
"""Build the source projection against the exact native F64 copysign parent."""
import argparse
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PARENT = "3bf860eafefd7cecfc34b9b3024b9583a8c3dd494c3012c01392500356446efe"
OLD_BODY = "926f63d70dcd2c283c0bea002001fee71526006d3a3ff8563342a4befa72b73d"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-tools", type=Path, default=HERE.parent)
    p.add_argument("--parent-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-build", type=Path)
    a = p.parse_args()
    parent = a.parent_source.read_bytes()
    old = (HERE.parent/"opt_in_copysign_f32/copysign.hepta").read_bytes()
    if (sha256(parent).hexdigest() != PARENT or sha256(old).hexdigest() != OLD_BODY
            or parent.count(old) != 1):
        raise ValueError("expected exact #96 F64 broadcast source projection")
    new = (HERE/"copysign.hepta").read_bytes()
    if not new.startswith(b"fn opt_in_copysign_f32(") or not new.endswith(b"end\n"):
        raise ValueError("expected one replacement body")
    with tempfile.TemporaryDirectory(prefix="heptatorch-copysign-broadcast-") as scratch:
        path = Path(scratch)/"runtime.hepta"
        path.write_bytes(parent.replace(old, new, 1))
        command = [sys.executable, "-I", "-B", str(a.runtime_tools.resolve()/"partition.py"),
                   "--runtime", str(path), "--name", "typed-relu775-opt-in-f32-f64-copysign-broadcast",
                   "--prefix", "copysign777", "--output", str(a.output)]
        if a.expected_build:
            command.extend(["--expected-build", str(a.expected_build)])
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
