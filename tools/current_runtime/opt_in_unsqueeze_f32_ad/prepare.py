#!/usr/bin/env python3
"""Layer bounded first-order unsqueeze AD over exact #86 opt-in source."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PARENT = "2336ca87ac4fce0b46380c3a3d8539d230ff5c765c1c6ba4069d08617d614177"
OLD = b"  branch graph_free wrap(sg,ng,dg,pg,rg) grad_error(sg)\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runtime-tools", type=Path, default=HERE.parent)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    parent = args.parent.read_bytes()
    if hashlib.sha256(parent).hexdigest() != PARENT or parent.count(OLD) != 1:
        raise ValueError("exact graph-free F32 unsqueeze parent required")
    candidate = parent.replace(OLD, (HERE / "grad_route.hepta").read_bytes(), 1)
    with tempfile.TemporaryDirectory(prefix="heptatorch-unsqueeze-ad-") as directory:
        path = Path(directory) / "runtime.hepta"
        path.write_bytes(candidate)
        subprocess.run([sys.executable, "-I", "-B", str(args.runtime_tools.resolve() / "partition.py"),
                        "--runtime", str(path), "--name", "typed-relu775-opt-in-unsqueeze-f32-ad",
                        "--prefix", "unsqueezead778", "--output", str(args.output)], check=True)


if __name__ == "__main__":
    main()
