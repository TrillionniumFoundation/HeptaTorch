#!/usr/bin/env python3
"""Project exceptional square gradient words over the exact #87 source."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
SOURCE = "0343f5706e8dd1346c2940d94b94a82574533c3f89fee3a24131df031226e9f7"
PREVIOUS = b'''  two:u64 = const() {"value":1073741824}
  twice:u64 = call(two,x) {"function":"mulf"}
  contribution:u64 = call(g,twice) {"function":"mulf"}
'''
NEXT = b'''  contribution:u64 = call(g,x) {"function":"opt_in_square_f32_gradient_word"}
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-tools", type=Path, default=HERE.parent)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-build", type=Path)
    a = p.parse_args()
    tools = a.runtime_tools.resolve()
    with tempfile.TemporaryDirectory(prefix="heptatorch-square-special-") as tmp:
        directory = Path(tmp)
        source_root = directory / "parent"
        subprocess.run([sys.executable, "-I", "-B", str(HERE.parent / "opt_in_square_f32_ordered/prepare.py"),
                        "--runtime-tools", str(tools), "--output", str(source_root)], check=True)
        source = (source_root / "runtime.hepta").read_bytes()
        if hashlib.sha256(source).hexdigest() != SOURCE or source.count(PREVIOUS) != 1:
            raise ValueError("not the exact #87 ordered square source")
        candidate = directory / "runtime.hepta"
        candidate.write_bytes(source.replace(PREVIOUS, NEXT, 1) + (HERE / "special.hepta").read_bytes())
        command = [sys.executable, "-I", "-B", str(tools / "partition.py"), "--runtime", str(candidate),
                   "--name", "typed-relu775-opt-in-square-f32-special-ad",
                   "--prefix", "squarespecial776", "--output", str(a.output)]
        if a.expected_build:
            command += ["--expected-build", str(a.expected_build)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
