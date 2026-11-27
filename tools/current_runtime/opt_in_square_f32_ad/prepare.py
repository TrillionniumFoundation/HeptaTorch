#!/usr/bin/env python3
"""Build the independent opt-in square AD source over the exact #74 projection."""

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PARENT_DIR = HERE.parent / "opt_in_square_f32"
PARENT_SOURCE = "12986a5280ced833a27c60bf773a866b76aeeb9dc8590a3774028d7c47b3fdd6"
BEFORE = b"""  branch grad_ok geometry(s,x,p) no_ad(s)
block no_ad(sg:mutbuf)
  codeg:u64 = const() {"value":5}
  errg:u64 = call(sg,codeg) {"function":"terror"}
  ret(errg)
"""
AFTER = b"""  jump geometry(s,x,p,grad_ok)
"""
BEFORE_GEOMETRY = b"""block geometry(st:mutbuf,xt:u64,pt:u64)
"""
AFTER_GEOMETRY = b"""block geometry(st:mutbuf,xt:u64,pt:u64,no_grad:bool)
"""
BEFORE_ROUTE = b"""  branch admitted allocate(st,xt,pt) unsupported(st)
"""
AFTER_ROUTE = b"""  branch admitted grad_route(st,xt,pt,no_grad) unsupported(st)
block grad_route(sr:mutbuf,xr:u64,pr:u64,nogradr:bool)
  branch nogradr allocate(sr,xr,pr) grad_product(sr,xr)
block grad_product(sb:mutbuf,xb:u64)
  opb:u64 = const() {"value":3}
  yb:u64 = call(sb,opb,xb,xb) {"function":"binary"}
  ret(yb)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-tools", type=Path, default=HERE.parent)
    parser.add_argument("--parent-projection", type=Path, default=PARENT_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-build", type=Path)
    a = parser.parse_args()
    tools, parent_script = a.runtime_tools.resolve(), a.parent_projection.resolve() / "prepare.py"
    with tempfile.TemporaryDirectory(prefix="heptatorch-square-ad-") as directory:
        temp = Path(directory)
        subprocess.run([sys.executable, "-I", "-B", str(parent_script), "--runtime-tools",
                        str(tools), "--output", str(temp / "parent")], check=True)
        source = (temp / "parent/runtime.hepta").read_bytes()
        if hashlib.sha256(source).hexdigest() != PARENT_SOURCE:
            raise ValueError("not the exact opt-in square #74 aggregate source")
        split = source.index(b"\nfn opt_in_square_f32(") + 1
        prefix, helper = source[:split], source[split:]
        for before, after in ((BEFORE, AFTER), (BEFORE_GEOMETRY, AFTER_GEOMETRY), (BEFORE_ROUTE, AFTER_ROUTE)):
            if helper.count(before) != 1:
                raise ValueError("not the exact square helper")
            helper = helper.replace(before, after, 1)
        path = temp / "candidate.hepta"
        path.write_bytes(prefix + helper)
        command = [sys.executable, "-I", "-B", str(tools / "partition.py"),
                   "--runtime", str(path), "--name", "typed-relu775-opt-in-square-f32-ad",
                   "--prefix", "squaread776", "--output", str(a.output)]
        if a.expected_build:
            command += ["--expected-build", str(a.expected_build)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
