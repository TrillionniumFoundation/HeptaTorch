#!/usr/bin/env python3
"""Materialize a bounded opt-in reciprocal runtime from the exact main source."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
MAIN = "0b000b51288af20490162efd8e525b86359a32cdd6d2521df97ecd9cddbd7b43"
OLD = b'  branch failed failure() route(s,operation,input)\n'
INSERT = b'''  branch failed failure() reciprocal_route(s,operation,input)
block reciprocal_route(sq:mutbuf,oq:u64,xq:u64)
  dtypeq:u64 = call(sq,xq) {"function":"tensor_dtype"}
  f32q:u64 = const() {"value":0}
  reciprocalq:u64 = const() {"value":17}
  typeq:bool = compare.eq(dtypeq,f32q)
  opq:bool = compare.eq(oq,reciprocalq)
  admitsq:bool = bool.and(typeq,opq)
  branch admitsq reciprocal_native(sq,xq) route(sq,oq,xq)
block reciprocal_native(ssq:mutbuf,xsq:u64)
  ysq:u64 = call(ssq,xsq) {"function":"opt_in_reciprocal_f32"}
  ret(ysq)
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runtime-tools", type=Path, default=HERE.parent)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expected-build", type=Path)
    args = ap.parse_args()
    tools = args.runtime_tools.resolve()
    source = (tools / "source/runtime.hepta").read_bytes()
    if hashlib.sha256(source).hexdigest() != MAIN:
        raise ValueError("not canonical typed-ReLU 775 source")
    begin = source.index(b"\nfn unary(") + 1
    end = source.index(b"\nend\n", begin) + len(b"\nend\n")
    unary = source[begin:end]
    helper = (HERE / "reciprocal.hepta").read_bytes()
    if unary.count(OLD) != 1 or not helper.startswith(b"fn opt_in_reciprocal_f32(") or not helper.endswith(b"end\n"):
        raise ValueError("source transition or reciprocal definition malformed")
    candidate = source[:begin] + unary.replace(OLD, INSERT, 1) + source[end:] + helper
    with tempfile.TemporaryDirectory(prefix="heptatorch-reciprocal-projection-") as temp:
        path = Path(temp) / "runtime.hepta"
        path.write_bytes(candidate)
        command = [sys.executable, "-I", "-B", str(tools / "partition.py"),
                   "--runtime", str(path), "--name", "typed-relu775-opt-in-reciprocal-f32",
                   "--prefix", "reciprocal776", "--output", str(args.output)]
        if args.expected_build:
            command += ["--expected-build", str(args.expected_build)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
