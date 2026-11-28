#!/usr/bin/env python3
"""Project opt-in integral whole-tensor product from exact main."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

MAIN = "0b000b51288af20490162efd8e525b86359a32cdd6d2521df97ecd9cddbd7b43"
HERE = Path(__file__).resolve().parent
OLD = b"  branch failed failure() route(s,operation,input)\n"
NEW = b"""  branch failed failure() integral_prod_route(s,operation,input)
block integral_prod_route(sp:mutbuf,opp:u64,xp:u64)
  product_op:u64 = const() {"value":22}
  asked_p:bool = compare.eq(opp,product_op)
  branch asked_p integral_dtype(sp,opp,xp) route(sp,opp,xp)
block integral_dtype(sd:mutbuf,opd:u64,xd:u64)
  dtype_p:u64 = call(sd,xd) {"function":"tensor_dtype"}
  bool_p:u64 = const() {"value":1}
  i64_p:u64 = const() {"value":2}
  is_bool_p:bool = compare.eq(dtype_p,bool_p)
  is_i64_p:bool = compare.eq(dtype_p,i64_p)
  integral_p:bool = bool.or(is_bool_p,is_i64_p)
  branch integral_p native_integral_prod(sd,xd) route(sd,opd,xd)
block native_integral_prod(sip:mutbuf,xip:u64)
  yp:u64 = call(sip,xip) {"function":"opt_in_integral_prod"}
  ret(yp)
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-tools", type=Path, default=HERE.parent)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--expected-build", type=Path)
    args = p.parse_args()
    runtime = args.runtime_tools.resolve()
    parent = (runtime / "source/runtime.hepta").read_bytes()
    if hashlib.sha256(parent).hexdigest() != MAIN:
        raise ValueError("unexpected canonical HIA source")
    start = parent.index(b"\nfn unary(")
    end = parent.index(b"\nend\n", start)
    body = parent[start:end]
    if body.count(OLD) != 1:
        raise ValueError("unexpected unary dispatch")
    helper = (HERE / "prod.hepta").read_bytes()
    if not helper.startswith(b"fn opt_in_integral_prod(") or not helper.endswith(b"end\n"):
        raise ValueError("invalid product helper extent")
    candidate = parent[:start] + body.replace(OLD, NEW, 1) + parent[end:] + helper
    with tempfile.TemporaryDirectory(prefix="heptatorch-integral-prod-") as temporary:
        source = Path(temporary) / "runtime.hepta"
        source.write_bytes(candidate)
        argv = [sys.executable, "-I", "-B", str(runtime / "partition.py"), "--runtime", str(source),
                "--name", "typed-relu775-opt-in-integral-prod", "--prefix", "integralprod776",
                "--output", str(args.output)]
        if args.expected_build:
            argv.extend(["--expected-build", str(args.expected_build)])
        subprocess.run(argv, check=True)


if __name__ == "__main__":
    main()
