#!/usr/bin/env python3
"""Project whole-tensor integral min/max onto the exact #100 source."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

PARENT = "21e2b3645d7536b25e32bc58d1a9e5756c864b3811293c1185c0cd4d6c70ce9d"
HERE = Path(__file__).resolve().parent
OLD = b"  branch failed failure() integral_prod_route(s,operation,input)\n"
NEW = b"""  branch failed failure() extrema_route(s,operation,input)
block extrema_route(se:mutbuf,ope:u64,xe:u64)
  minop:u64 = const() {"value":23}
  maxop:u64 = const() {"value":24}
  min_e:bool = compare.eq(ope,minop)
  max_e:bool = compare.eq(ope,maxop)
  is_e:bool = bool.or(min_e,max_e)
  branch is_e extrema_dtype(se,ope,xe) integral_prod_route(se,ope,xe)
block extrema_dtype(sed:mutbuf,oped:u64,xed:u64)
  de:u64 = call(sed,xed) {"function":"tensor_dtype"}
  one_e:u64 = const() {"value":1}
  two_e:u64 = const() {"value":2}
  bool_e:bool = compare.eq(de,one_e)
  i64_e:bool = compare.eq(de,two_e)
  ok_e:bool = bool.or(bool_e,i64_e)
  branch ok_e extrema_native(sed,xed,de,oped) integral_prod_route(sed,oped,xed)
block extrema_native(sen:mutbuf,xen:u64,den:u64,oen:u64)
  ye:u64 = call(sen,xen,den,oen) {"function":"opt_in_integral_extremum"}
  ret(ye)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-tools", type=Path, default=HERE.parent)
    parser.add_argument("--parent-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-build", type=Path)
    a = parser.parse_args()
    parent = a.parent_source.read_bytes()
    if hashlib.sha256(parent).hexdigest() != PARENT:
        raise ValueError("unexpected #100 source")
    start = parent.index(b"\nfn unary(")
    end = parent.index(b"\nend\n", start)
    body = parent[start:end]
    if body.count(OLD) != 1:
        raise ValueError("unexpected #100 unary entry")
    helper = (HERE / "minmax.hepta").read_bytes()
    if not helper.startswith(b"fn opt_in_integral_extremum(") or not helper.endswith(b"end\n"):
        raise ValueError("malformed HIA helper extent")
    with tempfile.TemporaryDirectory(prefix="heptatorch-integral-extrema-") as temporary:
        src = Path(temporary) / "runtime.hepta"
        src.write_bytes(parent[:start] + body.replace(OLD, NEW, 1) + parent[end:] + helper)
        argv = [sys.executable, "-I", "-B", str(a.runtime_tools.resolve() / "partition.py"),
                "--runtime", str(src), "--name", "typed-relu775-opt-in-integral-prod-minmax",
                "--prefix", "integralextrema777", "--output", str(a.output)]
        if a.expected_build:
            argv.extend(["--expected-build", str(a.expected_build)])
        subprocess.run(argv, check=True)


if __name__ == "__main__":
    main()
