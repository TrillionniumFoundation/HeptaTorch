#!/usr/bin/env python3
"""Project last-axis integral product onto the exact #105 source."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

PARENT = "a4371f0831d02e764bf27710abbbff99c58b256e39daf578c8367e93a8731396"
HERE = Path(__file__).resolve().parent
OLD = b"  branch failed failure() extrema_route(s,operation,input)\n"
NEW = b"""  branch failed failure() last_product_route(s,operation,input)
block last_product_route(se_lp:mutbuf,ope_lp:u64,xe_lp:u64)
  last_prod_lp:u64 = const() {"value":25}
  last_e_lp:bool = compare.eq(ope_lp,last_prod_lp)
  branch last_e_lp last_product_dtype(se_lp,ope_lp,xe_lp) extrema_route(se_lp,ope_lp,xe_lp)
block last_product_dtype(sed_lp:mutbuf,oped_lp:u64,xed_lp:u64)
  de_lp:u64 = call(sed_lp,xed_lp) {"function":"tensor_dtype"}
  one_e_lp:u64 = const() {"value":1}
  two_e_lp:u64 = const() {"value":2}
  bool_e_lp:bool = compare.eq(de_lp,one_e_lp)
  i64_e_lp:bool = compare.eq(de_lp,two_e_lp)
  ok_e_lp:bool = bool.or(bool_e_lp,i64_e_lp)
  branch ok_e_lp last_product_native(sed_lp,xed_lp) extrema_route(sed_lp,oped_lp,xed_lp)
block last_product_native(sen_lp:mutbuf,xen_lp:u64)
  ye_lp:u64 = call(sen_lp,xen_lp) {"function":"opt_in_integral_prod_last"}
  ret(ye_lp)
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
        raise ValueError("unexpected #105 source")
    start = parent.index(b"\nfn unary(")
    end = parent.index(b"\nend\n", start)
    body = parent[start:end]
    if body.count(OLD) != 1:
        raise ValueError("unexpected #105 unary entry")
    helper = (HERE / "prod_last.hepta").read_bytes()
    if not helper.startswith(b"fn opt_in_integral_prod_last(") or not helper.endswith(b"end\n"):
        raise ValueError("malformed HIA helper extent")
    with tempfile.TemporaryDirectory(prefix="heptatorch-integral-last-prod-") as temporary:
        src = Path(temporary) / "runtime.hepta"
        src.write_bytes(parent[:start] + body.replace(OLD, NEW, 1) + parent[end:] + helper)
        argv = [sys.executable, "-I", "-B", str(a.runtime_tools.resolve() / "partition.py"),
                "--runtime", str(src), "--name", "typed-relu775-opt-in-integral-reductions-last-prod",
                "--prefix", "integrallastprod778", "--output", str(a.output)]
        if a.expected_build:
            argv.extend(["--expected-build", str(a.expected_build)])
        subprocess.run(argv, check=True)


if __name__ == "__main__":
    main()
