#!/usr/bin/env python3
"""Add three ordinary F32 rounding operations to the exact #93 source."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PARENT_SOURCE = "8ed42a1695f78f5ac5d81749fc76f95eb02e59262e5265a4f2a14f9576eba08b"
OLD = b"  branch failed failure() square_route(s,operation,input)\n"
ROUTE = b'''  branch failed failure() rounding_route(s,operation,input)
block rounding_route(srnd:mutbuf,ornd:u64,xrnd:u64)
  lower_rnd:u64 = const() {"value":19}
  upper_rnd:u64 = const() {"value":21}
  above_rnd:bool = compare.ge(ornd,lower_rnd)
  below_rnd:bool = compare.le(ornd,upper_rnd)
  rounding_rnd:bool = bool.and(above_rnd,below_rnd)
  branch rounding_rnd rounding_dtype(srnd,ornd,xrnd) square_route(srnd,ornd,xrnd)
block rounding_dtype(sdr:mutbuf,odr:u64,xdr:u64)
  dtype_rnd:u64 = call(sdr,xdr) {"function":"tensor_dtype"}
  f32_rnd:u64 = const() {"value":0}
  f32_ok_rnd:bool = compare.eq(dtype_rnd,f32_rnd)
  branch f32_ok_rnd rounding_native(sdr,odr,xdr) rounding_refuse(sdr)
block rounding_refuse(srr:mutbuf)
  err_rnd:u64 = const() {"value":7}
  result_rnd:u64 = call(srr,err_rnd) {"function":"terror"}
  ret(result_rnd)
block rounding_native(snr:mutbuf,onr:u64,xnr:u64)
  ynr:u64 = call(snr,xnr,onr) {"function":"opt_in_round_f32"}
  ret(ynr)
'''


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runtime-tools", type=Path, default=HERE.parent)
    ap.add_argument("--parent-source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expected-build", type=Path)
    args = ap.parse_args()
    data = args.parent_source.read_bytes()
    if digest(data) != PARENT_SOURCE or data.count(OLD) != 1:
        raise ValueError("input is not exact #93 runtime source")
    candidate = data.replace(OLD, ROUTE, 1) + (HERE / "rounding.hepta").read_bytes()
    with tempfile.TemporaryDirectory(prefix="heptatorch-rounding-") as temp:
        path = Path(temp) / "runtime.hepta"
        path.write_bytes(candidate)
        command = [sys.executable, "-I", "-B", str(args.runtime_tools.resolve() / "partition.py"),
                   "--runtime", str(path), "--name", "typed-relu775-opt-in-rounding-f32",
                   "--prefix", "round782", "--output", str(args.output)]
        if args.expected_build:
            command += ["--expected-build", str(args.expected_build)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
