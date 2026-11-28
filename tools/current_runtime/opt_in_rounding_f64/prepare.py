#!/usr/bin/env python3
"""Add F64 rounding dispatch to the exact #97 source; do not edit F32."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PARENT_SOURCE = "32f1d664826a1ccf524d8fc049889f439f164c7c6cb6a28eae3054691af97d82"
OLD = b"  branch f32_ok_rnd rounding_native(sdr,odr,xdr) rounding_refuse(sdr)\n"
ROUTE = b'''  branch f32_ok_rnd rounding_native(sdr,odr,xdr) rounding_f64_dtype(sdr,odr,xdr)
block rounding_f64_dtype(sd64:mutbuf,od64:u64,xd64:u64)
  dt64:u64 = call(sd64,xd64) {"function":"tensor_dtype"}
  f64_round:u64 = const() {"value":4}
  f64_good:bool = compare.eq(dt64,f64_round)
  branch f64_good rounding_f64_native(sd64,od64,xd64) rounding_refuse(sd64)
block rounding_f64_native(sn64:mutbuf,on64:u64,xn64:u64)
  yn64:u64 = call(sn64,xn64,on64) {"function":"opt_in_round_f64"}
  ret(yn64)
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
        raise ValueError("input is not exact #97 runtime source")
    candidate = data.replace(OLD, ROUTE, 1) + (HERE / "rounding.hepta").read_bytes()
    with tempfile.TemporaryDirectory(prefix="heptatorch-rounding-") as temp:
        path = Path(temp) / "runtime.hepta"
        path.write_bytes(candidate)
        command = [sys.executable, "-I", "-B", str(args.runtime_tools.resolve() / "partition.py"),
                   "--runtime", str(path), "--name", "typed-relu775-opt-in-rounding-f32-f64",
                   "--prefix", "round784", "--output", str(args.output)]
        if args.expected_build:
            command += ["--expected-build", str(args.expected_build)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
