#!/usr/bin/env python3
"""Materialize the F64 successor from the exact F32 copysign aggregate."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
F32_SHA = "a59cd1a2aa892f2e49673a1e9e5d2f35969aef563ddbd6a186519fdab1b8efd1"
OLD = b'''block copysign_native(sn:mutbuf,ln:u64,rn:u64)
  resultn:u64 = call(sn,ln,rn) {"function":"opt_in_copysign_f32"}
  ret(resultn)
'''
NEW = b'''block copysign_native(sn:mutbuf,ln:u64,rn:u64)
  dn:u64 = call(sn,ln) {"function":"tensor_dtype"}
  dtype_n:u64 = const() {"value":4}
  f64_n:bool = compare.eq(dn,dtype_n)
  branch f64_n copysign64_only(sn,ln,rn) copysign32_only(sn,ln,rn)
block copysign32_only(s32:mutbuf,l32:u64,r32:u64)
  result32:u64 = call(s32,l32,r32) {"function":"opt_in_copysign_f32"}
  ret(result32)
block copysign64_only(s64:mutbuf,l64:u64,r64:u64)
  result64:u64 = call(s64,l64,r64) {"function":"opt_in_copysign_f64"}
  ret(result64)
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runtime-tools", type=Path, default=HERE.parent)
    ap.add_argument("--f32-source", required=True, type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expected-build", type=Path)
    args = ap.parse_args()
    raw = args.f32_source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != F32_SHA or raw.count(OLD) != 1:
        raise ValueError("not the exact F32 copysign projection")
    with tempfile.TemporaryDirectory(prefix="heptatorch-copysign-f64-") as temp:
        path = Path(temp) / "runtime.hepta"
        path.write_bytes(raw.replace(OLD, NEW, 1) + (HERE/"copysign.hepta").read_bytes())
        command = [sys.executable, "-I", "-B", str(args.runtime_tools.resolve()/"partition.py"),
                   "--runtime", str(path), "--name", "typed-relu775-opt-in-copysign-f32-f64",
                   "--prefix", "copysign777", "--output", str(args.output)]
        if args.expected_build:
            command += ["--expected-build", str(args.expected_build)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
