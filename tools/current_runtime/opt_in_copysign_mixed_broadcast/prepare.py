#!/usr/bin/env python3
"""Project mixed dimensioned F32/F64 copysign onto the exact #99 body tree."""
import argparse
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PARENT = "d55e48fc16404a5a0665411c2a004bfeb7411db5bf480405d23650eac107dd22"
OLD = b'  branch f64_n copysign64_only(sn,ln,rn) copysign32_only(sn,ln,rn)\n'
NEW = b'''  rd_n:u64 = call(sn,rn) {"function":"tensor_dtype"}
  zero_n:u64 = const() {"value":0}
  x32_n:bool = compare.eq(dn,zero_n)
  z32_n:bool = compare.eq(rd_n,zero_n)
  z64_n:bool = compare.eq(rd_n,dtype_n)
  x64_mix:bool = bool.and(f64_n,z32_n)
  z64_mix:bool = bool.and(x32_n,z64_n)
  mixed_n:bool = bool.or(x64_mix,z64_mix)
  branch mixed_n copysign_mixed_native(sn,ln,rn) copysign_old(sn,ln,rn,f64_n)
block copysign_mixed_native(smix:mutbuf,lmix:u64,rmix:u64)
  ymix:u64 = call(smix,lmix,rmix) {"function":"opt_in_copysign_mixed"}
  ret(ymix)
block copysign_old(so:mutbuf,lo:u64,ro:u64,is64:bool)
  branch is64 copysign64_only(so,lo,ro) copysign32_only(so,lo,ro)
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-tools", type=Path, default=HERE.parent)
    p.add_argument("--parent-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-build", type=Path)
    a = p.parse_args()
    parent = a.parent_source.read_bytes()
    new = (HERE/"copysign.hepta").read_bytes()
    if (sha256(parent).hexdigest() != PARENT or parent.count(OLD) != 1
            or not new.startswith(b"fn opt_in_copysign_mixed(") or not new.endswith(b"end\n")):
        raise ValueError("invalid parent or mixed extension")
    with tempfile.TemporaryDirectory(prefix="heptatorch-copysign-mixed-") as scratch:
        path = Path(scratch)/"runtime.hepta"
        path.write_bytes(parent.replace(OLD, NEW, 1) + new)
        command = [sys.executable, "-I", "-B", str(a.runtime_tools.resolve()/"partition.py"),
                   "--runtime", str(path), "--name", "typed-relu775-opt-in-mixed-copysign-broadcast",
                   "--prefix", "copysign778", "--output", str(a.output)]
        if a.expected_build:
            command.extend(["--expected-build", str(a.expected_build)])
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
