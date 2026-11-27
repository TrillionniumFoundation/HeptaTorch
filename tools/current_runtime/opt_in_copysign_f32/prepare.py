#!/usr/bin/env python3
"""Materialize a bounded opt-in copysign runtime from the exact main source."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
MAIN = "0b000b51288af20490162efd8e525b86359a32cdd6d2521df97ecd9cddbd7b43"
OLD = b'  branch failed2 failure() route(s2,o2,l2,r2)\n'
INSERT = b'''  branch failed2 failure() copysign_route(s2,o2,l2,r2)
block copysign_route(sc:mutbuf,oc:u64,lc:u64,rc:u64)
  opcode:u64 = const() {"value":16}
  match:bool = compare.eq(oc,opcode)
  branch match copysign_native(sc,lc,rc) route(sc,oc,lc,rc)
block copysign_native(sn:mutbuf,ln:u64,rn:u64)
  resultn:u64 = call(sn,ln,rn) {"function":"opt_in_copysign_f32"}
  ret(resultn)
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
    begin = source.index(b"\nfn binary(") + 1
    end = source.index(b"\nend\n", begin) + len(b"\nend\n")
    binary = source[begin:end]
    helper = (HERE / "copysign.hepta").read_bytes()
    if binary.count(OLD) != 1 or not helper.startswith(b"fn opt_in_copysign_f32(") or not helper.endswith(b"end\n"):
        raise ValueError("source transition or copysign definition malformed")
    candidate = source[:begin] + binary.replace(OLD, INSERT, 1) + source[end:] + helper
    with tempfile.TemporaryDirectory(prefix="heptatorch-copysign-projection-") as temp:
        path = Path(temp) / "runtime.hepta"
        path.write_bytes(candidate)
        command = [sys.executable, "-I", "-B", str(tools / "partition.py"),
                   "--runtime", str(path), "--name", "typed-relu775-opt-in-copysign-f32",
                   "--prefix", "copysign776", "--output", str(args.output)]
        if args.expected_build:
            command += ["--expected-build", str(args.expected_build)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
