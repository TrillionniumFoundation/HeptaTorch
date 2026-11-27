#!/usr/bin/env python3
"""Project an isolated native unsqueeze ABI experiment from pinned canonical source."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
SOURCE = "0b000b51288af20490162efd8e525b86359a32cdd6d2521df97ecd9cddbd7b43"
OLD = b"  branch v144 b13(v119,v120,v121,v122,v124,v125,v126,v127,v128,v139) b14(v119,v120,v121,v122,v124,v125,v126,v127,v128,v139)"
NEW = b'''  abi_op:u64 = const() {"value":129}
  abi_match:bool = compare.eq(v139,abi_op)
  abi_route:bool = bool.or(v144,abi_match)
  branch abi_route b13(v119,v120,v121,v122,v124,v125,v126,v127,v128,v139) b14(v119,v120,v121,v122,v124,v125,v126,v127,v128,v139)'''
ROUTE = b"  branch v11 b1(v0,v1,v2,v3,v4,v5,v6) b2(v0,v1,v2,v3,v4,v5)"
ABI = b'''  abi:u64 = const() {"value":129}
  match:bool = compare.eq(v6,abi)
  branch match abi_command(v0,v1,v2,v3,v4,v6) previous(v0,v1,v2,v3,v4,v5,v6,v11)
block abi_command(ar:readbuf,ao:mutbuf,as0:mutbuf,ai:u64,ac:u64,ak:u64)
  aa:u64 = call(ar,ao,as0,ai,ac,ak) {"function":"opt_in_unsqueeze_command"}
  ret(aa)
block previous(pr:readbuf,po:mutbuf,ps:mutbuf,pi:u64,pc:u64,pn:u64,pk:u64,pb:bool)
  branch pb b1(pr,po,ps,pi,pc,pn,pk) b2(pr,po,ps,pi,pc,pn)'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runtime-tools", type=Path, default=HERE.parent)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    tools = args.runtime_tools.resolve()
    source = (tools / "source/runtime.hepta").read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE or source.count(OLD) != 1:
        raise ValueError("canonical typed-ReLU 775 source mismatch")
    source = source.replace(OLD, NEW, 1)
    begin = source.index(b"\nfn predicate_command(")
    end = source.index(b"\nend\n", begin) + len(b"\nend\n")
    block = source[begin:end]
    if block.count(ROUTE) != 1:
        raise ValueError("unexpected predicate-command source")
    candidate = source[:begin] + block.replace(ROUTE, ABI, 1) + source[end:] + (HERE / "unsqueeze.hepta").read_bytes()
    with tempfile.TemporaryDirectory(prefix="heptatorch-unsqueeze-") as directory:
        path = Path(directory) / "runtime.hepta"
        path.write_bytes(candidate)
        subprocess.run([sys.executable, "-I", "-B", str(tools / "partition.py"), "--runtime", str(path),
                        "--name", "typed-relu775-opt-in-unsqueeze-f32", "--prefix", "unsqueeze778",
                        "--output", str(args.output)], check=True)


if __name__ == "__main__":
    main()
