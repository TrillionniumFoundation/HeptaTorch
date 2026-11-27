#!/usr/bin/env python3
"""Project one exact, opt-in F32 square derivative over the #77 experiment."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PARENT = "5505bbf3a7d6462c70d02ba657567286028e501b73eb628d16ebb68f02dbaad5"


def replace_once(chunk, old, new):
    if chunk.count(old) != 1:
        raise ValueError("square source transition mismatch")
    return chunk.replace(old, new, 1)


def edit(source, name, changes):
    start = source.index(b"\nfn " + name + b"(") + 1
    end = source.index(b"\nend\n", start) + 5
    target = source[start:end]
    for old, new in changes:
        target = replace_once(target, old, new)
    return source[:start] + target + source[end:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-tools", type=Path, default=HERE.parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-build", type=Path)
    args = parser.parse_args()
    tools = args.runtime_tools.resolve()
    with tempfile.TemporaryDirectory(prefix="heptatorch-square-ordered-") as directory:
        temp = Path(directory)
        subprocess.run([sys.executable, "-I", "-B", str(HERE.parent / "opt_in_square_f32_ad/prepare.py"),
                        "--runtime-tools", str(tools), "--output", str(temp / "parent")], check=True)
        source = (temp / "parent/runtime.hepta").read_bytes()
        if hashlib.sha256(source).hexdigest() != PARENT:
            raise ValueError("not the exact #77 split-multiply source")
        source = edit(source, b"opt_in_square_f32", (
            (b"""  branch nogradr allocate(sr,xr,pr) grad_product(sr,xr)
block grad_product(sb:mutbuf,xb:u64)
  opb:u64 = const() {"value":3}
  yb:u64 = call(sb,opb,xb,xb) {"function":"binary"}
  ret(yb)
block unsupported""",
             b"""  rrzero:u64 = const() {"value":0}
  rrone:u64 = const() {"value":1}
  rrgrad:u64 = select(nogradr,rrzero,rrone)
  jump allocate(sr,xr,pr,rrgrad)
block unsupported"""),
            (b"block allocate(sa:mutbuf,xa:u64,pa:u64)", b"block allocate(sa:mutbuf,xa:u64,pa:u64,ga:u64)"),
            (b'call(sa,rank,shape,zeroa)', b'call(sa,rank,shape,ga)'),
            (b"start(sa,xa,pa,y)", b"start(sa,xa,pa,y,ga)"),
            (b"block start(ss:mutbuf,xs:u64,ps:u64,ys:u64)", b"block start(ss:mutbuf,xs:u64,ps:u64,ys:u64,gs:u64)"),
            (b"jump loop(ss,xs,ys,count,zeros)", b"jump loop(ss,xs,ys,count,zeros,gs)"),
            (b"block loop(sl:mutbuf,xl:u64,yl:u64,nl:u64,il:u64)", b"block loop(sl:mutbuf,xl:u64,yl:u64,nl:u64,il:u64,gl:u64)"),
            (b"element(sl,xl,yl,nl,il) done(yl)", b"element(sl,xl,yl,nl,il,gl) done(sl,xl,yl,gl)"),
            (b"block element(se:mutbuf,xe:u64,ye:u64,ne:u64,ie:u64)", b"block element(se:mutbuf,xe:u64,ye:u64,ne:u64,ie:u64,ge:u64)"),
            (b"jump loop(se,xe,ye,ne,next)\nblock done(yd:u64)\n  ret(yd)",
             b"""jump loop(se,xe,ye,ne,next,ge)
block done(sd:mutbuf,xd:u64,yd:u64,gd:u64)
  zd:u64 = const() {"value":0}
  ngd:bool = compare.eq(gd,zd)
  branch ngd plain(yd) record_square(sd,xd,yd,zd)
block plain(yp:u64)
  ret(yp)
block record_square(sr2:mutbuf,xr2:u64,yr2:u64,zr2:u64)
  op2:u64 = const() {"value":51}
  rr:u64 = call(sr2,yr2,op2,xr2,zr2) {"function":"record"}
  ret(rr)"""),
        ))
        # Keep the input version/storage check and ordinary saved-owner lifetime.
        source = edit(source, b"record_body", ((
            b"  v378:u64 = select(v376,v377,v363)",
            b"""  square_op:u64 = const() {"value":51}
  square_saves_input:bool = compare.eq(v357,square_op)
  save_input:bool = bool.or(v376,square_saves_input)
  v378:u64 = select(save_input,v377,v363)""",
        ),))
        source = edit(source, b"partial_apply_node", ((
            b"  branch v7 b1(v0,v1) b2(v0,v1,v5)",
            b"""  square_op:u64 = const() {"value":51}
  is_square:bool = compare.eq(v5,square_op)
  branch is_square square_backward(v0,v1) previous(v0,v1,v5,v7)
block square_backward(ss:mutbuf,ys:u64)
  rs:u64 = call(ss,ys) {"function":"opt_in_square_f32_backward"}
  ret(rs)
block previous(sp:mutbuf,yp:u64,op:u64,is_as_strided:bool)
  branch is_as_strided b1(sp,yp) b2(sp,yp,op)""",
        ),))
        # The one-to-one edge writes its first contribution, including -0,
        # rather than adding to a synthetic +0 scratch entry.
        source = edit(source, b"gradient_direct", ((
            b"  branch v11 b1(v0) b2(v0,v1,v2)",
            b"""  square_op:u64 = const() {"value":51}
  square_edge:bool = compare.eq(v6,square_op)
  direct_edge:bool = bool.or(v11,square_edge)
  branch direct_edge b1(v0) b2(v0,v1,v2)""",
        ),))
        # A derivative product is a new contiguous contribution, not an alias
        # of the incoming seed. This matters when two nodes feed one leaf.
        source = edit(source, b"to_original_incoming_describe", ((
            b"  branch v352 b30(v306) b31(v306)",
            b"""  square_op:u64 = const() {"value":51}
  square_contribution:bool = compare.eq(v311,square_op)
  owned_contribution:bool = bool.or(v352,square_contribution)
  branch owned_contribution b30(v306) b31(v306)""",
        ),))
        path = temp / "runtime.hepta"
        path.write_bytes(source + (HERE / "ordered.hepta").read_bytes())
        command = [sys.executable, "-I", "-B", str(tools / "partition.py"), "--runtime", str(path),
                   "--name", "typed-relu775-opt-in-square-f32-ordered-ad",
                   "--prefix", "squareordered776", "--output", str(args.output)]
        if args.expected_build:
            command += ["--expected-build", str(args.expected_build)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
