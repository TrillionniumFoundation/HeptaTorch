"""Add a separate opt-in F32 no-grad conv2d-with-bias wire operation."""
import argparse
import hashlib
from pathlib import Path
import re

PARENT = "115d01ac9377070f6841d619c1d9adf4a52cba809f676eb8b1522adefe88fcf1"
PARTS = rb"^fn (\w+)\([^\n]*\n.*?^end\n"
ENTRY = b'  v7:u64 = const() {"value":41}\n'
ROUTE = b"""  bias_opcode:u64 = const() {"value":125}
  is_bias:bool = compare.eq(v6,bias_opcode)
  branch is_bias b132(v0,v1,v2,v3,v4,v6) b135(v0,v1,v2,v3,v4,v5,v6)
block b135(v0b:readbuf,v1b:mutbuf,v2b:mutbuf,v3b:u64,v4b:u64,v5b:u64,v6b:u64)
"""
TAIL = b"""block b132(br:readbuf,bw:mutbuf,ba:mutbuf,bc:u64,bo:u64,bop:u64)
  snap_off:u64 = const() {"value":88}
  count_off:u64 = const() {"value":8}
  snap:u64 = call(ba,count_off) {"function":"getw"}
  store_snap:u64 = call(ba,snap_off,snap) {"function":"putw"}
  arg1:u64 = int.add.wrap(bc,count_off)
  arg2_off:u64 = const() {"value":16}
  arg2:u64 = int.add.wrap(bc,arg2_off)
  arg3_off:u64 = const() {"value":24}
  arg3:u64 = int.add.wrap(bc,arg3_off)
  arg4_off:u64 = const() {"value":32}
  arg4:u64 = int.add.wrap(bc,arg4_off)
  arg5_off:u64 = const() {"value":40}
  arg5:u64 = int.add.wrap(bc,arg5_off)
  arg6_off:u64 = const() {"value":48}
  arg6:u64 = int.add.wrap(bc,arg6_off)
  arg7_off:u64 = const() {"value":56}
  arg7:u64 = int.add.wrap(bc,arg7_off)
  xx:u64 = call(br,arg1) {"function":"getr"}
  ww:u64 = call(br,arg2) {"function":"getr"}
  bb:u64 = call(br,arg3) {"function":"getr"}
  ss:u64 = call(br,arg4) {"function":"getr"}
  pp:u64 = call(br,arg5) {"function":"getr"}
  dd:u64 = call(br,arg6) {"function":"getr"}
  gg:u64 = call(br,arg7) {"function":"getr"}
  result:u64 = call(ba,xx,ww,bb,ss,pp,dd,gg) {"function":"conv_bias_forward"}
  z:u64 = const() {"value":0}
  error:u64 = call(ba,z) {"function":"getw"}
  ok:bool = compare.eq(error,z)
  branch ok b122(br,bw,ba,bo,bop,result,z) b128(br,bw,ba,bo,bop,z,z)
"""


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def mapping(raw):
    hits = list(re.finditer(PARTS, raw, re.M | re.S))
    out = {m[1]: m[0] for m in hits}
    if len(out) != len(hits):
        raise ValueError("duplicate native function")
    return out


def project(parent, bias):
    if sha(parent) != PARENT:
        raise ValueError("unexpected exact native conv2d parent")
    old = mapping(parent)
    cmd = old[b"command"]
    begin, former = cmd.split(ENTRY, 1)
    first, rest = former.split(b"block b1(", 1)
    first = re.sub(rb"\bv[0-6]\b", lambda m: m[0]+b"b", ENTRY+first)
    rewritten = begin + ROUTE + first + b"block b1(" + rest[:-4] + TAIL + b"end\n"
    result = parent.replace(cmd, rewritten, 1) + bias
    fresh = mapping(result)
    if (list(fresh) != [*old, b"conv_bias_forward"]
        or [k for k in old if old[k] != fresh[k]] != [b"command"]):
        raise ValueError("unexpected native scope")
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--parent", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    result = project(args.parent.read_bytes(), Path(__file__).with_name("conv_bias.hepta").read_bytes())
    with args.output.open("xb") as handle:
        handle.write(result)
    print(sha(result))
