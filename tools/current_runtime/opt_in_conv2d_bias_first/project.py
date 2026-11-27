"""Add an explicit native F32 channel-bias-first accumulator experiment."""
import argparse
import hashlib
from pathlib import Path
import re

PARENT = "96ccc2c837b25a07acb2b66b089d4f55889c40a84c50dbf765eb887f6da9aee2"
PARTS = re.compile(rb"^fn (\w+)\([^\n]*\n.*?^end\n", re.M | re.S)
BLOCK = re.compile(rb"^block (\w+)\([^\n]*\n.*?(?=^block |^end\n)", re.M | re.S)


def one(raw, needle, replacement):
    if raw.count(needle) != 1:
        raise ValueError(f"exact source anchor not found: {needle!r}")
    return raw.replace(needle, replacement, 1)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def mapping(raw):
    spans=list(PARTS.finditer(raw))
    items={m[1]:m[0] for m in spans}
    if len(spans)!=len(items):
        raise ValueError("duplicate native definition")
    return items


def extra_arg(fn, labels, arg):
    """Thread one value through exactly the listed CFG blocks and their edges."""
    out=[]
    current=None
    for line in fn.splitlines(keepends=True):
        m=re.match(rb"(?:block|fn) (\w+)\(",line)
        if m:
            current=(m[1] if line.startswith(b"block ") else b"b0")
            if current in labels:
                q=line.index(b")")
                line=line[:q]+(b"," if not line.startswith(b"block "+current+b"()") else b"")+arg+b"_"+current+b":u64"+line[q:]
        elif current in labels:
            line=line.replace(b"$"+arg,b"%"+arg+b"_"+current)
            for target in labels:
                line=re.sub(rb"\b("+target+rb")\(([^\n()]*)\)",
                            lambda m:m[1]+b"("+m[2]+(b"," if m[2] else b"")+arg+b"_"+current+b")",line)
            line=line.replace(b"%",b"")
        out.append(line)
    return b"".join(out)


def project(parent):
    if sha(parent)!=PARENT:
        raise ValueError("unexpected exact post-sum parent")
    f=mapping(parent)
    cmd=f[b"command"]
    gate=b"  is_bias:bool = compare.eq(v6,bias_opcode)\n"
    cmd1=one(cmd,gate,gate+b"""  first_opcode:u64 = const() {"value":126}
  is_first:bool = compare.eq(v6,first_opcode)
  any_bias:bool = bool.or(is_bias,is_first)
""")
    cmd1=one(cmd1,b"  branch is_bias b132(",b"  branch any_bias b132(")
    old_call=b'  result:u64 = call(ba,xx,ww,bb,ss,pp,dd,gg) {"function":"conv_bias_forward"}\n'
    new_call=b"""  pre_code:u64 = const() {"value":126}
  do_pre:bool = compare.eq(bop,pre_code)
  branch do_pre b137(br,bw,ba,bo,bop,xx,ww,bb,ss,pp,dd,gg) b138(br,bw,ba,bo,bop,xx,ww,bb,ss,pp,dd,gg)
block b137(rr:readbuf,rw:mutbuf,ra:mutbuf,ro:u64,rp:u64,rx:u64,ry:u64,rb:u64,rs:u64,rpad:u64,rd:u64,rg:u64)
  pre_result:u64 = call(ra,rx,ry,rb,rs,rpad,rd,rg) {"function":"conv_bias_first"}
  jump b139(rr,rw,ra,ro,rp,pre_result)
block b138(sr:readbuf,sw:mutbuf,sa:mutbuf,so:u64,sp:u64,sx:u64,sy:u64,sb:u64,st:u64,spad:u64,sd:u64,sg:u64)
  post_result:u64 = call(sa,sx,sy,sb,st,spad,sd,sg) {"function":"conv_bias_forward"}
  jump b139(sr,sw,sa,so,sp,post_result)
block b139(br2:readbuf,bw2:mutbuf,ba2:mutbuf,bo2:u64,bop2:u64,result:u64)
"""
    before,after=cmd1.split(old_call,1)
    after=re.sub(rb"\b(br|bw|ba|bo|bop)\b",lambda m:m[0]+b"2",after)
    cmd1=before+new_call+after
    wrap=one(f[b"conv_bias_forward"],b"fn conv_bias_forward(",b"fn conv_bias_first(")
    wrap=wrap.split(b"  out:u64 = call(ca,cx,cw,cs,cp,cd,cg)",1)[0]+b"""  seeded_out:u64 = call(ca,cx,cw,cs,cp,cd,cg,cb) {"function":"conv_first_geometry"}
  ret(seeded_out)
end
"""
    geo=one(f[b"conv2d"],b"fn conv2d(",b"fn conv_first_geometry(")
    geo=one(geo,b'call(conv_arena,conv_out,conv_desc,conv_count,conv_input,conv_weight) {"function":"conv_direct_forward"}',
                 b'call(conv_arena,conv_out,conv_desc,conv_count,conv_input,conv_weight,$seed) {"function":"conv_first_forward"}')
    geo=extra_arg(geo,tuple(m[1] for m in BLOCK.finditer(geo)),b"seed")

    forward=one(f[b"conv_direct_forward"],b"fn conv_direct_forward(",b"fn conv_first_forward(")
    forward=one(forward,b'  v785:u64 = call(v744,v757,v760,v784,conv_x_b46,conv_w_b46) {"function":"conv_direct_point"}',
    b"""  channel_ptr_offset:u64 = const() {"value":104}
  channel_ptr:u64 = int.add.wrap(v758,channel_ptr_offset)
  output_channels:u64 = call(v744,channel_ptr) {"function":"getw"}
  hout_offset:u64 = const() {"value":112}
  wout_offset:u64 = const() {"value":120}
  hout_ptr:u64 = int.add.wrap(v758,hout_offset)
  wout_ptr:u64 = int.add.wrap(v758,wout_offset)
  hout:u64 = call(v744,hout_ptr) {"function":"getw"}
  wout:u64 = call(v744,wout_ptr) {"function":"getw"}
  hw:u64 = int.mul.wrap(hout,wout)
  nc:u64 = int.div.checked(v760,hw)
  channel:u64 = int.rem.checked(nc,output_channels)
  bias_value:u64 = call(v744,$seed,channel) {"function":"tget"}
  v785:u64 = call(v744,v757,v760,v784,conv_x_b46,conv_w_b46,bias_value) {"function":"conv_first_point"}""")
    forward=extra_arg(forward,(b"b0",b"b45",b"b46"),b"seed")

    point=one(f[b"conv_direct_point"],b"fn conv_direct_point(",b"fn conv_first_point(")
    point=one(point,b"v10,v11,v11,conv_x_b0",b"v10,v11,$seed,conv_x_b0")
    point=extra_arg(point,(b"b0",),b"seed")
    out=one(parent,cmd,cmd1)+wrap+geo+forward+point
    out_f=mapping(out)
    if list(out_f)!=[*f,b"conv_bias_first",b"conv_first_geometry",b"conv_first_forward",b"conv_first_point"]:
        raise ValueError("unexpected function set")
    if [k for k in f if f[k]!=out_f[k]]!=[b"command"]:
        raise ValueError("outside native scope")
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("parent","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    output=project(args.parent.read_bytes())
    with args.output.open("xb") as stream:
        stream.write(output)
    print(sha(output))
