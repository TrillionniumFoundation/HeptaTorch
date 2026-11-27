"""Construct the opt-in no-grad conv2d image from the exact avg-pool projection."""
import argparse
import hashlib
from pathlib import Path
import re

PARENT = "0439b0afaece473530051c168f179ecc8b27c6954a00df1071517d4ec73a39bf"
RESULT = "115d01ac9377070f6841d619c1d9adf4a52cba809f676eb8b1522adefe88fcf1"
PARTS = rb"^fn (\w+)\([^\n]*\n.*?^end\n"
ANCHOR = b"""  v722:u64 = call(v701,v721,v708) {"function":"putw"}
"""
ROUTE = b"""  conv_ad_off:u64 = const() {"value":32}
  conv_ad_field:u64 = int.add.wrap(v719,conv_ad_off)
  conv_ad:u64 = call(v701,conv_ad_field) {"function":"getw"}
  conv_zero:u64 = const() {"value":0}
  conv_noad:bool = compare.eq(conv_ad,conv_zero)
  branch conv_noad b48(v701,v714,v715,v716,v702,v703) b49(v701,v702,v703,v708,v714,v715,v716,v721)
block b49(conv_arena2:mutbuf,conv_x2:u64,conv_w2:u64,conv_groups2:u64,conv_out2:u64,conv_desc2:u64,conv_count2:u64,conv_groups_field2:u64)
"""
TAIL = b"""block b48(conv_arena:mutbuf,conv_out:u64,conv_desc:u64,conv_count:u64,conv_input:u64,conv_weight:u64)
  conv_result:u64 = call(conv_arena,conv_out,conv_desc,conv_count,conv_input,conv_weight) {"function":"conv_direct_forward"}
  ret(conv_result)
"""
BLOCK = re.compile(rb"^block (b\d+)\([^\n]*\n.*?(?=^block |^end\n)", re.M | re.S)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def one_replace(raw, old, new):
    if raw.count(old) != 1:
        raise ValueError(f"native projection anchor mismatch: {old!r}")
    return raw.replace(old, new, 1)


def mapping(raw):
    rows = list(re.finditer(PARTS, raw, re.M | re.S))
    out = {m[1]: m[0] for m in rows}
    if len(rows) != len(out):
        raise ValueError("duplicate native function")
    return out


def extend_blocks(raw, labels):
    """Pass original operands along live forward edges, not via AD descriptor slots."""
    current = None
    lines = []
    for line in raw.splitlines(keepends=True):
        block = re.match(rb"block (b\d+)\(", line)
        if block:
            current = block[1]
            if current in labels:
                line = line.rstrip(b")\n") + b",conv_x_" + current + b":u64,conv_w_" + current + b":u64)\n"
        elif current in labels:
            named = (b"conv_x_" + current, b"conv_w_" + current)
            line = re.sub(rb"\b(conv_x|conv_w)\b", lambda m: named[m[0] == b"conv_w"], line)
            for label in labels:
                line = re.sub(rb"(\b" + label + rb"\()([^\n()]*)(\))",
                              rb"\1\2," + b",".join(named) + rb"\3", line)
        lines.append(line)
    return b"".join(lines)


def direct_term(term):
    # Keep the parent's exact bounded geometry, grouped coordinates and arithmetic.
    blocks = {m[1]: m[0] for m in BLOCK.finditer(term)}
    keep = b"".join(blocks[k] for k in (b"b0", b"b1", b"b2", b"b3", b"b4", b"b5", b"b6"))
    # The shared term's forward-only multiply used AD operand slots even though
    # a graph-free descriptor intentionally never stores edges. Take handles
    # directly instead, and leave all training / backward blocks untouched.
    start = keep.index(b"  v227:u64")
    keep = keep[:start] + b"""  conv_a:u64 = call(v167,conv_x,v221) {"function":"tget"}
  conv_b:u64 = call(v167,conv_w,v226) {"function":"tget"}
  conv_product:u64 = call(conv_a,conv_b) {"function":"mulf"}
  ret(conv_product)
"""
    keep = extend_blocks(keep, (b"b0", b"b2", b"b3", b"b4", b"b6"))
    return (b"fn conv_direct_term(v0:mutbuf,v1:u64,v2:u64,v3:u64,v4:u64,conv_x_b0:u64,conv_w_b0:u64)"
            b" -> (u64) effects(read,write,trap) entry(b0)\n" + keep + b"end\n")


def direct_point(point):
    out = one_replace(point, b"fn conv_point(v0:mutbuf,v1:u64,v2:u64,v3:u64)",
                      b"fn conv_direct_point(v0:mutbuf,v1:u64,v2:u64,v3:u64,conv_x_b0:u64,conv_w_b0:u64)")
    out = one_replace(out, b'call(v21,v22,v23,v27,v24) {"function":"conv_term"}',
                      b'call(v21,v22,v23,v27,v24,conv_x,conv_w) {"function":"conv_direct_term"}')
    return extend_blocks(out, (b"b0", b"b1", b"b2", b"b4", b"b6"))


def direct_forward(conv):
    blocks = {m[1]: m[0] for m in BLOCK.finditer(conv)}
    out = b"".join(blocks[k] for k in (b"b45", b"b46", b"b47"))
    out = one_replace(out, b'call(v744,v757,v760,v784) {"function":"conv_point"}',
                      b'call(v744,v757,v760,v784,conv_x,conv_w) {"function":"conv_direct_point"}')
    return (b"fn conv_direct_forward(area:mutbuf,output:u64,desc:u64,count:u64,conv_x:u64,conv_w:u64)"
            b" -> (u64) effects(read,write,trap) entry(b0)\n"
            b"block b0(area:mutbuf,output:u64,desc:u64,count:u64,conv_x:u64,conv_w:u64)\n"
            b'  conv_zero:u64 = const() {"value":0}\n'
            b"  jump b45(area,output,desc,count,conv_zero,conv_x,conv_w)\n"
            + extend_blocks(out, (b"b45", b"b46")) + b"end\n")


def project(parent):
    if sha(parent) != PARENT:
        raise ValueError("unexpected exact native average-pool parent")
    old = mapping(parent)
    conv = old[b"conv2d"]
    training, rest = conv.split(ANCHOR, 1)
    old_training, tail = rest.split(b"block b45(", 1)
    training_aliases = dict(zip(
        map(str.encode, ("v701", "v702", "v703", "v708", "v714", "v715", "v716", "v721")),
        map(str.encode, ("conv_arena2", "conv_x2", "conv_w2", "conv_groups2", "conv_out2",
                         "conv_desc2", "conv_count2", "conv_groups_field2"))))
    training_body = re.sub(rb"\bv\d+\b", lambda m: training_aliases.get(m[0], m[0]),
                           ANCHOR + old_training)
    new_conv = training + ROUTE + training_body + b"block b45(" + tail[:-4] + TAIL + b"end\n"
    extras = direct_term(old[b"conv_term"]) + direct_point(old[b"conv_point"]) + direct_forward(conv)
    image = one_replace(parent, conv, new_conv) + extras
    new = mapping(image)
    if (list(new) != [*old, b"conv_direct_term", b"conv_direct_point", b"conv_direct_forward"]
            or [key for key in old if old[key] != new[key]] != [b"conv2d"]
            or sha(image) != RESULT
            or one_replace(image[:-len(extras)], new_conv, conv) != parent):
        raise ValueError("unexpected HIA scope change")
    return image


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--parent", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    result = project(args.parent.read_bytes())
    with args.output.open("xb") as f:
        f.write(result)
    print(sha(result))
