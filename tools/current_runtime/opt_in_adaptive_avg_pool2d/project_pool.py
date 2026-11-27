"""Construct the isolated F32 spatial candidate from a pinned native source."""
import argparse
import hashlib
from pathlib import Path
import re

PARENT = "0b000b51288af20490162efd8e525b86359a32cdd6d2521df97ecd9cddbd7b43"
HELPERS = "d66e57274addc757017fe6c3f8c48c441f3c879be878c4574245df642532bb94"
RESULT = "0439b0afaece473530051c168f179ecc8b27c6954a00df1071517d4ec73a39bf"
PARTS = rb"^fn (\w+)\([^\n]*\n.*?^end\n"
BRANCH = b"  jump b40(v633,v642,v646,v647,v656)\n"
NO_GRAD_BRANCH = b"""  v755:u64 = const() {"value":32}
  v756:u64 = int.add.wrap(v642,v755)
  v757:u64 = call(v633,v756) {"function":"getw"}
  v758:bool = compare.eq(v757,v656)
  avg_mode:u64 = const() {"value":23}
  avg_only:bool = compare.eq(v634,avg_mode)
  avg_nograd:bool = bool.and(v758,avg_only)
  branch avg_nograd b45(v633,v646,v635,v634) b40(v633,v642,v646,v647,v656)
"""
TAIL = b"""block b45(v759:mutbuf,v760:u64,v761:u64,v762:u64)
  v763:u64 = call(v759,v760,v761,v762) {"function":"pool_no_grad_forward"}
  ret(v763)
"""


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def mapping(raw):
    rows = list(re.finditer(PARTS, raw, re.M|re.S))
    result = {m[1]:m[0] for m in rows}
    if len(rows) != len(result):
        raise ValueError("duplicate HIA function")
    return result


def project(parent, helpers):
    if sha(parent) != PARENT:
        raise ValueError("unexpected canonical typed-relu775 parent")
    if sha(helpers) != HELPERS:
        raise ValueError("unexpected native no-grad helpers")
    old = mapping(parent)
    extra = mapping(helpers)
    if (list(extra) != [b"pool_point_forward", b"pool_no_grad_forward"]
            or set(extra) & set(old) or helpers != b"".join(extra.values())):
        raise ValueError("unexpected HIA helper set")
    pool = old[b"pool2d"]
    if pool.count(BRANCH) != 1 or not pool.endswith(b"end\n") or b"\nblock b45(" in pool:
        raise ValueError("native route anchor mismatch")
    new_pool = pool[:-4].replace(BRANCH, NO_GRAD_BRANCH, 1) + TAIL + b"end\n"
    image = parent.replace(pool, new_pool, 1) + helpers
    new = mapping(image)
    if (list(new) != [*old, *extra]
            or [key for key in old if old[key] != new[key]] != [b"pool2d"]
            or sha(image) != RESULT
            or image[:-len(helpers)].replace(new_pool, pool, 1) != parent):
        raise ValueError("unexpected changes outside native pooling route")
    return image


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    result = project(args.parent.read_bytes(), Path(__file__).with_name("pool_no_grad.hepta").read_bytes())
    with args.output.open("xb") as f:
        f.write(result)
    print(sha(result))
