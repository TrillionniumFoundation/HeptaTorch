"""Project pinned square and unsqueeze HIA functions over exact #94 source."""
import argparse
import hashlib
from pathlib import Path
import re

PARENT = "318508540c44cdc60c18cb3a876c13b06a6cb60a34153ad4be621e1a964df4f8"
OVERLAY = "eed5ed4e29e816f73f9d0c740fd69c0cb66f7579f90679e54e4267788329b62b"
SOURCE = "f58637bb9eda972f9a335c26f286538211390ea8b09a63eaf66c2e508e48f91e"
FUNCTION = re.compile(rb"^fn (\w+)\([^\n]*?\n.*?^end\n", re.M | re.S)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def functions(data):
    matches = [(m[1], m[0]) for m in FUNCTION.finditer(data)]
    result = dict(matches)
    if len(result) != len(matches):
        raise ValueError("duplicate HIA function name")
    return result


def reclaim_scratch(image):
    """Do not retain the geometry Hold on the successful graph-free route."""
    for name, tail, block_tail in (
        (b"conv2d", b"", b""),
        (b"conv_first_geometry", b",seed_b44", b",seed_b48:u64"),
    ):
        method = functions(image)[name]
        where = b"conv_noad b48(v701,v714,v715,v716,v702,v703" + tail + b")"
        block = b"conv_weight:u64" + block_tail + b")\n  conv_result:u64"
        done = b"  ret(conv_result)\nend\n"
        restore = b'''  conv_hold_at:u64 = const() {"value":176}
  conv_hold_field:u64 = int.add.wrap(conv_desc,conv_hold_at)
  conv_hold_clear:u64 = const() {"value":0}
  conv_clear:u64 = call(conv_arena,conv_hold_field,conv_hold_clear) {"function":"putw"}
  conv_released:u64 = call(conv_arena,conv_hold) {"function":"hold_release"}
  ret(conv_result)
end
'''
        revised = method
        for old, new in ((where, where[:-1] + b",v708)"),
                         (block, block.replace(b")\n", b",conv_hold:u64)\n", 1)),
                         (done, restore)):
            if revised.count(old) != 1:
                raise ValueError(f"ambiguous scratch cleanup anchor for {name!r}")
            revised = revised.replace(old, new, 1)
        image = image.replace(method, revised, 1)
    return image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    parent = a.parent.read_bytes()
    overlay = Path(__file__).with_name("overlay.hepta").read_bytes()
    if digest(parent) != PARENT or digest(overlay) != OVERLAY:
        raise ValueError("wrong spatial source or HIA overlay")
    known = functions(parent)
    changed = functions(overlay)
    if b"".join(changed.values()) != overlay or len(changed) != 13:
        raise ValueError("HIA overlay does not contain the exact 13 functions")
    image = parent
    for name, new in changed.items():
        old = known.get(name)
        if old is None:
            image += new
        else:
            if image.count(old) != 1:
                raise ValueError(f"ambiguous HIA function {name!r}")
            image = image.replace(old, new, 1)
    image = reclaim_scratch(image)
    if digest(image) != SOURCE:
        raise ValueError("unexpected projected HIA source")
    with a.output.open("xb") as output:
        output.write(image)
    print(SOURCE)


if __name__ == "__main__":
    main()
