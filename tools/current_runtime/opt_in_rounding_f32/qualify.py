#!/usr/bin/env python3
"""Execute ordinary F32 trunc, floor and ceil requests on independent PyTorch and native images."""
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import random
import runpy
import struct
import sys


PARENT = "5cd57293d53ec23d8207603729516fa98d84e0bb9e2d140a2effbb50ab7d547f"
CANDIDATE = "c5f542e32675e3e058ca409a5bda57f092f344b8d0a58903797731937f6652c6"
OPERATIONS = {19: "trunc", 20: "floor", 21: "ceil"}
WORDS = (0, 0x80000000, 1, 0x807FFFFF, 0x00800000, 0x80000001, 0x007FFFFF, 0x80800000, 0x7FBFFFFF, 0xFFBFFFFF, 0xFFFFFFFF, 0x217FFFFF, 0x21800000,
         0x21800001, 0x26000000, 0x3F7FFFFF, 0x3F800001, 0x3FC00000, 0xC0200000,
         0x5F7FFFFF, 0x5F800000, 0x7F7FFFFF, 0x7F800000, 0xFF800000,
         0x7F812345, 0xFF812345, 0x7FC76543, 0xFFC76543)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def raw(words):
    return struct.pack("<" + "I" * len(words), *words)


def cases():
    yield from ((((), (word,)) for word in WORDS))
    shapes = [(0,), (2, 0, 3), (0, 2, 3, 4), (1,), (7,), (8,), (9,), (31,), (32,),
              (33,), (257,), (2, 3), (2, 1, 17), (2, 3, 2, 7), (1, 1, 1, 4096)]
    rng = random.Random(77518)
    for shape in shapes:
        n = math.prod(shape)
        sample = list(itertools.islice(itertools.cycle(WORDS), n))
        for i in range(len(WORDS), n, 2):
            sample[i] = rng.getrandbits(32)
        yield shape, tuple(sample)
    for exponent in range(126, 152):
        pivot = exponent << 23
        for magnitude in (pivot - 1, pivot, pivot + 1, pivot + 0x400000, pivot + 0x7FFFFF):
            yield (), (magnitude,)
            yield (), (magnitude | 0x80000000,)
    threshold_words = tuple(bits | sign for sign in (0, 0x80000000)
                            for boundary in (1, 0x3F800000, 0x40000000, 0x4A800000, 0x4B000000)
                            for bits in range(max(0, boundary - 64), boundary + 64))
    yield (len(threshold_words),), threshold_words
    for begin, end in ((0, 1024), (0x007FFE00, 0x00800200), (0x7F7FFE00, 0x7F800200),
                       (0x7FBFFE00, 0x7FC00200), (0x7FFFFC00, 0x80000000)):
        yield (2, 1024), tuple(value | sign for sign in (0, 0x80000000) for value in range(begin, end))


def tensor(wire, shape, words, requires=0):
    return (1, len(shape), requires, *shape, *([0] * (4 - len(shape))),
            wire.Payload(raw(words), 4))


def observe_torch(repo, output, inputs, operation):
    os.chdir(repo)
    os.environ["ATEN_CPU_CAPABILITY"] = "avx2"
    saved_argv = sys.argv
    try:
        sys.argv = [str(repo / "tests/pytorch_oracle.py"), str(output / "NO_NATIVE"),
                    str(output / "oracle-error.json"), str(repo / "compatibility/pytorch-oracle-lock.json")]
        ctx = runpy.run_path(sys.argv[0], run_name="heptatorch_rounding_reference_only")
    finally:
        sys.argv = saved_argv
    assert ctx["PROVENANCE"]["status"] == "PASS"
    torch = ctx["torch"]
    assert torch.backends.cpu.get_cpu_capability() == "AVX2"
    assert torch.get_num_threads() == 1
    out = []
    for shape, words in inputs:
        bits = raw(words)
        seed = torch.frombuffer(bytearray(bits), dtype=torch.float32) if bits else torch.empty(0, dtype=torch.float32)
        x = seed.reshape(shape).clone()
        y = getattr(torch.ops.aten, OPERATIONS[operation]).default(x)
        output_raw = bytes(y.untyped_storage())
        assert tuple(y.shape) == shape and y.dtype == x.dtype and not y.requires_grad
        assert y.storage_offset() == y._version == 0
        assert y.untyped_storage()._cdata != x.untyped_storage()._cdata
        assert bytes(x.untyped_storage()) == bits
        out.append((tuple(y.stride()), output_raw))
    return out, ctx["PROVENANCE"]


def physical(session, decoder, token, meta):
    owner = decoder(session.owner()["bytes"])
    node = owner.node(token)
    storage, version = owner.row(2, node[27]), owner.row(3, node[28])
    assert (node[27], node[28], node[29], storage[3], version[2]) == (
        meta["storage_id"], meta["version_id"], meta["storage_offset"], meta["storage_bytes"], meta["version"])
    assert node[72] == 0 and all(node[i] == 0 for i in (4, 5, 6, 7, 45, 48, 49, 50, 51, 83))
    start = owner.start + storage[2]
    value = owner.raw[start:start + storage[3]]
    assert len(value) == storage[3]
    return value


def run_positive(wire, decoder, session, shape, words, stride, expected, operation):
    (x,) = session.run([tensor(wire, shape, words)])
    x = x["value"]
    count = len(words)
    xmet, xbytes = session.run([(13, x), (39, x, 0, count)])
    assert xbytes["payload"] == raw(words)
    old = wire.meta(xmet)
    assert physical(session, decoder, x, old) == raw(words)
    (result_row,) = session.run([(3, operation, x)])
    y = result_row["value"]
    ymet, got, xmet2, xbytes2 = session.run([(13, y), (39, y, 0, count), (13, x), (39, x, 0, count)])
    result = wire.meta(ymet)
    assert y != x and result["storage_id"] != old["storage_id"] and result["version_id"] != old["version_id"]
    assert result["rank"] == len(shape) and tuple(result["shape"][:len(shape)]) == shape
    assert tuple(result["strides"][:len(shape)]) == stride and result["numel"] == count
    assert result["storage_offset"] == result["requires_grad"] == result["version"] == 0
    assert result["storage_bytes"] == 4 * count
    assert got["value"] == count and got["payload"] == expected, (shape, got["payload"][:64].hex(), expected[:64].hex())
    assert wire.meta(xmet2) == old and xbytes2["payload"] == xbytes["payload"]
    assert physical(session, decoder, x, old) == raw(words)
    assert physical(session, decoder, y, result) == expected
    session.run([(41, x)])
    (owned,) = session.run([(39, y, 0, count)])
    assert owned["payload"] == expected
    session.run([(41, y)])
    return {"op": OPERATIONS[operation], "shape": shape, "elements": count, "raw_input_sha256": sha(raw(words)),
            "raw_result_sha256": sha(expected), "shape_stride_exact": True, "fresh_ownership": True,
            "survives_release": True, "input_unchanged": True, "physical_storage_exact": True}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repository", type=Path, required=True)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    assert not sys.flags.optimize
    repo, out = a.repository.resolve(), a.output.resolve()
    parent, candidate = a.parent.resolve(), a.candidate.resolve()
    assert sha(parent.read_bytes()) == PARENT and sha(candidate.read_bytes()) == CANDIDATE
    out.mkdir(parents=True, exist_ok=False)
    path = repo / "tests/integer_tensor_review/wire/protocol.py"
    spec = importlib.util.spec_from_file_location("heptatorch_rounding_wire", path)
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    sys.path.insert(0, str(repo / "tests"))
    from half_logical.native import Memory
    inputs = list(cases())
    rows = []
    base = wire.Session(parent, directory=out / "parent")
    native = wire.Session(candidate, directory=out / "candidate")
    for operation in OPERATIONS:
        reference, prov = observe_torch(repo, out, inputs, operation)
        for (shape, words), (stride, expected) in zip(inputs, reference, strict=True):
            (x,) = base.run([tensor(wire, shape, words)])
            base.error([(3, operation, x["value"])], 4)
            base.run([(41, x["value"])])
            rows.append(run_positive(wire, Memory, native, shape, words, stride, expected, operation))
    (grad,) = native.run([tensor(wire, (2,), (0x40000000, 0x40400000), requires=1)])
    for operation in OPERATIONS:
        native.error([(3, operation, grad["value"])], 5)
    native.run([(41, grad["value"])])
    (backing,) = native.run([tensor(wire, (4,), (0x3F800000, 0x40000000, 0x40400000, 0x40800000))])
    for shape, stride, offset in [((2, 2), (1, 2), 0), ((2,), (1,), 1)]:
        config = struct.pack("<10Q", len(shape), *shape, *([0] * (4 - len(shape))),
                             *stride, *([0] * (4 - len(shape))), offset)
        (view,) = native.run([(111, backing["value"], wire.Payload(config), 80)])
        for operation in OPERATIONS:
            native.error([(3, operation, view["value"])], 7)
        native.run([(41, view["value"])])
    native.run([(41, backing["value"])])
    # Wire 97 builds an ordinary zero-offset F64 tensor; these subops are F32-only.
    (other,) = native.run([(97, 4, wire.Payload(struct.pack("<9Q", 1, 0, 1, 0, 0, 0, 160, 8, 0x3FF0000000000000)))])
    for operation in OPERATIONS:
        native.error([(3, operation, other["value"])], 7)
    native.run([(41, other["value"])])
    base.finish()
    native.finish()
    old = wire.Session(parent, directory=out / "previous-parent")
    new = wire.Session(candidate, directory=out / "previous-candidate")
    prior = 0

    def same(commands):
        nonlocal prior
        packet = wire.packet(commands)
        expected = old.exchange(packet)
        observed = new.exchange(packet)
        assert expected == observed and wire.parse(expected)[0] == 0
        prior += 1
        return wire.parse(expected)[1]

    for operation in (5, 6, 11, 12, 13, 14, 15, 16, 17, 18):
        (input_row,) = same([tensor(wire, (3,), (0x3E800000, 0x3F800000, 0x40000000))])
        x = input_row["value"]
        (unary_row,) = same([(3, operation, x)])
        y = unary_row["value"]
        same([(13, x), (39, x, 0, 3), (13, y), (39, y, 0, 3)])
        same([(41, x), (41, y)])
    old.finish()
    new.finish()
    report = {"status": "SCOPED_PASS", "parent_sha256": PARENT, "candidate_sha256": CANDIDATE,
              "oracle_lock_sha256": prov["lock_sha256"], "oracle_checks": prov["checks"],
              "parent_ordinary_rounding_refusals": len(rows), "candidate_boundary_refusals": 12,
              "prior_byte_exact_response_frames": prior, "cases": len(rows), "elements": sum(row["elements"] for row in rows), "rows": rows,
              "native_executed": True, "pytorch_executed": True, "canonical_credit": False}
    (out / "qualification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
