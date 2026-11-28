#!/usr/bin/env python3
"""Execute ordinary F64 trunc, floor and ceil requests on independent PyTorch and native images."""
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


PARENT = "c5f542e32675e3e058ca409a5bda57f092f344b8d0a58903797731937f6652c6"
CANDIDATE = "949c8435546617dcf3e6298b5393b76b2dac2cf2b8b183fd82cb7d9bf3299eb8"
OPERATIONS = {19: "trunc", 20: "floor", 21: "ceil"}
WORDS = (0, 0x8000000000000000, 1, 0x800FFFFFFFFFFFFF, 0x0010000000000000,
         0x8000000000000001, 0x000FFFFFFFFFFFFF, 0x8010000000000000,
         0x7FF7FFFFFFFFFFFF, 0xFFF7FFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
         0x3FEFFFFFFFFFFFFF, 0x3FF0000000000001, 0x3FF8000000000000, 0xC004000000000000,
         0x432FFFFFFFFFFFFF, 0x4330000000000000, 0x7FEFFFFFFFFFFFFF,
         0x7FF0000000000000, 0xFFF0000000000000, 0x7FF0000000001234,
         0xFFF0000000001234, 0x7FF8000000007654, 0xFFF8000000007654)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def raw(words):
    return struct.pack("<" + "Q" * len(words), *words)


def cases():
    yield from ((((), (word,)) for word in WORDS))
    shapes = [(0,), (2, 0, 3), (0, 2, 3, 4), (1,), (7,), (8,), (9,), (31,), (32,),
              (33,), (257,), (2, 3), (2, 1, 17), (2, 3, 2, 7), (1, 1, 1, 4096)]
    rng = random.Random(77518)
    for shape in shapes:
        n = math.prod(shape)
        sample = list(itertools.islice(itertools.cycle(WORDS), n))
        for i in range(len(WORDS), n, 2):
            sample[i] = rng.getrandbits(64)
        yield shape, tuple(sample)
    for exponent in range(1022, 1077):
        pivot = exponent << 52
        for magnitude in (pivot - 1, pivot, pivot + 1, pivot + 0x8000000000000, pivot + 0xFFFFFFFFFFFFF):
            yield (), (magnitude,)
            yield (), (magnitude | 0x8000000000000000,)
    threshold_words = tuple(bits | sign for sign in (0, 0x8000000000000000)
                            for boundary in (1, 0x3FF0000000000000, 0x4000000000000000, 0x4320000000000000, 0x4330000000000000)
                            for bits in range(max(0, boundary - 64), boundary + 64))
    yield (len(threshold_words),), threshold_words
    for begin, end in ((0, 1024), (0x000FFFFFFFFFFE00, 0x0010000000000200),
                       (0x7FEFFFFFFFFFFE00, 0x7FF0000000000200),
                       (0x7FF7FFFFFFFFFE00, 0x7FF8000000000200), (0x7FFFFFFFFFFFFC00, 0x8000000000000000)):
        yield (2, 1024), tuple(value | sign for sign in (0, 0x8000000000000000) for value in range(begin, end))


def tensor(wire, shape, words, requires=0):
    bits = raw(words)
    config = struct.pack("<8Q", len(shape), requires, *shape, *([0] * (4 - len(shape))), 160, len(bits))
    return (97, 4, wire.Payload(config + bits))


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
        seed = torch.frombuffer(bytearray(bits), dtype=torch.float64) if bits else torch.empty(0, dtype=torch.float64)
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
    assert node[72] == 4 and all(node[i] == 0 for i in (4, 5, 6, 7, 45, 48, 49, 50, 51, 83))
    start = owner.start + storage[2]
    value = owner.raw[start:start + storage[3]]
    assert len(value) == storage[3]
    return value


def run_positive(wire, decoder, session, shape, words, stride, expected, operation):
    (x,) = session.run([tensor(wire, shape, words)])
    x = x["value"]
    count = len(words)
    xmet, xbytes = session.run([(13, x), (98, x, 0, count)])
    assert xbytes["payload"] == raw(words)
    old = wire.meta(xmet)
    assert physical(session, decoder, x, old) == raw(words)
    (result_row,) = session.run([(3, operation, x)])
    y = result_row["value"]
    ymet, got, xmet2, xbytes2 = session.run([(13, y), (98, y, 0, count), (13, x), (98, x, 0, count)])
    result = wire.meta(ymet)
    assert y != x and result["storage_id"] != old["storage_id"] and result["version_id"] != old["version_id"]
    assert result["rank"] == len(shape) and tuple(result["shape"][:len(shape)]) == shape
    assert tuple(result["strides"][:len(shape)]) == stride and result["numel"] == count
    assert result["storage_offset"] == result["requires_grad"] == result["version"] == 0
    assert result["storage_bytes"] == 8 * count
    assert struct.unpack("<4Q", session.run([(96, y, 0)])[0]["payload"])[:2] == (4, 8)
    assert got["value"] == count and got["payload"] == expected, (shape, got["payload"][:64].hex(), expected[:64].hex())
    assert wire.meta(xmet2) == old and xbytes2["payload"] == xbytes["payload"]
    assert physical(session, decoder, x, old) == raw(words)
    assert physical(session, decoder, y, result) == expected
    session.run([(41, x)])
    (owned,) = session.run([(98, y, 0, count)])
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
            base.error([(3, operation, x["value"])], 7)
            base.run([(41, x["value"])])
            rows.append(run_positive(wire, Memory, native, shape, words, stride, expected, operation))
    factory_refusals = 0
    for observed in [base, native]:
        observed.error([tensor(wire, (2, 2), WORDS[:4], requires=1)], 5)
        factory_refusals += 1
        (backing,) = observed.run([tensor(wire, (2, 2), WORDS[:4])])
        for shape, stride, offset in [((2, 2), (1, 2), 0), ((2,), (1,), 1)]:
            config = struct.pack("<10Q", len(shape), *shape, *([0] * (4 - len(shape))),
                                 *stride, *([0] * (4 - len(shape))), offset)
            observed.error([(111, backing["value"], wire.Payload(config), 80)], 7)
            factory_refusals += 1
        observed.run([(41, backing["value"])])
    (other,) = native.run([(97, 2, wire.Payload(struct.pack("<9Q", 1, 0, 1, 0, 0, 0, 160, 8, 1)))])
    for operation in OPERATIONS:
        native.error([(3, operation, other["value"])], 7)
    native.run([(41, other["value"])])
    base.finish()
    native.finish()
    for label, expected in [("parent", PARENT), ("candidate", CANDIDATE)]:
        receipt = json.loads((out / label / "transport.json").read_bytes())
        assert receipt["responses_before_stdin_eof"] and receipt["exit_code"] == 0
        assert receipt["executable_sha256"] == expected
    report = {"status": "SCOPED_PASS", "parent_sha256": PARENT, "candidate_sha256": CANDIDATE,
              "oracle_lock_sha256": prov["lock_sha256"], "oracle_checks": prov["checks"],
              "parent_ordinary_rounding_refusals": len(rows), "candidate_wrong_dtype_refusals": 3,
              "upstream_factory_refusals_across_both_images": factory_refusals, "cases": len(rows), "elements": sum(row["elements"] for row in rows), "rows": rows,
              "native_executed": True, "pytorch_executed": True, "canonical_credit": False}
    (out / "qualification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
