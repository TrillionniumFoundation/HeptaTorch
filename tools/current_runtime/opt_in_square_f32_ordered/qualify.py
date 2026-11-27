#!/usr/bin/env python3
"""Exercise ordered F32 square on the native wire and locked Torch."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import struct
import sys

HERE = Path(__file__).resolve().parent
CANDIDATE = "50f9532542dfef84037049301c1a2c0a72c77a23ababeb15170dbf0b9f1b2d8e"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for term in ("repository", "candidate", "parent", "output"):
        p.add_argument("--" + term, type=Path, required=True)
    a = p.parse_args()
    assert not sys.flags.optimize
    repo, out, parent, exe = (getattr(a, term).resolve()
                              for term in ("repository", "output", "parent", "candidate"))
    old = load("square_predecessor", HERE.parent / "opt_in_square_f32_ad/qualify.py")
    plain = load("square_nograd", HERE.parent / "opt_in_square_f32/qualify.py")
    wire = load("square_wire", repo / "tests/integer_tensor_review/wire/protocol.py")
    assert plain.sha(parent.read_bytes()) == old.CANDIDATE
    assert plain.sha(exe.read_bytes()) == CANDIDATE
    out.mkdir(parents=True, exist_ok=False)
    ngcases = list(plain.cases())
    ngwant, provenance = plain.observe_torch(repo, out, ngcases)
    torch = sys.modules["torch"]

    def eager_words(words, shape, grad=False):
        raw = plain.raw(words)
        value = torch.frombuffer(bytearray(raw), dtype=torch.float32) if raw else torch.empty(0, dtype=torch.float32)
        return value.reshape(shape).requires_grad_(grad)

    def put(session, shape, words, requires=0):
        return session.run([plain.tensor(wire, shape, words, requires=requires)])[0]["value"]

    def get(session, handle, count):
        row, = session.run([(39, handle, 0, count)])
        assert row["value"] == count
        return row["payload"]

    def release(session, *handles):
        session.run([(41, handle) for handle in handles])

    base = wire.Session(parent, directory=out / "split")
    native = wire.Session(exe, directory=out / "ordered")
    no_grad_frames = 0
    for (shape, words), (_, want) in zip(ngcases, ngwant, strict=True):
        x = None
        for stage in range(4):
            if stage == 0:
                commands = [plain.tensor(wire, shape, words)]
            elif stage == 1:
                commands = [(3, 16, x)]
            elif stage == 2:
                commands = [(39, y, 0, len(words))]
            else:
                commands = [(41, x), (41, y)]
            request = wire.packet(commands)
            prev, got = base.exchange(request), native.exchange(request)
            assert prev == got
            status, rows = wire.parse(got)
            assert status == 0
            if stage == 0:
                x = rows[0]["value"]
            elif stage == 1:
                y = rows[0]["value"]
                assert y != x
            elif stage == 2:
                assert rows[0]["payload"] == want
            no_grad_frames += 1

    dyadic = [(shape, old.bits(values[0]), [old.bits(v) for v in values[1:]])
              for shape, values in old.numeric_cases()]
    rng = random.Random(7751651)

    def random_finite_word():
        word = rng.getrandbits(32)
        return word ^ 0x00800000 if (word & 0x7f800000) == 0x7f800000 else word

    random_cases = []
    for shape in [(1,), (7,), (8,), (9,), (33,), (2, 3, 17), (2, 2, 4, 257)]:
        n = math.prod(shape)
        random_cases.append((shape, tuple(random_finite_word() for _ in range(n)),
                             [tuple(random_finite_word() for _ in range(n))]))
    named = []
    for x, g in ((-0.0, 1.0), (+0.0, -1.0), (-0.0, -1.0),
                 (2.0**-149, 1.0), (-2.0**-149, 1.0),
                 (2.0**-126, 2.0**-24), (-2.0**-126, 2.0**-24),
                 (2.0**-74, 2.0**-76), (-2.0**-74, 2.0**-76),
                 (3.4028234663852886e38, 2.0**-126), (-3.4028234663852886e38, 2.0**-126)):
        named.append(((), old.bits((x,)), [old.bits((g,))]))
    cases = dyadic + random_cases + named
    checked = 0
    case_rows = []
    for idx, (shape, words, seeds) in enumerate(cases):
        x = put(native, shape, words, 1)
        pyx = eager_words(words, shape, True)
        rows = []
        for seed in seeds:
            pyout = torch.ops.aten.square.default(pyx)
            pyout.backward(eager_words(seed, shape))
            y, = native.run([(3, 16, x)])
            y = y["value"]
            meta_x, meta_y = (wire.meta(entry) for entry in native.run([(13, x), (13, y)]))
            assert meta_y["requires_grad"] == 1 and meta_y["version"] == 0
            assert meta_x["storage_id"] != meta_y["storage_id"] and meta_x["version_id"] != meta_y["version_id"]
            assert meta_y["storage_offset"] == 0
            assert tuple(meta_y["shape"][:len(shape)]) == shape
            assert tuple(meta_y["strides"][:len(shape)]) == tuple(pyout.stride())
            assert get(native, y, len(words)) == bytes(pyout.untyped_storage()), ("forward", idx)
            seed_h = put(native, shape, seed)
            bw, = native.run([(29, y, seed_h)])
            assert bw["value"] == 1
            got, = native.run([(11, x)])
            ref = bytes(pyx.grad.untyped_storage())
            assert got["value"] == len(words)
            assert got["payload"] == ref, (idx, shape, got["payload"][:80].hex(), ref[:80].hex())
            rows.append({"gradient_sha256": plain.sha(ref),
                         "forward_sha256": plain.sha(bytes(pyout.untyped_storage()))})
            release(native, y, seed_h)
            checked += len(words)
        assert get(native, x, len(words)) == plain.raw(words)
        release(native, x)
        case_rows.append({"shape": shape, "rounds": rows})

    def fanout(session, value, fan):
        x = put(session, (), old.bits((value,)), 1)
        y = session.run([(3, 16, x)])[0]["value"]
        if fan:
            y2 = session.run([(3, 16, x)])[0]["value"]
            target = session.run([(2, 1, y, y2)])[0]["value"]
        else:
            y2, target = None, y
        seed_h = put(session, (), old.bits((2.0**-76,)))
        session.run([(29, target, seed_h)])
        _, = session.run([(13, x)])
        got, = session.run([(11, x)])
        release(session, seed_h, target, *([y, y2] if fan else []), x)
        return got["payload"]

    edges = []
    for sign in (1, -1):
        for fan in (False, True):
            value = sign * 2.0**-74
            pyx = torch.tensor(value, dtype=torch.float32, requires_grad=True)
            pyout = torch.ops.aten.square.default(pyx)
            if fan:
                pyout = pyout + torch.ops.aten.square.default(pyx)
            pyout.backward(torch.tensor(2.0**-76, dtype=torch.float32))
            ref = bytes(pyx.grad.untyped_storage())
            got, split = fanout(native, value, fan), fanout(base, value, fan)
            assert got == ref and split != ref
            assert (struct.unpack("<I", ref)[0] & 0x7fffffff) == 1 + int(fan)
            edges.append({"sign": sign, "two_square_fanout": fan, "torch_and_ordered_le": ref.hex(),
                          "split_le": split.hex()})

    # The owned input version is truly advanced by the existing no-grad add.
    # Both implementations must reject backward through a saved old version.
    stale = []
    for session in (base, native):
        inputs = old.bits((1.25, -2.0))
        x = put(session, (2,), inputs, 1)
        y = session.run([(3, 16, x)])[0]["value"]
        seed = put(session, (2,), old.bits((1., 1.)))
        before = wire.meta(session.run([(13, x)])[0])
        session.run([(95, x, old.bits((1.,))[0])])
        after = wire.meta(session.run([(13, x)])[0])
        assert before["version"] == 0 and after["version"] == 1
        assert get(session, x, 2) == plain.raw(old.bits((2.25, -1.0)))
        session.error([(29, y, seed)], 6)
        release(session, y, seed, x)
        stale.append({"version_before": before["version"], "version_after": after["version"], "backward_status": 6})
    pt = torch.tensor((1.25, -2.0), dtype=torch.float32, requires_grad=True)
    saved = torch.ops.aten.square.default(pt)
    with torch.no_grad():
        pt.add_(1)
    try:
        saved.backward(torch.ones(2, dtype=torch.float32))
    except RuntimeError as exc:
        assert "modified by an inplace operation" in str(exc)
    else:
        raise AssertionError("Torch did not refuse the stale square input")

    # Compare unrelated multiply/add/relu graph operations on fresh instances.
    common = []
    for exe2, name in ((parent, "split-unrelated"), (exe, "ordered-unrelated")):
        session = wire.Session(exe2, directory=out / name)
        leaf = put(session, (3,), old.bits((-2.0, 0.5, 3.0)), 1)
        weight = put(session, (3,), old.bits((0.125, -1.0, 2.0)))
        mul = session.run([(2, 3, leaf, weight)])[0]["value"]
        both = session.run([(2, 1, mul, leaf)])[0]["value"]
        relu = session.run([(3, 5, both)])[0]["value"]
        seed = put(session, (3,), old.bits((0.75, -1., 1.25)))
        session.run([(29, relu, seed)])
        common.append((get(session, relu, 3), session.run([(11, leaf)])[0]["payload"]))
        release(session, seed, relu, both, mul, weight, leaf)
        session.finish()
    assert common[0] == common[1]
    base.finish()
    native.finish()
    report = {"status": "SCOPED_PASS", "native_elf_sha256": CANDIDATE, "split_elf_sha256": old.CANDIDATE,
              "oracle_checks": provenance["checks"], "finite_input_seed_cases": len(cases),
              "seeded_rounds": sum(len(s) for _, _, s in cases), "forward_and_gradient_words_each": checked,
              "no_grad_cases": len(ngcases), "no_grad_byte_exact_frames": no_grad_frames,
              "rounding_and_fanout": edges, "saved_input_version_refusals": stale,
              "unrelated_graph_equal": True, "cases": case_rows, "canonical_credit": False}
    report["frame_sha256"] = {str(p.relative_to(out)): hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
                              for p in sorted(out.glob("*/*.frames.bin"))}
    (out / "qualification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, sort_keys=True))


if __name__ == "__main__":
    main()
