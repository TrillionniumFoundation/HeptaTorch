#!/usr/bin/env python3
"""Qualify isolated first-order exceptional square words and ordered replay."""
import argparse
import hashlib
import importlib.util
from itertools import product
import json
import math
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
CANDIDATE = "04756ef2c21cb781802b03b89db12b78731a5a05972330e6c2daef04db11e5e8"
WORDS = (0, 0x80000000, 1, 0x80000001, 0x3f800000, 0xbf800000,
         0x7f800000, 0xff800000, 0x7f7fffff, 0xff7fffff, 0x7f812345,
         0x7f812346, 0xff812345, 0x7fc123ab, 0x7fc123de, 0xffc123ab)
PROBES = {(0x7f800000, 0): 0xffc00000, (0, 0x7f800000): 0xffc00000,
          (0x7f7fffff, 0x80000000): 0xffc00000,
          (0x7f812345, 0xffc123ab): 0x7fc12345,
          (0xff812345, 0x7fc123de): 0xffc12345,
          (0x3f800000, 0xff812345): 0xffc12345}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report_paths(root):
    return {str(p.relative_to(root)): hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
            for p in sorted(root.glob("*/*.frames.bin"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for part in ("repository", "split-parent", "ordered-parent", "candidate", "output"):
        parser.add_argument("--" + part, type=Path, required=True)
    a = parser.parse_args()
    assert not sys.flags.optimize
    repo, split, ordered_parent, candidate, out = (
        getattr(a, part).resolve() for part in ("repository", "split_parent", "ordered_parent", "candidate", "output"))
    ordered = load("square_ordered", HERE.parent / "opt_in_square_f32_ordered/qualify.py")
    plain = load("square_plain", HERE.parent / "opt_in_square_f32/qualify.py")
    wire = load("square_wire", repo / "tests/integer_tensor_review/wire/protocol.py")
    assert plain.sha(ordered_parent.read_bytes()) == ordered.CANDIDATE
    assert plain.sha(candidate.read_bytes()) == CANDIDATE
    out.mkdir(parents=True, exist_ok=False)
    # An unmodified replay of #87's test code, with just the independently
    # pinned candidate identity substituted in the test's executable guard.
    ordered.CANDIDATE = CANDIDATE
    argv = sys.argv
    try:
        sys.argv = [str(HERE / "qualify.py"), "--repository", str(repo), "--parent", str(split),
                    "--candidate", str(candidate), "--output", str(out / "ordered-replay")]
        ordered.main()
    finally:
        sys.argv = argv
    earlier = json.loads((HERE.parent / "opt_in_square_f32_ordered/evidence.json").read_text())
    replay = json.loads((out / "ordered-replay/qualification.json").read_text())
    assert all(value == earlier[key] for key, value in replay.items() if key != "native_elf_sha256")
    torch = sys.modules["torch"]
    pairs = tuple(product(WORDS, repeat=2))

    def reference(shape, xs, gs):
        x = torch.frombuffer(bytearray(plain.raw(xs)), dtype=torch.float32).reshape(shape).requires_grad_()
        seed = torch.frombuffer(bytearray(plain.raw(gs)), dtype=torch.float32).reshape(shape)
        y = torch.ops.aten.square.default(x)
        y.backward(seed)
        return bytes(y.untyped_storage()), bytes(x.grad.untyped_storage())

    def apply(session, shape, xs, gs):
        x, seed = (session.run([plain.tensor(wire, shape, bits, requires=grad)])[0]["value"]
                   for bits, grad in ((xs, 1), (gs, 0)))
        y, = session.run([(3, 16, x)])
        y = y["value"]
        want, = session.run([(39, y, 0, len(xs))])
        bw, = session.run([(29, y, seed)])
        assert bw["value"] == 1
        g, = session.run([(11, x)])
        assert g["value"] == len(xs)
        session.run([(41, y), (41, seed), (41, x)])
        return want["payload"], g["payload"]

    fixtures = [((), (pair,)) for pair in pairs]
    fixtures += [
        ((16, 16), pairs),
        ((1, 16, 16), pairs[1:] + pairs[:1]),
        ((1, 1, 1, 259), pairs[-3:] + pairs),
    ]
    for n in (7, 8, 9, 31, 32, 33):
        fixtures.append(((n,), tuple(pairs[(i * 127 + n) % len(pairs)] for i in range(n))))
    seen = {}
    mismatch_by_shape = {}
    count = 0
    baseline = wire.Session(ordered_parent, directory=out / "ordered-parent")
    actual = wire.Session(candidate, directory=out / "special")
    for shape, entries in fixtures:
        xs, gs = zip(*entries, strict=True)
        assert len(xs) == math.prod(shape)
        expected = reference(shape, xs, gs)
        previous = apply(baseline, shape, xs, gs)
        got = apply(actual, shape, xs, gs)
        assert previous[0] == got[0] == expected[0]
        assert got[1] == expected[1], (shape, got[1].hex(), expected[1].hex())
        if not shape:
            seen[entries[0]] = struct.unpack("<I", got[1])[0]
        old_words, new_words = (struct.unpack("<" + "I" * len(xs), data) for data in (previous[1], got[1]))
        key = "scalar" if not shape else "x".join(map(str, shape))
        mismatch_by_shape[key] = mismatch_by_shape.get(key, 0) + sum(a != b for a, b in zip(old_words, new_words))
        count += len(xs)
    baseline.finish()
    actual.finish()
    assert mismatch_by_shape["scalar"] == mismatch_by_shape["16x16"] == 168
    assert all(seen[case] == expected for case, expected in PROBES.items())
    report = {
        "status": "SCOPED_PASS", "candidate_elf_sha256": CANDIDATE,
        "predecessor_elf_sha256": plain.sha(ordered_parent.read_bytes()),
        "cases": len(fixtures), "forward_and_gradient_words_each": count,
        "old_gradient_word_mismatches": mismatch_by_shape,
        "fixed_witnesses_hex": [[f"{x:08x}", f"{g:08x}", f"{seen[(x, g)]:08x}"] for x, g in PROBES],
        "wire_frame_sha256": report_paths(out), "canonical_credit": False,
        "ordered_replay_equal_except_elf_identity": True,
        "ordered_replay_report_sha256": plain.sha((out / "ordered-replay/qualification.json").read_bytes()),
    }
    (out / "qualification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
