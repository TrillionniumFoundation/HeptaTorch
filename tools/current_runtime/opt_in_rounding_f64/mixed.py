#!/usr/bin/env python3
"""Check same-runtime copysign/rounding pipelines in interleaved F32/F64."""
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualify import CANDIDATE, observe_torch, tensor

WORDS = {
    4: (0x80000000, 0xBFC00000, 0x3F7FFFFF, 0x40000001, 0xFFC01234, 1, 0x7F800000, 0xBF000000),
    8: (0x8000000000000000, 0xBFF8000000000000, 0x3FEFFFFFFFFFFFFF, 0x4000000000000001,
        0xFFF8000000001234, 1, 0x7FF0000000000000, 0xBFE0000000000000),
}


def raw(width, words):
    return struct.pack("<" + ("I" if width == 4 else "Q") * len(words), *words)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("repository", "candidate", "output"):
        parser.add_argument("--" + label, type=lambda path: Path(path).resolve(), required=True)
    args = parser.parse_args()
    assert not sys.flags.optimize
    assert hashlib.sha256(args.candidate.read_bytes()).hexdigest() == CANDIDATE
    args.output.mkdir(parents=True, exist_ok=False)
    _, provenance = observe_torch(args.repository, args.output, [((), (0xBFF8000000000000,))], 19)
    import torch

    spec = importlib.util.spec_from_file_location("rounding_mixed_wire", args.repository / "tests/integer_tensor_review/wire/protocol.py")
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    sys.path.insert(0, str(args.repository / "tests"))
    from half_logical.native import Memory

    transport = args.output / "mixed"
    session = wire.Session(args.candidate, directory=transport)
    words = ops = 0
    for index, shape in enumerate([(), (0,), (9,), (2, 4), (1, 2, 1, 19)]):
        count = math.prod(shape)
        formats = [(4, torch.float32, 39), (8, torch.float64, 98)]
        ids, expected, casestore, caseversion = {}, {}, set(), set()

        def node_frame(width, data):
            if width == 8:
                return tensor(wire, shape, data)
            return (1, len(shape), 0, *shape, *([0] * (4 - len(shape))), wire.Payload(raw(width, data), width))

        def ref_input(width, dtype, data):
            storage = bytearray(raw(width, data))
            sample = torch.frombuffer(storage, dtype=dtype) if storage else torch.empty(0, dtype=dtype)
            return sample.reshape(shape).clone()

        for width, dtype, _ in formats:
            seq = WORDS[width]
            lhs = tuple(itertools.islice(itertools.cycle(seq[index:] + seq[:index]), count))
            rhs = tuple(itertools.islice(itertools.cycle(seq[3:] + seq[:3]), count))
            token_rows = [session.run([node_frame(width, data)])[0] for data in (lhs, rhs)]
            ids[width] = [row["value"] for row in token_rows]
            left, right = ref_input(width, dtype, lhs), ref_input(width, dtype, rhs)
            copy = torch.ops.aten.copysign.Tensor(left, right)
            # Each dtype checks a different first rounding step, then both truncate.
            middle = (torch.ops.aten.floor.default if width == 4 else torch.ops.aten.ceil.default)(copy)
            tail = torch.ops.aten.trunc.default(middle)
            expected[width] = [left, right, copy, middle, tail]

        copies = session.run([(2, 16, *ids[width]) for width, _, _ in formats])
        for (width, _, _), row in zip(formats, copies, strict=True):
            ids[width].append(row["value"])
        rounds = session.run([(3, 21, ids[8][2]), (3, 20, ids[4][2])])
        ids[8].append(rounds[0]["value"])
        ids[4].append(rounds[1]["value"])
        terminals = session.run([(3, 19, ids[width][3]) for width, _, _ in formats])
        for (width, _, read), row in zip(formats, terminals, strict=True):
            ids[width].append(row["value"])
            for step, (token, reference) in enumerate(zip(ids[width], expected[width], strict=True)):
                meta_row, result, dtype_row = session.run([(13, token), (read, token, 0, count), (96, token, 0)])
                m = wire.meta(meta_row)
                assert m["rank"] == len(shape) and tuple(m["strides"][:len(shape)]) == tuple(reference.stride())
                assert tuple(m["shape"][:len(shape)]) == shape and m["numel"] == count
                assert m["storage_offset"] == m["version"] == m["requires_grad"] == 0
                assert struct.unpack("<4Q", dtype_row["payload"])[:2] == ((0, 4) if width == 4 else (4, 8))
                assert result["payload"] == bytes(reference.untyped_storage()), (width, shape, step, result["payload"].hex(), bytes(reference.untyped_storage()).hex())
                owner = Memory(session.owner()["bytes"])
                n = owner.node(token)
                s, v = owner.row(2, n[27]), owner.row(3, n[28])
                assert (n[27], n[28], n[29], s[3], v[2]) == (m["storage_id"], m["version_id"], 0, count * width, 0)
                assert n[27] not in casestore and n[28] not in caseversion
                casestore.add(n[27])
                caseversion.add(n[28])
                assert owner.raw[owner.start + s[2]:owner.start + s[2] + s[3]] == result["payload"]
            session.run([(41, token) for token in ids[width][:-1]])
            (tail,) = session.run([(read, ids[width][-1], 0, count)])
            assert tail["payload"] == bytes(expected[width][-1].untyped_storage())
            session.run([(41, ids[width][-1])])
            words += 3 * count
            ops += 3

    session.finish()
    receipt = json.loads((transport / "transport.json").read_bytes())
    assert receipt["exit_code"] == 0 and receipt["responses_before_stdin_eof"] and receipt["executable_sha256"] == CANDIDATE
    summary = {"status": "SCOPED_PASS", "cases": 5, "operations": ops, "result_words": words,
               "candidate_sha256": CANDIDATE, "oracle_lock_sha256": provenance["lock_sha256"], "frames": receipt["frames"],
               **{f"{x}_sha256": hashlib.sha256((transport / f"{x}.frames.bin").read_bytes()).hexdigest()
                  for x in ("requests", "responses")}}
    (args.output / "mixed.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
