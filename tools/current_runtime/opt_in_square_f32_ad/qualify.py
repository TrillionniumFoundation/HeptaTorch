#!/usr/bin/env python3
"""Compare bounded square reverse AD with the actual locked Torch eager op."""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys


HERE = Path(__file__).resolve().parent
CANDIDATE = "1360f89da5f0f2c2f387df463ffe6f9be6013a04a318f562852aad6d20f2707c"
SHAPES = ((), (), (0,), (1,), (3,), (7,), (17,), (2, 3), (2, 0), (2, 0, 3), (2, 1, 7),
          (2, 3, 2, 2), (0, 3, 2, 2), (1, 8, 4, 8))
VALUES = (-2.75, -1.25, -0.5, -0.0625, 0.0, 0.125, 0.75, 1.125, 5.5)
SEEDS1 = (0.625, 1.25, -1.5, -0.875, 0.25, 2.125)
SEEDS2 = (-0.125, 0.375, 0.75, -0.125, 2.0, -0.25, -1.0)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bits(values):
    return struct.unpack("<" + "I" * len(values), struct.pack("<" + "f" * len(values), *values))


def numeric_cases():
    for i, shape in enumerate(SHAPES):
        count = math.prod(shape)
        yield shape, tuple(tuple(table[(i + j) % len(table)] for j in range(count))
                           for table in (VALUES, SEEDS1, SEEDS2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository", type=Path, required=True)
    p.add_argument("--parent-projection", type=Path, default=HERE.parent / "opt_in_square_f32")
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert not sys.flags.optimize
    repo, out = a.repository.resolve(), a.output.resolve()
    parent, candidate = a.parent.resolve(), a.candidate.resolve()
    old = load_module("square_previous_qualification", a.parent_projection.resolve() / "qualify.py")
    assert old.sha(parent.read_bytes()) == old.CANDIDATE and old.sha(candidate.read_bytes()) == CANDIDATE
    wire = load_module("square_ad_wire", repo / "tests/integer_tensor_review/wire/protocol.py")
    out.mkdir(parents=True, exist_ok=False)
    # Reuse the predecessor's independent reference + exact provenance path.
    no_grad_cases = list(old.cases())
    no_grad_values, provenance = old.observe_torch(repo, out, no_grad_cases)
    torch = sys.modules["torch"]
    eager = []
    for shape, values in numeric_cases():
        x = torch.tensor(values[0], dtype=torch.float32).reshape(shape).detach().requires_grad_(True)
        rounds = []
        for source in values[1:]:
            seed = torch.tensor(source, dtype=torch.float32).reshape(shape)
            y = torch.ops.aten.square.default(x)
            assert y.requires_grad and not y.is_leaf and y.untyped_storage()._cdata != x.untyped_storage()._cdata
            y.backward(seed)
            assert x.grad.dtype == torch.float32 and tuple(x.grad.shape) == shape
            rounds.append((tuple(y.stride()), bytes(y.untyped_storage()), bytes(x.grad.untyped_storage())))
        eager.append(rounds)
    base = wire.Session(parent, directory=out / "parent-grad")
    native = wire.Session(candidate, directory=out / "candidate-grad")
    rows = []
    for (shape, values), rounds in zip(numeric_cases(), eager, strict=True):
        count = math.prod(shape)
        packet = old.tensor(wire, shape, bits(values[0]), requires=1)
        (previous,) = base.run([packet])
        (leaf,) = native.run([packet])
        x = leaf["value"]
        base.error([(3, 16, previous["value"])], 5)
        base.run([(41, previous["value"])])
        (xmet,) = native.run([(13, x)])
        xmeta = wire.meta(xmet)
        receipts = []
        for vals, (strides, forward, grad) in zip(values[1:], rounds, strict=True):
            y_row, = native.run([(3, 16, x)])
            y = y_row["value"]
            ymet, ybits = native.run([(13, y), (39, y, 0, count)])
            ymeta = wire.meta(ymet)
            assert y != x and xmeta["storage_id"] != ymeta["storage_id"] and xmeta["version_id"] != ymeta["version_id"]
            assert tuple(ymeta["shape"][:len(shape)]) == shape and tuple(ymeta["strides"][:len(shape)]) == strides
            assert ymeta["requires_grad"] == 1 and ymeta["version"] == 0
            assert ybits["payload"] == forward
            (incoming,) = native.run([old.tensor(wire, shape, bits(vals))])
            seed_handle = incoming["value"]
            (bw,) = native.run([(29, y, seed_handle)])
            assert bw["value"] == 1
            (native_grad,) = native.run([(11, x)])
            assert native_grad["value"] == count and native_grad["payload"] == grad, (
                shape, native_grad["payload"].hex(), grad.hex())
            native.run([(41, seed_handle), (41, y)])
            receipts.append({"forward_sha256": old.sha(forward), "leaf_gradient_sha256": old.sha(grad)})
        (leafraw,) = native.run([(39, x, 0, count)])
        assert leafraw["payload"] == old.raw(bits(values[0]))
        native.run([(41, x)])
        rows.append({"shape": shape, "elements": count, "rounds": receipts})
    # An explicit rounding boundary is retained as a mismatch, not a passed
    # fixture. MUL's two rounded contributions need not commute with scaling
    # before multiplying at the bottom of the F32 subnormal range.
    edge_x, edge_s = 2.0 ** -74, 2.0 ** -76
    torch_x = torch.tensor(edge_x, dtype=torch.float32, requires_grad=True)
    torch_y = torch.ops.aten.square.default(torch_x)
    torch_y.backward(torch.tensor(edge_s, dtype=torch.float32))
    (edge_leaf,) = native.run([old.tensor(wire, (), bits((edge_x,)), requires=1)])
    (edge_seed,) = native.run([old.tensor(wire, (), bits((edge_s,)))])
    (edge_out,) = native.run([(3, 16, edge_leaf["value"])])
    native.run([(29, edge_out["value"], edge_seed["value"])])
    (edge_native,) = native.run([(11, edge_leaf["value"])])
    edge_eager = bytes(torch_x.grad.untyped_storage())
    assert edge_eager == bytes.fromhex("01000000")
    assert edge_native["payload"] == bytes.fromhex("00000000")
    native.run([(41, edge_out["value"]), (41, edge_seed["value"]), (41, edge_leaf["value"])])
    mismatch = {"input": "0x1p-74", "seed": "0x1p-76",
                "torch_gradient_le": edge_eager.hex(),
                "native_gradient_le": edge_native["payload"].hex()}
    base.finish()
    native.finish()

    base = wire.Session(parent, directory=out / "parent-nograd")
    native = wire.Session(candidate, directory=out / "candidate-nograd")
    no_grad_frames = 0

    def same(commands):
        nonlocal no_grad_frames
        request = wire.packet(commands)
        expected, got = base.exchange(request), native.exchange(request)
        assert expected == got
        status, records = wire.parse(got)
        assert status == 0
        no_grad_frames += 1
        return records

    for (shape, words), (_, expected) in zip(no_grad_cases, no_grad_values, strict=True):
        (xrow,) = same([old.tensor(wire, shape, words)])
        x = xrow["value"]
        (yrow,) = same([(3, 16, x)])
        y = yrow["value"]
        ymeta, ybits = same([(13, y), (39, y, 0, len(words))])
        assert wire.meta(ymeta)["requires_grad"] == 0 and ybits["payload"] == expected
        same([(41, x), (41, y)])
    base.finish()
    native.finish()
    report = {"status": "SCOPED_PASS", "rows": rows, "candidate_sha256": CANDIDATE,
              "predecessor_sha256": old.CANDIDATE, "canonical_credit": False,
              "eager_provenance_checks": provenance["checks"], "cases": len(rows),
              "seeded_backward_rounds": 2 * len(rows),
              "forward_and_gradient_words_each": 2 * sum(row["elements"] for row in rows),
              "predecessor_actual_requires_grad_refusals": len(rows),
              "no_grad_square_cases_replayed": len(no_grad_cases),
              "no_grad_byte_exact_frames": no_grad_frames,
              "outside_scope_rounding_mismatch": mismatch}
    (out / "qualification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
