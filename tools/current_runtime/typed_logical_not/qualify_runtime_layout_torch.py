"""Differential audit of F64 layout and F32 no-gradient behavior on opt-in opcode 127."""
import argparse
from collections import Counter
from itertools import permutations
import json
import math
from pathlib import Path
import struct

from qualify_runtime_torch import load_oracle
from test_runtime_candidate import NATIVE_SHA256, digest, factory, load_wire, require


# Include signed zero, both signs of subnormal, infinity, quiet/signaling NaN.
WORDS = (0, 1 << 63, 1, (1 << 63) | 1, 0x7FF0000000000000,
         0xFFF0000000000000, 0x7FF8000000000555, 0xFFF0000000000001)


def cases():
    for shape in ((2, 3), (2, 3, 4), (2, 3, 4, 5), (2, 1, 3, 1)):
        for order in permutations(range(len(shape))):  # fastest to slowest
            steps = [0] * len(shape)
            span = 1
            for axis in order:
                steps[axis] = span
                span *= shape[axis]
            yield "dense_permutation", shape, tuple(steps), 0
            yield "offset_dense_permutation", shape, tuple(steps), 2
            yield "gapped_permutation", shape, tuple(2 * step for step in steps), 1
            yield "broadcast_permutation", shape, tuple(
                0 if i == order[-1] else s for i, s in enumerate(steps)), 1
    for shape, strides, offset in (
        ((2, 3), (1, 1), 0), ((2, 3), (0, 0), 1), ((3, 2, 3), (1, 2, 1), 0),
        ((2, 2, 2, 2), (1, 2, 1, 4), 2), ((2, 1, 3), (1, 99, 4), 1),
        ((1, 1), (20, 15), 2), ((0, 3), (40, 1), 1),
        ((2, 0, 3), (0, 2, 1), 0), ((2, 3, 0, 2), (21, 2, 0, 1), 2),
    ):
        yield "edge", shape, strides, offset


def run(wire, binary, torch):
    require(__debug__, "assertions required by wire module")
    require(digest(binary) == NATIVE_SHA256, "wrong exact opt-in executable")
    counters = Counter()
    failures = []
    comparisons = 0
    session = wire.Session(binary)
    try:
        def one(*command):
            return session.run([command])[0]

        def inspect(handle):
            metadata = wire.meta(one(13, handle))
            dtype, width, _, _ = struct.unpack("<4Q", one(96, handle, 0)["payload"])
            data = one(98, handle, 0, metadata["numel"])
            require(data["value"] == metadata["numel"], "wrong typed read extent")
            return metadata, dtype, width, data["payload"]

        for kind, shape, strides, offset in dict.fromkeys(cases()):
            span = offset + (1 + sum((n-1)*s for n, s in zip(shape, strides)) if math.prod(shape) else 1)
            words = (WORDS * ((span + len(WORDS)-1)//len(WORDS)))[:span]
            reference = torch.frombuffer(bytearray(struct.pack(f"<{span}Q", *words)),
                                         dtype=torch.float64)
            reference = reference.as_strided(shape, strides, offset)
            expected = torch.logical_not(reference)
            status, records = wire.parse(session.exchange(b"".join(factory(4, (span,), words))))
            require(status == 0 and len(records) == 1 and records[0]["opcode"] == 97,
                    "typed factory failed")
            base = records[0]["value"]
            dims = len(shape)
            payload = struct.pack("<10Q", dims, *shape, *([0] * (4-dims)),
                                  *strides, *([0] * (4-dims)), offset)
            rc, rows = wire.parse(session.exchange(wire.packet([(111, base, wire.Payload(payload), len(payload))])))
            if rc:
                failures.append({"family": kind, "shape": shape, "strides": strides,
                                 "offset": offset, "stage": "input_view", "native_status": rc})
                one(41, base)
                continue
            require(len(rows) == 1 and rows[0]["opcode"] == 111, "wrong public view reply")
            source = rows[0]["value"]
            input_before = inspect(source)
            output = one(127, source)["value"]
            actual, dtype, width, data = inspect(output)
            require((dtype, width, data) == (1, 1, bytes(expected.flatten().tolist())),
                    ("numeric mismatch", kind, shape, strides, offset))
            require(actual["rank"] == expected.ndim and tuple(actual["shape"][:dims]) == tuple(expected.shape)
                    and actual["storage_offset"] == expected.storage_offset()
                    and actual["requires_grad"] == int(expected.requires_grad),
                    ("output metadata mismatch", kind, shape, strides, offset))
            require(output != source and actual["storage_id"] != input_before[0]["storage_id"]
                    and actual["version_id"] != input_before[0]["version_id"]
                    and inspect(source) == input_before, "output aliases or mutates the input")
            if tuple(actual["strides"][:dims]) != expected.stride():
                failures.append({"family": kind, "shape": shape, "strides": strides,
                                 "offset": offset, "stage": "output_strides",
                                 "native": tuple(actual["strides"][:dims]), "torch": expected.stride()})
            comparisons += expected.numel()
            one(41, output)
            one(41, source)
            one(41, base)
            counters[kind] += 1
        for shape, data in (((), [0.0]), ((2, 3), [0.0, -0.0, -3.5, 0.25, 4.0, 0.0]), ((0, 3), [])):
            source = one(*wire.tensor(shape, data, requires=1))["value"]
            input_before = inspect(source)
            require(input_before[0]["requires_grad"] == 1, "expected gradient-enabled native input")
            reference = torch.tensor(data, dtype=torch.float32, requires_grad=True).reshape(shape)
            expected = torch.logical_not(reference)
            require(reference.requires_grad and not expected.requires_grad and expected.grad_fn is None,
                    "unexpected PyTorch gradient semantics")
            output = one(127, source)["value"]
            actual, dtype, width, observed = inspect(output)
            dims = len(shape)
            require((dtype, width, observed) == (1, 1, bytes(expected.flatten().tolist())),
                    "gradient-enabled input produced wrong logical output")
            require(actual["rank"] == dims and tuple(actual["shape"][:dims]) == tuple(expected.shape)
                    and tuple(actual["strides"][:dims]) == expected.stride()
                    and actual["requires_grad"] == 0 and actual["storage_offset"] == 0,
                    "gradient-enabled input produced wrong output metadata")
            require(output != source and actual["storage_id"] != input_before[0]["storage_id"]
                    and actual["version_id"] != input_before[0]["version_id"]
                    and inspect(source) == input_before, "output aliases or mutates the gradient-enabled input")
            comparisons += expected.numel()
            counters["gradient_enabled_f32"] += 1
            one(41, output)
            one(41, source)
        require(one(53)["value"] == 1, "native owner audit failed")
    finally:
        session.finish()
    return {"cases": sum(counters.values()), "cases_by_family": dict(counters),
            "value_comparisons": comparisons, "failed": len(failures), "failures": failures, "native_frames": session.frames}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--binary", required=True, type=Path)
    p.add_argument("--oracle-lock", required=True, type=Path)
    p.add_argument("--install-report", required=True, type=Path)
    a = p.parse_args()
    require(digest(a.binary) == NATIVE_SHA256, "wrong exact opt-in image")
    wire = load_wire(a.protocol)
    torch, provenance = load_oracle(a.oracle_lock, a.install_report)
    result = run(wire, a.binary.resolve(), torch)
    print(json.dumps({"status": "FAIL_SCOPED_LAYOUT" if result["failed"] else "PASS_SCOPED_LAYOUT",
                      "scope": "opt-in opcode 127 F64 strided layouts and F32 gradient-enabled input; not production activation",
                      "native_image_sha256": NATIVE_SHA256, "oracle": provenance, **result},
                     indent=2, sort_keys=True))
    raise SystemExit(int(bool(result["failed"])))


if __name__ == "__main__":
    main()
