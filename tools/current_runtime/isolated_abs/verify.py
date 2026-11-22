#!/usr/bin/env python3
"""Build and test the isolated native wide abs component, not HeptaTorch runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
import struct
import subprocess

SOURCE = Path(__file__).with_name("abs_word.hepta")
COMPILER_SHA = "b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f"
LINKER_SHA = "761a941fd7a2d53eb9b45869fcfc20e970640e9dd11d60f7c61a89812b61d74e"
MASK64 = (1 << 64) - 1
SIGN = 1 << 63
INFINITY = 0x7ff0000000000000
REQUEST = b"HTOWNR01"
RESPONSE = b"HTOWNS01"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def snapshot(path, digest, target):
    data = path.read_bytes()
    ensure(sha(data) == digest, "unexpected native tool identity: " + path.name)
    target.write_bytes(data)
    target.chmod(0o500)
    return target


def run(argv, data, env, timeout=30):
    p = subprocess.run(list(map(str, argv)), input=data, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=timeout, env=env)
    return p


def succeeded(p, step):
    ensure(p.returncode == 0 and not p.stderr, f"{step} failed: {p.returncode}, {p.stderr[:200]!r}")
    return p.stdout


def native_build(source, compiler, linker, root):
    root.mkdir()
    obj = succeeded(run([compiler], source, {"PATH": ""}), "native compile")
    ensure(len(obj) >= 64 and obj[:6] == b"\x7fELF\x02\x01"
           and struct.unpack_from("<HH", obj, 16) == (1, 62), "unexpected relocatable output")
    link = b"HIACGS01" + struct.pack("<3Q", 1, 4, 0) + b"step" + struct.pack("<Q", len(obj)) + obj
    image = succeeded(run([linker], link, {"PATH": ""}), "native link")
    ensure(len(image) >= 64 and image[:6] == b"\x7fELF\x02\x01"
           and struct.unpack_from("<HH", image, 16) == (2, 62), "unexpected native image")
    output = root / "abs-word"
    (root / "abs-word.o").write_bytes(obj)
    output.write_bytes(image)
    output.chmod(0o500)
    return output, obj, image


def cases():
    rng = random.Random(0x484941)
    ints = [
        0, 1, 2, SIGN - 2, SIGN - 1, SIGN, SIGN + 1, SIGN + 2, MASK64 - 1, MASK64,
        *(1 << i for i in range(64)),
        *((-(1 << i)) & MASK64 for i in range(64)),
        *(rng.getrandbits(64) for _ in range(2048)),
    ]
    positive = [
        0, 1, 2, 0x000fffffffffffff, 0x0010000000000000, 0x3ff0000000000000,
        INFINITY - 1, INFINITY, INFINITY | 1, INFINITY | (1 << 51) | 1, SIGN - 1,
    ]
    floats = positive + [SIGN | p for p in positive] + [rng.getrandbits(64) for _ in range(2048)]
    return ints, floats


def mathematical_oracle(dtype, word):
    if dtype == 2:
        return abs(int.from_bytes(word.to_bytes(8, "little"), "little", signed=True)) % (1 << 64)
    if word % SIGN > INFINITY:
        # The required NaN transformation concerns the bit payload, not float equality.
        return word % SIGN
    number, = struct.unpack("<d", word.to_bytes(8, "little"))
    return int.from_bytes(struct.pack("<d", abs(number)), "little")


def request(body):
    return REQUEST + struct.pack("<Q", len(body)) + body


def native_cases(image, groups):
    good = [(dtype, word) for dtype, words in groups for word in words]
    valid = [bytes((dt,)) + struct.pack("<Q", w) for dt, w in good]
    malformed = [b"\x02" * n for n in (0, 1, 2, 7, 8, 10, 16, 31, 32, 64)]
    malformed += [bytes((tag,)) + b"\0" * 8 for tag in range(256) if tag not in (2, 4)]
    # Place every refusal between two different successful values to detect stale replies.
    sentinel_a, sentinel_b = (2, MASK64), (4, SIGN | INFINITY)
    a = bytes((2,)) + struct.pack("<Q", sentinel_a[1])
    b = bytes((4,)) + struct.pack("<Q", sentinel_b[1])
    packet = b"".join(map(request, valid)) + b"".join(request(a) + request(m) + request(b) for m in malformed)
    output = succeeded(run([image], packet, {"PATH": ""}), "native execution")
    expected = [mathematical_oracle(*case) for case in good]
    expected += [value for _ in malformed for value in
                 (mathematical_oracle(*sentinel_a), None, mathematical_oracle(*sentinel_b))]
    ensure(len(output) == 48 * len(expected), "native response count/length")
    numerical = []
    for index, want in enumerate(expected):
        at = 48 * index
        header = RESPONSE + struct.pack("<Q", 32)
        ensure(output[at:at+16] == header, f"response {index}: frame header")
        body = output[at+16:at+48]
        exact = b"\x01" + b"\0" * 31 if want is None else b"\0" + struct.pack("<Q", want) + b"\0" * 23
        ensure(body == exact, f"response {index}: want {exact.hex()}, got {body.hex()}")
        if index < len(good):
            numerical.append(struct.unpack_from("<Q", body, 1)[0])
    for frame in (b"x" * 16, REQUEST + b"\0" * 4, request(a)[:-1]):
        result = run([image], frame, {"PATH": ""})
        ensure(result.returncode == 2 and not result.stdout and not result.stderr, "malformed outer frame")
    cold = run([image], b"", {"PATH": ""})
    ensure(cold.returncode == 0 and not cold.stdout and not cold.stderr, "clean outer EOF")
    return numerical, len(malformed), sha(packet), sha(output), len(expected)


def torch_check(interpreter, groups, numerical):
    inputs = b"".join(struct.pack("<Q", w) for _, words in groups for w in words)
    script = Path(__file__).with_name("torch_oracle.py")
    env = os.environ.copy()
    # Fixed reference dispatch before importing torch.
    env.update(ATEN_CPU_CAPABILITY="avx2", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    argv = [interpreter, "-I", "-B", script, *(str(len(words)) for _, words in groups)]
    external = succeeded(run(argv, inputs, env, 90), "independent PyTorch reference")
    expected = b"".join(struct.pack("<Q", word) for word in numerical)
    ensure(external == expected, "exact PyTorch differential mismatch")
    return {"status": "PASS_PINNED_VERSION_AND_CPU_PATH_DIFF", "cases": len(numerical),
            "input_sha256": sha(inputs), "output_sha256": sha(external),
            "wheel_install_provenance_verified": False}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--compiler", type=Path, required=True)
    cli.add_argument("--linker", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--torch-python", type=Path)
    args = cli.parse_args()
    output = args.output.resolve()
    output.mkdir()
    compiler = snapshot(args.compiler, COMPILER_SHA, output / "compiler")
    linker = snapshot(args.linker, LINKER_SHA, output / "linker")
    source = SOURCE.read_bytes()
    first, obj1, image1 = native_build(source, compiler, linker, output / "build-a")
    _, obj2, image2 = native_build(source, compiler, linker, output / "build-b")
    ensure(obj1 == obj2 and image1 == image2, "cold native builds differ")
    ints, floats = cases()
    groups = ((2, ints), (4, floats))
    numerical, refusals, inp, out, frames = native_cases(first, groups)
    ensure(SOURCE.read_bytes() == source and sha(compiler.read_bytes()) == COMPILER_SHA
           and sha(linker.read_bytes()) == LINKER_SHA, "physical input drift")
    oracle = torch_check(args.torch_python, groups, numerical) if args.torch_python else {"status": "NOT_RUN"}
    receipt = {"status": "PASS_ISOLATED_NATIVE_ABS_WORD", "native_toolchain": {
        "compiler_sha256": COMPILER_SHA, "linker_sha256": LINKER_SHA},
        "source_sha256": sha(source), "test_sha256": sha(Path(__file__).read_bytes()),
        "torch_reference_source_sha256": sha(Path(__file__).with_name("torch_oracle.py").read_bytes()),
        "object_sha256": sha(obj1), "elf_sha256": sha(image1), "elf_bytes": len(image1),
        "matching_cold_builds": 2, "i64_math_cases": len(ints), "f64_math_cases": len(floats),
        "typed_refusals_with_same_process_followup": refusals, "native_frames": frames,
        "malformed_outer_controls": 3, "clean_outer_eof_control": True,
        "native_request_sha256": inp, "native_response_sha256": out,
        "torch_reference": oracle, "tensor_dispatch_integrated": False, "full_contract_closed": False}
    (output / "verify.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
