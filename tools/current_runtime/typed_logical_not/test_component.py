"""Independent runner for the standalone native HIA logical-not component."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile

COMPILER_SHA256 = "87c247e4d1c3c9a449d2947fd4227fdc186d519bffd203a04cafce7861a4ed54"
FORMATS = {0: "f", 2: "q", 3: "d", 4: "d", 5: "e"}
WIDTHS = {0: 4, 1: 1, 2: 8, 3: 8, 4: 8, 5: 2, 6: 2}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def binary_value(dtype, word):
    if dtype == 1:
        return bool(word)
    # Decode into an independent IEEE/Python numerical value. A NaN is
    # truthy; both signed zeros are false; all finite subnormals are truthy.
    if dtype == 6:
        return struct.unpack("<f", struct.pack("<I", word << 16))[0]
    return struct.unpack("<" + FORMATS[dtype], word.to_bytes(WIDTHS[dtype], "little"))[0]


def oracle(dtype, words):
    return [int(not binary_value(dtype, word)) for word in words]


def known_words(dtype):
    if dtype == 1:
        return [0, 1] * 257
    if dtype in (5, 6):
        return list(range(1 << 16))
    if dtype == 2:
        signed_edge = (1 << 63) - 1
        edge = [0, 1, 2, signed_edge - 1, signed_edge, signed_edge + 1,
                signed_edge + 2, (1 << 64) - 2, (1 << 64) - 1]
    else:
        bits = WIDTHS[dtype] * 8
        frac = 23 if bits == 32 else 52
        exp = ((1 << (bits - frac - 1)) - 1) << frac
        pos = [0, 1, 2, (1 << frac) - 1, 1 << frac, (1 << frac) + 1,
               exp - 1, exp, exp + 1, exp + (1 << (frac - 1)), exp + ((1 << frac) - 1)]
        edge = pos + [w | (1 << (bits - 1)) for w in pos]
        edge += pos * 3
    rng = random.Random(20261122 + dtype)
    return edge + [rng.getrandbits(WIDTHS[dtype] * 8) for _ in range(4096)]


def torch_reference():
    os.environ["ATEN_CPU_CAPABILITY"] = "avx2"
    import torch
    require(torch.__version__ == "2.15.0+cpu", "exact official reference version required")
    torch.set_num_threads(1)
    dtypes = {0: torch.float32, 2: torch.int64, 3: torch.float64,
              4: torch.float64, 5: torch.float16, 6: torch.bfloat16}

    def compare(dtype, words, actual):
        if not words:
            return
        if dtype == 1:
            values = torch.tensor(words, dtype=torch.bool)
        else:
            raw = bytearray().join(w.to_bytes(WIDTHS[dtype], "little") for w in words)
            values = torch.frombuffer(raw, dtype=dtypes[dtype])
        result = torch.logical_not(values)
        require(result.dtype == torch.bool and result.shape == values.shape,
                "unexpected Torch dtype/shape")
        require(result.tolist() == actual, f"Torch result differs for dtype {dtype}")

    return compare


def run(source, compiler, use_torch=False):
    source_raw, compiler_raw = source.read_bytes(), compiler.read_bytes()
    require(sha(compiler_raw) == COMPILER_SHA256, "wrong pinned HIA compiler bytes")
    compare_torch = torch_reference() if use_torch else None
    with tempfile.TemporaryDirectory(prefix="heptatorch-logical-not-") as td:
        out = Path(td) / "native"
        images = []
        for _ in range(2):
            p = subprocess.run([compiler.resolve()], input=source_raw, capture_output=True, timeout=30, env={"PATH": ""})
            require(p.returncode == 0 and not p.stderr and p.stdout.startswith(b"\x7fELF"),
                    f"native compilation failed: exit={p.returncode}, stderr={p.stderr!r}")
            images.append(p.stdout)
        require(images[0] == images[1], "independent native builds differ")
        out.write_bytes(images[0])
        out.chmod(0o700)

        def native(request, expected):
            p = subprocess.run([out], input=request, capture_output=True, timeout=30, env={"PATH": ""})
            require(p.returncode == 0 and not p.stderr, f"native execution failed: {p.returncode}, {p.stderr!r}")
            expect_bytes = struct.pack("<" + str(len(expected)) + "Q", *expected)
            require(p.stdout == expect_bytes, f"native response differs for header {request[:16].hex()}")

        counts = {}
        for dtype in WIDTHS:
            words = known_words(dtype)
            counts[str(dtype)] = len(words)
            sizes = (0, 1, 7, 25, 60000, 60000)
            start = 0
            for size in sizes:
                chunk = words[start: start + size]
                start += len(chunk)
                expected = oracle(dtype, chunk)
                request = struct.pack("<" + str(len(chunk)+2) + "Q", dtype, len(chunk), *chunk)
                native(request, [0, *expected])
                if compare_torch is not None:
                    compare_torch(dtype, chunk, expected)
            require(start == len(words), "case not exercised")
        malformed = [
            b"", b"\0" * 8, b"\0" * 15,
            struct.pack("<2Q", 7, 0),
            struct.pack("<2Q", 0, (1 << 64) - 1),
            struct.pack("<2Q", 0, 60001),
            struct.pack("<2Q", 0, 1),
            struct.pack("<3Q", 0, 0, 1),
            struct.pack("<3Q", 0, 2, 1),
        ]
        for request in malformed:
            native(request, [1])
        maximal = [0, 1] * 30000
        native(struct.pack("<60002Q", 4, len(maximal), *maximal), [0, *oracle(4, maximal)])
        require(sha(compiler.read_bytes()) == COMPILER_SHA256 and source.read_bytes() == source_raw,
                "source or compiler changed during test")
        return {"source_sha256": sha(source_raw), "compiler_sha256": COMPILER_SHA256,
                "elf_sha256": sha(images[0]), "independent_identical_compiles": 2,
                "dtype_cases": counts, "campaign_values": sum(counts.values()),
                "max_count_extra_values": len(maximal), "malformed_controls": len(malformed),
                "pytorch_2_15_cpu_run": use_torch,
                "scope": "standalone HIA raw-word component only; no production runtime qualification"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--torch", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.compiler, args.torch), indent=2, sort_keys=True))
