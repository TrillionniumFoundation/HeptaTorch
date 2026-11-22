#!/usr/bin/env python3
"""Read I64 and F64 bit words, independently apply pinned CPU torch.abs."""
import os
import struct
import sys

import torch


def main():
    if sys.version_info[:2] != (3, 12) or torch.__version__ != "2.15.0+cpu":
        raise ValueError("reference must be CPython 3.12, Torch 2.15.0+cpu")
    if os.environ.get("ATEN_CPU_CAPABILITY") != "avx2" or torch.backends.cpu.get_cpu_capability() != "AVX2":
        raise ValueError("reference must execute the fixed AVX2 CPU path")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    ints, floats = map(int, sys.argv[1:])
    data = sys.stdin.buffer.read()
    if len(data) != 8 * (ints + floats):
        raise ValueError("wrong number of oracle input words")
    # Reinterpret F64 inputs without converting them to Python floats (NaN bits matter).
    values = struct.unpack("<" + "q" * (ints + floats), data)
    i64 = torch.tensor(values[:ints], dtype=torch.int64, device="cpu")
    f64 = torch.tensor(values[ints:], dtype=torch.int64, device="cpu").view(torch.float64)
    with torch.no_grad():
        result = torch.abs(i64).tolist() + torch.abs(f64).view(torch.int64).tolist()
    sys.stdout.buffer.write(struct.pack("<" + "q" * len(result), *result))


if __name__ == "__main__":
    main()
