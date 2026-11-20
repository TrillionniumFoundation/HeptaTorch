# Native BF16 copy qualification

This suite compares every output byte of commands 108 (`to`) and 109
(`_to_copy`) with a newly executed, pinned PyTorch 2.15.0 CPU reference.
It covers F32 input and BF16 output in the admitted rank 0–4 CPU layouts.
Both preserve and contiguous memory formats are exercised. Every result also
has its complete shape, strides, dtype, storage extent, offset, Version,
gradient flags and ownership checked. The native input must remain unchanged,
and the independent result must survive release of that input.

The 70 literal input recipes yield **280 comparisons**. They include:

- Positive and negative quiet/signaling NaNs with distinct payloads, signed
  zero, subnormals, infinities and finite rounding ties.
- Scalar and empty tensors, singleton dimensions with unusual strides,
  vector lengths around 8/16/32-element boundaries, all six rank-three dense
  permutations, channels-last, offset, hole, overlap and broadcast layouts.
- Every 16-bit upper word combined separately with lower words `0x0000`,
  `0x7fff`, `0x8000` and `0xffff`, plus 65,536 deterministic input bit patterns.

`cases.py` generates only inputs. `reference.py` executes actual Torch
operations and retains their complete bytes; it does not calculate expected
BF16 values in Python. Its separate supervised process records the exact
interpreter and arguments, source inputs, official wheel provenance, exit
status, stdout/stderr and both EOFs. `run.py` executes the native HIA image,
retains its raw requests/responses and complete owner snapshots, then reapplies
the original comparisons through a separate readback path.

## Why conversion depends on layout

For the pinned x86 AVX2/AVX512 CPU implementation, the
[vector BF16 conversion](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/cpu/vec/vec512/vec512_bfloat16.h)
produces the BF16 NaN word `0xffff`. The
[scalar conversion](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/headeronly/util/BFloat16.h)
produces `0x7fc0`. The
[copy kernel](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/cpu/CopyKernel.cpp)
chooses between them using the actual TensorIterator geometry. A strided
transpose can therefore produce different NaN words under preserve and
contiguous output formats. Rank-zero and inner broadcast paths use scalar
conversion; even a short contiguous non-scalar tail uses vector conversion.

The HIA implementation inspects the actual newly allocated output layout and
corresponding input stride. It changes only the special NaN result word for
F32-to-BF16 copies. The original RNG support, ordinary finite rounding,
same-dtype copies, BF16 widening and F16 conversions remain separate functions.
The implementation executes this decision and all conversion work natively.

The reference records and checks its actual AVX2/AVX512 capability. This result
does not establish the default scalar CPU kernel, other architectures,
accelerators, general dtype promotion or complete upstream `to` parity.
The complete upstream declaration and subsystem requirements remain in the
whole-project acceptance ledger.

## Run

Use the exact current cold build and the fixed Torch interpreter:

```sh
python tests/bf16_copy/run.py \
  --build-directory /absolute/path/to/current-build \
  --expected-build-sha256 SHA256_OF_BUILD_JSON \
  --torch-python /absolute/path/to/pinned-env/bin/python \
  --output /absolute/path/to/fresh-bf16-result
```

The result requires `report.json` and `physical-qualification.json`. A complete
current-runtime acceptance run must also execute every original mandatory
group on the same new image; a successful BF16 run alone is insufficient.
