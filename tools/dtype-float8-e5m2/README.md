# Native HIA float8 E5M2 conversions

HIA Core v8 implements binary32 -> float8_e5m2 round-to-nearest-ties-even
and float8_e5m2 -> binary32. The native ELF directly calculates exponent,
significand, finite overflow, underflow, subnormals, NaN quieting and sign
preservation with HIA integer instructions; it does not call a host float8
conversion routine. It accepts the same bounded exact-length packet protocol
as the separately qualified native FP16 converter: little-endian u32 mode,
u32 count and words (mode 0 f32, mode 1 byte E5M2). Output begins with u32
status, followed by byte E5M2 (mode 0) or u32 f32 (mode 1). Invalid length,
mode and count >200000 return status=1.

`tests/float8_e5m2_oracle.py` uses two separately derived references:
a CPython enumerated codepoint lattice with exact halfway-to-even selection,
and a separately pinned, provenance-checked real PyTorch 2.15.0+cpu CPU
converter. It compares all 256 E5M2 codewords, both signs of all 124
midpoints and their neighboring f32 patterns, 100,000 seeded f32 bit
patterns, count boundaries, special values, and admission refusals.
Every finite value is compared bit for bit, with no epsilon.
NaNs require correct sign, quiet classification and nonzero payload; exact
NaN payload parity is not claimed.

This is an independently compiled native component; integration with
general Tensor storage, dtype promotion, AD, devices and the 2590 frozen
upstream schemas remains open. Full project acceptance stays HOLD.

## Observed hosted qualification

[Run 43802551513](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43802551513)
executed the pinned native compiler, produced the locked E5M2 ELF twice
byte-identically and passed both CPython and official PyTorch 2.15.0+cpu
reference suites. Each suite passed all 16 cases, 500,887 non-NaN bitwise
comparisons and 391 NaN classification/sign-preservation checks with zero
failures; all 11 independent PyTorch wheel/source/provenance checks passed.
[The full two-reference receipt](../../validation/hosted-native-float8-e5m2.json)
binds the exact source, compiler image, native executable and oracle script.
The original three-case native infinity-overflow failure is retained unchanged
in [the initial failure receipt](../../validation/native-float8-e5m2-first-failure.json)
and is not relabeled as a pass. The corrected native image is
`971beb1762275b501e5cc240c53e258426f01a86979c2f2cb8870f8d91f7bf3e`.
