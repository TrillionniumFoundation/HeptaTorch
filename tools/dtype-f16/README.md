# Native HIA IEEE binary16 conversion

This component implements binary32 to IEEE binary16 round-to-nearest/ties-to-even,
and exact binary16 to binary32 widening. HIA integer operations handle normal and
subnormal values, signed zero, carry into the next exponent, the 65520 overflow
rounding threshold, infinity and NaN quieting. No hardware floating-point mode,
Python numerical runtime, libc conversion routine or upstream kernel supplies
production results.

The packet is little-endian `u32 mode, u32 count, payload`. Mode 0 accepts packed
binary32 words and returns packed binary16 words; mode 1 reverses the conversion.
The response starts with `u32 status`: zero for success or one for refused input.
Exact input length, modes 0/1 and a maximum count of 200,000 are checked before
conversion. The independently compiled executable is a bounded standalone
Linux x86-64 program. These scalar conversion functions are preparation for dtype
integration; the general Tensor dtype/layout/promotion/AD/device API remains open.

## Actual local verification and hosted gate

The fixed native compiler compiles the source twice into byte-identical executables.
The current image is `97c52928ef9dc201c02818c73bb2f82351dbc5fe0305f3b3dc58b32aa92031ac`.
Sixteen native cases passed against CPython's independent `struct` binary16
conversion: 817,083 non-NaN values matched bit-for-bit and 4,471 NaNs passed
classification and native quiet-bit checks. All 65,536 half encodings are widened;
all are also converted back from binary32. The rounding corpus includes 190,464
signed midpoint/neighbor values, covering every adjacent finite positive half pair
and the overflow boundary. There are 100,000 seeded random binary32 encodings,
exact 200,000-item limits in both directions, empty inputs and seven malformed
packet cases. `validation/native-f16-cpython.json` preserves the exact input,
response and independent-reference hashes, source and test identities.

```sh
python3 -I -S -B tests/f16_oracle.py \
  --compiler /absolute/path/to/original-native-hia-compiler \
  --reference cpython --output /absolute/path/to/new-native-review

python3 tests/f16_oracle.py \
  --compiler /absolute/path/to/original-native-hia-compiler \
  --reference pytorch --output /absolute/path/to/new-pytorch-review
```

The dedicated workflow cold-reproduces the fixed compiler and repeats both
references. The PyTorch mode reuses the existing frozen provenance checker and
requires all eleven exact CPU-wheel/source/platform/install checks. Its result
must come from an actual hosted execution; the CPython result alone does not
claim a PyTorch PASS. Inputs and tolerances are shared unchanged between the two
independent references. All non-NaN comparisons are exact, with no epsilon.
NaN payload bits are deliberately not claimed equal across CPU conversion paths;
classification and quieting are checked explicitly and remain distinct from full
upstream dtype parity.

The upstream conversion contract is recorded at the frozen release source:
[torch/headeronly/util/Half.h](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/headeronly/util/Half.h).
The native implementation here uses integer exponent/significand decomposition;
it does not invoke that C++ implementation. The byte packet helpers come from the
previously qualified native BF16 component at source SHA-256
`2f4790a050ccf24bc869c4a9a46c4be4fd740355f363b086eac512e360718469`.
The full 2,590-schema, 17-subsystem acceptance remains HOLD.
