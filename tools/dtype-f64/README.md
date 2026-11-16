# Native HIA binary64 arithmetic

`f64.hepta` implements binary64 addition, subtraction, multiplication, division,
square root, exact binary32 widening and rounded binary32 narrowing. Every target
operation uses HIA unsigned integer instructions. The emitted ELF has no Python,
PyTorch, libc floating-point or host arithmetic dependency.

This is a scalar arithmetic foundation. It is not yet connected to the numerical
Tensor dtype field, promotion, dispatch, autograd or device-specific kernels.
The complete PyTorch dtype and operator gates remain open.

## Numerical contract

Values are IEEE binary64 encodings. The selected rounding rule is nearest with
ties to even, including gradual underflow and finite overflow. Signed zeros and
infinities are handled explicitly. Invalid operations return a quiet NaN; exact
NaN payload and sign propagation are not claimed. Floating-point exception flags,
alternate rounding modes and device-specific modes are outside this profile.

The core has 14 functions, 191 blocks and 669 HIA instructions. Four helpers
(`getr`, `putw`, `shrjam`, `packf`) are copied byte-for-byte from the canonical
binary32 runtime and individually identified in `profile.json`.

Addition aligns extended significands with a sticky bit. The shared binary64
packer retains three rounding bits, normalizes, handles subnormal results and
rounds once. Multiplication computes both limbs of a 53-by-53-bit product from
32-bit partial products; it preserves discarded information before rounding.
Division uses 56 binary quotient steps and jams a nonzero final remainder.

Square root processes a 112-bit scaled radicand two bits at a time. Its running
root and remainder fit within unsigned 64-bit words. The integer recurrence
retains an exact remainder, so the final rounding distinguishes an exact boundary
from values above it. No approximate reciprocal or approximate square-root seed
is used. Exponents are stored with a positive internal bias to avoid signed
underflow in intermediate exponent arithmetic.

## Packet contract

All words are little endian. A request begins with a 32-bit operation number and
a 32-bit count. Binary operations then carry pairs of 64-bit words; unary and
conversion operations carry one 64-bit word per value. The count limit is 60,000,
and the entire request length must match exactly.

| Operation | Number | Input per value | Output per value |
| --- | ---: | --- | --- |
| Add | 0 | Two binary64 words | One binary64 word |
| Subtract | 1 | Two binary64 words | One binary64 word |
| Multiply | 2 | Two binary64 words | One binary64 word |
| Divide | 3 | Two binary64 words | One binary64 word |
| Square root | 4 | One binary64 word | One binary64 word |
| Widen binary32 | 5 | Binary32 encoding in a zero-extended 64-bit word | One binary64 word |
| Narrow to binary32 | 6 | One binary64 word | Binary32 encoding in a zero-extended 64-bit word |

A successful response begins with an eight-byte zero status and the output words.
Malformed requests return only status 1. A widening input with nonzero high
32 bits returns only status 2; all such inputs are checked before writing output
values. There is no persistent target state.

## Independent qualification

```sh
python3 -I -S -B tests/f64_oracle.py \
  --compiler /absolute/path/to/pinned-hia-native-compiler \
  --reference cpython --output /absolute/path/to/new-cpython-review

python3 tests/f64_oracle.py \
  --compiler /absolute/path/to/pinned-hia-native-compiler \
  --reference pytorch --output /absolute/path/to/new-pytorch-review
```

The compiler must have SHA-256
`87c247e4d1c3c9a449d2947fd4227fdc186d519bffd203a04cafce7861a4ed54`.
Each run compiles the actual HIA source twice, requires byte-identical native
images and executes them with an empty PATH. The image identity is
`4225dd162b5c9fcf2a44987839bd410b5e37aef32665683389fb1f4f281f25a2`.
The profile, function bodies, source, producer and image are checked against
their recorded identities.

The fixed 55-case campaign includes special-value cross products, every exponent
field with representative mantissas, adjacent binades, 100,000 arbitrary bit pairs,
100,000 close-exponent pairs, halfway inputs and neighbors, multiplication/division
scaling boundaries, square boundaries, binary32 conversion midpoints, all half
encodings widened through binary32, empty packets and 16 refusal cases. Inputs
are deterministic and every request, response and reference stream has a hash.

The CPython campaign passed 1,968,156 non-NaN bit comparisons and 66,910 NaN
classification checks. The PyTorch mode uses actual `torch.float64` operations
and conversions, verifies all 11 existing wheel provenance checks, and compares
the same corpus without numerical tolerances. The relevant pinned upstream
[binary](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp)
and [unary](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp)
kernel source identities are recorded in the profile.

This campaign does not exhaust all 2^64 input encodings or all operand pairs.
Passing it is scoped arithmetic evidence, not full Tensor or upstream schema
closure. The separately required full-project acceptance record is unchanged.
