# Native binary32 arithmetic

Every tensor value is an IEEE binary32 bit pattern stored in four little-endian
bytes. The HIA interface carries scalar patterns in a `u64`; arithmetic functions
use the low 32 bits. `fromu` accepts the complete unsigned 64-bit integer range.

## Basic operations

`addf`, `subf`, `mulf`, `divf`, and `sqrtf` implement round-to-nearest,
ties-to-even with gradual underflow. Integer significands, guard/round/sticky
bits, normalization, and explicit packing determine the result. Multiplication
uses a 48-bit significand product; division carries a remainder into the sticky
bit; square root uses an integer square root plus a nonzero-remainder sticky bit.

Signed zero, infinities, normal/subnormal boundaries and invalid arithmetic are
handled explicitly. Arithmetic NaNs become `0x7fc00000`. `negf` and `absf` modify
the sign bit and preserve the remaining payload. `ltf` follows unordered NaN
comparison and equal positive/negative zero semantics. Exceptions are represented
in the returned floating-point value; there is no exposed floating-point status
register or alternate rounding mode.

The component validation executed 2,314,020 operations against independent
NumPy 2.3.5 binary32 operations. Random bit patterns, dense subnormal inputs,
near cancellation, square-root edge cases, and every integer power-of-two
boundary were included. Results agreed bit for bit after canonicalizing
arithmetic NaNs. This is finite executed evidence, not exhaustive enumeration.

## Elementary functions

`expf` uses high/low ln(2) range reduction and an eighth-degree polynomial.
`logf` normalizes the mantissa and evaluates a twentieth-degree log1p polynomial.
`sigmoidf` uses separate positive/negative forms to avoid unnecessary overflow.
`tanhf` uses expm1 near zero and preserves signed zero and tiny results.

| Operation | Maximum observed ULP difference vs NumPy | Maximum observed ULP difference vs high precision |
| --- | ---: | ---: |
| exp | 2 | 1 |
| log | 3 | 1 |
| sigmoid | 2 | 2 |
| tanh | 3 | 2 |

The NumPy comparison covered 3,424,359 native evaluations. The high-precision
control used mpmath at 90 decimal digits for 20,025 inputs, rounded by comparing
neighboring binary32 values, including the worst NumPy disagreements. The
observed maxima are sample statistics and are not universal error bounds.
Special-value classification and infinity-sign disagreements were zero in the
measured sets. Complete records are preserved in `validation/components.json`.

## Software fused multiply-add

`fmaf(a,b,c)` retains the full integer significand product, aligns it with the
addend using guard and sticky information, adds or subtracts according to sign,
then rounds once through the existing binary32 packer. It handles cancellation
even when a separately rounded product would overflow, as well as subnormal
results and signed zeros. Invalid operations and NaNs use the same canonical
quiet-NaN convention as the other arithmetic helpers. The implementation is
native HIA integer code and introduces no external math-library dependency.

An independent scratch entry executed 485,163 exact bit comparisons against
system `libm fmaf`, including 17,576 special-value triples, 250,000 random triples,
156,267 product-cancellation cases, 15,240 exponent/mantissa boundaries and
46,080 subnormal rounding cases. Non-NaN outputs, including zero signs, matched
exactly; arithmetic NaNs were canonicalized before comparison. This is finite
executed evidence rather than a proof over every input triple. Source and image
identities are retained in `validation/session-native.json`.

## Tensor and optimizer numerics

Kernels use deterministic scalar accumulation in logical element order. Adam
uses the software fused multiply-add helper for its first and second moments;
rank-above-one index_add uses it for its alpha-scaled update as described below.
Matrix and convolution accumulation retain their documented separate operations.
There is no tensor core, parallel reduction, TF32 or fast-math backend. A
different PyTorch device or reduction order can therefore produce different
rounding while representing the same operation.

Division's denominator gradient is evaluated as `-g * ((x / y) / y)` so that an
unnecessary squared denominator does not overflow or underflow. Adam's first
moment uses the stable two-branch lerp form. Its second moment uses
`fma(round((1-beta2)*g),g,round(beta2*v))`, and its denominator uses
`sqrt(v)/sqrt(1-beta2^step)+epsilon`; large-gradient regression cases retain
finite moments and valid updates. Optimizer hyperparameters and bias corrections
are binary32. Step counters and explicit state tensors are supplied by callers.

The first hosted graph-lifetime run exposed two long Adam trajectory mismatches.
The earlier weighted-sum first moment and separately rounded second moment
could accumulate a different trajectory over 512 updates. The correction follows
the pinned upstream [Adam implementation](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/optim/adam.py),
[lerp kernel](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/cpu/LerpKernel.cpp)
and [pointwise operations](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/cpu/PointwiseOpsKernel.cpp).
The failed report remains in `validation/hosted-session-failure.json`. The
159-case oracle's inputs, optimizer hyperparameters, expected case count and
tolerances were kept unchanged for the correction. Six native golden response
files were regenerated only after their original independent optimizer/training
references passed; two 512-step native regression pairs were added.

See PyTorch's pinned [numerical accuracy guidance](https://docs.pytorch.org/docs/2.15/notes/numerical_accuracy.html)
for the compatibility target's own distinctions between devices and evaluation
orders. Scoped capability checks record dtype, layout, tolerances and references.

## Concatenation, layout and empty shapes

Cat and stack copy binary32 bits without arithmetic conversion. Forward tests
include signed zeros, subnormals, infinities and NaN payloads; copying retains
those payloads, while later arithmetic continues to use the runtime's canonical
NaN rule. Their first-order derivatives split or select the incoming cotangent.
Repeated inputs add their contributions in logical order. Physical output
layout does not change the logical ordering of gradient/work buffers.

Cat's channels-last selection follows the pinned
[memory-format stride rules](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/c10/core/MemoryFormat.h)
and the [cat/stack implementation](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/TensorShape.cpp).
It applies when every input suggests that format; exact singleton strides can
resolve cases where both contiguous layouts are possible. A `(0,)` input still
participates in format selection even though cat skips it for shape matching.
Rank-three stack with insertion after the last axis inspects the virtual
unsqueezed layout. Forward writes honor the resulting physical strides.

The general descriptor now validates every dimension before selecting numel
zero for an empty shape. It constructs ordinary contiguous strides by
multiplying by `max(dimension,1)`, matching the pinned
[empty-tensor restriding code](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/c10/core/TensorImpl.h).
For example, `[2,0,3]` has strides `[3,3,1]`; a late zero in
`[262144,262144,0,262144]` does not make the earlier large prefix an allocation
request. Every dimension must still satisfy the 262,144 bound. Empty rank-four
stride products are at most `2^54`, so this admission does not introduce an
integer overflow path.

The current native review passed 81 cat/stack cases with 1,673 exact numeric
comparisons and 16 gradient/work-buffer refusal checks. Eight actual hosted
PyTorch layout observations were then replayed against the merged native
image: all 924 forward values and all input/output shapes and strides agreed.
These executed records are in
[tensor composition evidence](../validation/tensor-composition-native.json);
the original hosted observations retain their own CI/source provenance in
[boundary observations](../validation/hosted-boundary-observations.json).

## Indexed addition and alpha rounding

Scatter add begins with self and adds source values at the supplied indices.
Index add begins with self and applies an explicit binary32 alpha coefficient
to the selected source slices. Repeated destinations accumulate in the native
loop order. Noncontiguous input values are accessed by their actual strides;
saved integer indices are exact signed i64 values, independent of float32
conversion or precision.

The pinned CPU index_add paths have a rounding distinction. For rank zero and
rank one, the native implementation computes
`addf(current, mulf(alpha, source))`, rounding the product and sum separately.
For rank above one it computes `fmaf(alpha, source, current)`, rounding once.
This follows the measured distinction between the pinned CPU's scalar and
vectorized [ufunc addition](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/ufunc/add.h)
through [index_add](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/TensorAdvancedIndexing.cpp).
For suitable large finite inputs, the separately rounded product overflows
while the fused product-plus-add remains finite. The coefficient is not
restricted to finite values: signed zeros, infinities and NaNs use the same
binary32 arithmetic conventions as the underlying helpers.

The first candidate used separate rounding at every rank. An actual six-case
PyTorch probe exposed three differences; the original failing receipt,
source/kernel hashes and focused kernel diff remain preserved. After changing only the rank-above-one forward
update, all six recorded cases and 102 observed values matched. The same
independent references and thresholds were retained. Nine supplemental cases
compared 1,652 values, including shared transpose views and special-value alpha
arithmetic; 100 main indexed-addition cases compared another 1,553 values.

This rule is qualified for the locked `2.15.0+cpu` wheel on the recorded Linux
x86-64 CPU/ISA profile. It does not claim that every upstream CPU instruction
set, compiler, device, parallel schedule or future PyTorch build makes the same
rounding choices. Release source identity and wheel build identity remain
separate in the [oracle lock](../compatibility/pytorch-oracle-lock.json).
The original six-case observations and all 12 indexed boundary observations
are retained in [boundary observations](../validation/hosted-boundary-observations.json).

Source-gradient compatibility and delayed index refusals are part of the
operator contract, not floating-point tolerance adjustments. See
[Scatter add and index add](protocol.md#scatter-add-and-index-add) for the
forward-success/backward-refusal cases and gradient preflight behavior. The
new combined main oracle is configured for 268 cases, run separately against
the default module-linked image and the single-unit reference image. No
successful hosted result for that complete current gate is asserted here until
its receipt has been recorded.
