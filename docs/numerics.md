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
matrix and convolution accumulation retain their documented separate operations.
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
