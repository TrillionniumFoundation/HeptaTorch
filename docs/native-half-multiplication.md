# Native F16/BF16 tensor multiplication

The native implementation provides ordinary, no-grad `torch.mul.Tensor`
when either public operand has dtype F16 or BF16. The public request is command
`2`, operation `3`. HIA performs dtype selection, input conversion,
broadcasting, output layout selection, multiplication, and publication. Torch
runs only in the independent reference process.

This is one concrete capability within the complete PyTorch rewrite. It does
not close the full dtype gate, autograd gate, CPU backend gate, or the complete
2,590-schema inventory.

The identities below record the original758 multiplication increment. Its
sixteen helpers remain byte-identical in current765, which also implements
[negation](native-half-negation.md) and [addition/subtraction](native-half-addsub.md).
Current combined requirements and actual results are in [the audit](current-audit.md).

## Original758 source and executable

| Item | Identity |
| --- | --- |
| Core source | `027d56d8d8103a48a158d77d6ac750b6eb5d1992a49dd6479e961fe085c51334` |
| Native ELF | `21c062d0f87f1a6eeeb132207930669b0fa0619955beccf685a4ffbf34b1f467` |
| Source manifest | `2d387831207047472596d98710b936e4a774da4d897c5c70dadd9dc5a9b5655d` |
| Source census | 758 core functions; 13 core compilation units plus unchanged RNG and owner support units |
| Immediate predecessor | 742-function source `abbc84b8eca16670e72b8872d1d7ab479f8a1d44f7b2b65deea1649f7682b2c7` |
| Differential corpus | 312 cases; recipe SHA256 `d93e12ad6d96b3c0ec77ceb63986711f0105fc2ff9c9b696572ae0a7bd7576a5` |

The only modified predecessor bodies are `binary` and `command_argument`.
Sixteen helpers are appended. The complete old `binary` body remains available
as `half_mul_previous_binary`, with only its declaration name changed; the
wrapper delegates all non-Half pairs to that body. All other 740 predecessor
function bodies are byte identical.

`tools/current_runtime/half_mul_source_transition.py` and its source delta
reconstruct every predecessor byte, verify the predecessor SHA256, and perform
an exact forward round trip. This proves source custody; it does not transfer
an earlier runtime PASS to the new executable.

## Public behavior

The accepted operand types are Bool, I64, F32, F64, F16 and BF16, provided that at
least one is F16 or BF16. All 20 ordered dtype pairs are covered. Rank zero
through the existing rank-four descriptor limit, right-aligned broadcasting,
empty dimensions, singleton axes, dense permutations, and reachable input
storage offsets are retained. Each successful request creates a fresh result
Tensor, Storage and Version, leaves both inputs unchanged, and has no AD or
saved-input retention edges.

For floating operands in the same rank category, promotion follows this table:

| Other operand | F16 operand | BF16 operand |
| --- | --- | --- |
| Bool or I64 | F16 | BF16 |
| F16 | F16 | F32 |
| BF16 | F32 | BF16 |
| F32 | F32 | F32 |
| F64 | F64 | F64 |

A positive-rank floating tensor takes precedence over a rank-zero floating
operand. For example, an F16 vector multiplied by a rank-zero F64 Tensor
produces F16. An F16 vector multiplied by a positive-rank F64 tensor produces
F64, including a one-element F64 tensor that broadcasts. These are Tensor
promotion rules; Python scalar, wrapped-scalar and scalar-overload APIs are
separate contracts.

Input promotion occurs before the ordinary elementwise operation. The native
implementation materializes promoted inputs with the same preserve-format
layout as the fixed Torch conversion. This is observable when, for example, a
large I64 value converts to F16 infinity before multiplication. Output strides
follow the existing native TensorIterator-compatible layout routine applied to
the promoted inputs, rather than unconditionally forcing row-major order.

The reduced-dtype scalar-RHS specialization requires a further distinction.
Torch's iterator scalar predicate tests all post-conversion dimensions: each
must have size one or zero stride. It is not simply a rank-zero test. For this
specialization, multiplication reads the original RHS value in F32 opmath,
then narrows the product. The HIA implementation retains that original-value
route, including the difference between left and right operand order. A dtype
conversion can materialize a former zero-stride input and thereby change
whether the scalar specialization applies.

Shape incompatibility produces the existing native shape error, with full
transaction rollback. Invalid or stale handles retain the capability error;
unsupported Half arithmetic remains explicitly refused. F32 operands that
require gradients keep the explicit Half AD boundary. A successful product
followed by an invalid command in the same request is rolled back in full.

Direct Half/F64 `as_strided`, Half view/reshape APIs, in-place multiplication,
`out=`, Half autograd, division in the current765 candidate, private dtype3 arithmetic,
and other devices remain open. Noncontiguous Half/F64 inputs in this suite are
constructed through the already supported F32 view and preserve-format cast.
The suite does not fabricate unsupported input descriptors.

## Bitwise numerical contract

Expected bytes come from the exact official CPU wheel fixed in
`compatibility/pytorch-oracle-lock.json`: Torch `2.15.0+cpu`, CPython 3.12,
Linux x86-64, wheel SHA256
`0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3`.
The reference child explicitly sets `ATEN_CPU_CAPABILITY=avx2` before importing
Torch and records that the actual selected capability is AVX2. The original
oracle already fixes one Torch thread; this suite verifies and records that
setting. No older reference environment is changed.

This profile matters for NaN payloads. The HIA result matches the selected
compiled kernel, including signed and signaling NaNs, rather than treating all
NaNs as equivalent. Finite products preserve the existing software IEEE
rounding, signed zero, subnormal and overflow behavior. Invalid finite-input
products such as zero times infinity produce the selected kernel's negative
indefinite value before the result dtype conversion.

Three less visible details are implemented explicitly:

1. Dense Half-to-F32 promotion uses the wheel's FBGEMM conversion behavior.
   Physical groups of eight retain the quieted Half sign and payload. A NaN in
   the final scalar remainder becomes positive F32 `0x7fffffff`. Dense
   permutations use physical storage order for this boundary.
2. The selected BF16 multiplication kernel processes complete groups of 32
   along a coalesced contiguous inner loop. NaNs in those groups become
   `0xffff`; scalar remainder NaNs become `0x7fc0`. Broadcasting, coalescing,
   and generic strided loops therefore affect the exact result bytes.
3. When both operands are NaN, the compiled F16/F32/F64 kernel's operand
   selection depends on the inner loop and its remainder. Native helpers
   preserve that operand priority, including scalar broadcasting and the
   optional smaller vector groups in a remainder.

The AVX2 numerical profile is the qualified target of this suite. DEFAULT
kernel dispatch, other wheel/compiler builds, non-x86 CPUs, and device kernels
have not been established by these results. The complete CPU backend target
remains open.

The semantic sources are the pinned upstream
[TensorIterator](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/TensorIterator.cpp),
[TypeProperties](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/TypeProperties.cpp),
[CPU multiplication kernel](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp),
[CPU loop templates](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/cpu/Loops.h),
and [copy dispatch](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/Copy.cpp).
The oracle lock records why these release source trees equal the wheel build's
source trees despite their distinct Git commit IDs. Actual retained Torch
bytes determine compiled-kernel payload behavior.

## Differential evidence and the historical refusal transition

`tests/half_mul/reference.py` runs all 312 input recipes through real Torch. It
retains each complete logical result and physical storage blob, dtype, shape,
strides, offset, ownership properties, and actual broadcast exception. It also
retains the original eleven wheel provenance checks, actual process arguments
and PID, interpreter identity, and both process streams.

The native runner compares every input and output byte and metadata field,
checks input and RNG immutability, fresh ownership, full error rollback and
result lifetime after input release, and releases all owners. Twenty cases
exercise every one of the 65,536 F16 or BF16 bit patterns. Other groups cover
rank-sensitive promotion, layouts, paired NaN payloads, vector/scalar tails,
errors and unsupported AD. The observer stores complete physical request and
response streams and owner snapshots. `qualify()` replays the original bytes
through the same predicates without launching a compiler, native executable,
or Torch process.

The last 22 recipes are exact successors to 22 historical Half47 refusals.
`tests/half_mul/legacy_controls.py` identifies every original label and literal
operand recipe and binds the original control source and function body hashes.
These requests succeed and must match fresh actual Torch output. The initial758
partition left217 refusals. Current765 also requires two negation and42 add/sub
successors, leaving173 refusals in the same complete239-request recipe. All
original numerical groups, boundaries and four resource-failure controls remain
required. Frozen reference packages and historical evidence are unchanged; no
former refusal is silently omitted or counted as a continuing refusal.

A current765 standalone run creates a new multiplication reference and consumes
the actual1006 reference plus its recorded process, produced by the common
coordinator. `make test` generates and shares these references automatically.

```sh
python3 -B tests/half_mul/run.py \
  --build-directory /absolute/path/to/current-build \
  --expected-build-sha256 BUILD_RECEIPT_SHA256 \
  --torch-python /absolute/path/to/pinned-python \
  --addsub-reference-directory /absolute/path/to/current-review/references/half-addsub1006 \
  --addsub-reference-execution /absolute/path/to/current-review/references/half-addsub1006-execution \
  --output /absolute/path/to/new-half-mul-evidence
```

The coordinator can create the fresh reference once before the Half47
successor and share it with this suite by passing both
`--reference-directory` and `--reference-execution`. The reader binds the
original process and every reference byte again. It does not rerun Torch or
rewrite any reference evidence.

Four original op1/op2 refusal controls now require exact add/sub results from
that shared1006 reference. The current765 run with these actual bindings passed
9,664 frames,37,168 assertions,2,868 owner snapshots and1,313,606 exact elements;
its original312 input corpus and multiplication reference producer are unchanged.
