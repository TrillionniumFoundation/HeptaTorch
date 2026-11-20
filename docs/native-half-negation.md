# Native F16/BF16 tensor negation

Public command `3`, operation `6` now performs no-grad negation of F16 and BF16
Tensors in HIA. The operation preserves the input dtype and shape, allocates a
fresh Tensor/Storage/Version, and follows the existing native TensorIterator
layout selection. All arithmetic and payload handling occur in the native HIA
executable. Torch is used only to produce independent test observations.

This is an additional capability within the full PyTorch rewrite. Half
absolute value, other Half unary operations, Half autograd, direct Half views,
in-place/out variants and other devices remain open. The scalar numerical
command's `abs` operation is a separate protocol; no Tensor `abs` route is
introduced here.

The identities below record the original761 negation increment. Current765
preserves these three native helper bodies and the complete371-case corpus,
and adds [Half addition/subtraction](native-half-addsub.md). Its current24-group
requirements and distinct execution identities are in [the audit](current-audit.md).

## Original761 source and executable

| Item | SHA256 / count |
| --- | --- |
| Canonical HIA source | `ab3ed0e0898d2006fbdf145357c79cafd599c2be344fcb3cb65fd6ad55efb231` |
| Native ELF | `756de9418c05f4ecfc12319c7262cf1ff22e335e4ec32b140721b6c77ee226d7` |
| Source manifest | `ef76a93ddee90e2fc546b2a6f95a5133b7663bd535e1f7693bae3b519345060c` |
| Core functions | 761 |
| Compilation units | 13 core units plus unchanged RNG and owner support units |
| Immediate predecessor | 758-function source `027d56d8d8103a48a158d77d6ac750b6eb5d1992a49dd6479e961fe085c51334` |
| Differential recipe corpus | 371 cases; `cb177901644162f6f127fd4aa6e4dfe0665f7c708d91d5afdd593b52fcd2897c` |

Only the predecessor `unary` and `command_argument` bodies change. The complete
old unary body is preserved as `half_neg_previous_unary`, with only its
function declaration renamed; every non-Half input delegates to that body.
`half_neg_value` and `half_neg_tensor` are the other two new helpers. All 756
remaining predecessor bodies, including all sixteen multiplication helpers,
remain byte identical. The multiplication test fixtures are unchanged.

`tools/current_runtime/half_neg_source_transition.py` verifies the exact
761→758 source inverse and forward round trip. It explicitly verifies the
preserved unary body, the single added command admission route and every
multiplication helper. The inverse composes with the existing 758→742→740
chain. Reconstructing source never transfers an old runtime PASS.

## Numerical behavior

The reference is the exact CPU wheel in
`compatibility/pytorch-oracle-lock.json`: Torch `2.15.0+cpu`, CPython 3.12,
Linux x86-64. The dedicated child selects `ATEN_CPU_CAPABILITY=avx2` before
importing Torch and records the actual AVX2 capability and one-thread setting.
This leaves every earlier suite's reference environment unchanged.

Negation has a significant difference between the selected vector loop and its
scalar remainder:

| Input class | Complete 32-element vector groups | Scalar remainder |
| --- | --- | --- |
| Finite F16/BF16, including subnormals and signed zero | Flip the sign bit | Flip the sign bit |
| F16 infinity | Flip the sign bit | Flip the sign bit |
| F16 NaN | Flip sign; preserve payload and signaling/quiet bit | Flip sign, preserve payload and set the quiet bit |
| BF16 infinity | Flip the sign bit | Flip the sign bit |
| BF16 NaN | Flip sign; preserve payload and signaling/quiet bit | Return positive canonical `0x7fc0` |

For example, an F16 `0x7c01` becomes signaling `0xfc01` in a complete vector
group and quiet `0xfe01` in the scalar remainder. BF16 `0x7f81` becomes
signaling `0xff81` in a complete vector group and canonical `0x7fc0` in the
scalar remainder. A blanket sign-bit flip would therefore be wrong for small
tensors and scalar tails.

The native value helper implements these exact bit rules. The loop uses the
actual coalesced inner extent and physical result position, so a dense
permutation receives the correct vector/scalar boundary even when logical
iteration order differs from storage order. Output stride selection uses the
existing pointwise layout routine with the unary input supplied to both read
positions; duplicating the same shape/stride constraints leaves the iterator
ordering unchanged.

The upstream source explains the two execution paths:

- [CPU negation kernel](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp)
  selects scalar unary negation and vector `neg()` and registers the normal CPU
  dispatch route.
- [AVX2 16-bit floating vector](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/cpu/vec/vec256/vec256_16bit_float.h)
  implements vector negation by XORing the sign bit.
- [Half scalar operators](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/headeronly/util/Half.h)
  and [BFloat16 scalar operators](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/headeronly/util/BFloat16.h)
  route CPU scalar negation through F32 conversion.
- [CPU loop templates](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/cpu/Loops.h)
  determine the complete-vector and remainder boundaries.

Actual retained Torch bytes are the numerical oracle. DEFAULT dispatch,
other builds, non-x86 CPUs and device kernels are outside the established
profile. The complete backend target remains open.

## Inputs, ownership and errors

The operation covers the existing rank-zero through rank-four public Tensor
capacity. The corpus exercises empty tensors, singleton axes, every dense
rank-four axis permutation, rank-three layouts spanning vector tails,
channels-last layouts, and reachable results of F32 views followed by Half
conversion. Existing raw-set command `12` then installs literal signaling and
quiet NaN payloads in those noncontiguous Half inputs.

The Torch reference uses an integer dtype view to assign the same literal
input bits. Both the native raw setter and that public Torch assignment
increment the input Version once per write. This construction preserves exact
payloads and creates equivalent input versions before negation starts.

The result has no gradient history, Saved input, view edge or input-retention
edge. The tests retain its complete bytes after releasing every input handle,
then release the result and verify complete owner cleanup and unchanged RNG.
Invalid, stale and wrong-kind handles retain their existing errors. Unsupported
Half unary operations remain refused. Half requires-grad factories and
conversion from gradient-tracked F32 remain explicit unsupported boundaries;
Torch's general Half autograd support is not represented as an upstream error.

Same-frame factory→FE-reference→negation requests are exercised both as
successful transactions and with a later invalid command that requires full
rollback. Separate controls constrain the real process address-space limit,
observe allocation refusal, restore the limit, and repeat the identical
request successfully with every result byte checked against actual Torch.

## Differential suite and predecessor controls

`tests/half_neg/reference.py` observes all 371 recipes using real Torch. The
corpus includes every one of the 65,536 input words for each dtype, every signed
NaN payload in scalar batches of at most 31 elements, sizes zero through 70,
individual scalar special values, all selected layouts, and two historical
control successors. No expected negation value is computed by the host.

`tests/half_neg/run.py` retains complete native request/response streams, owner
snapshots, resource-control receipts and the original actual reference process
and blobs. Its pure `qualify()` path replays these bytes through the original
predicates without launching Torch, a native process, a compiler or a new
resource-limit change.

Exactly two original Half47 control outcomes change:

| Original label | Original request | New outcome |
| --- | --- | --- |
| `unsupported-half5-neg` | Command 3 / operation 6 / F16 input | Accepted, exact actual-Torch result |
| `unsupported-half6-neg` | Command 3 / operation 6 / BF16 input | Accepted, exact actual-Torch result |

Together with the prior22 multiplication successors, the original761 partition
left **215** of the original **239** controls as refusals. Current765 adds42
explicit add/sub successors, including two original FE requests, and retains173
refusals. The two negation mappings themselves remain unchanged. Original frozen packages,
input recipes and evidence are not rewritten. The two new accepted results
require their own current-image evidence; they cannot be omitted or reported
as continuing refusals.

The standalone runner uses the same CLI form as the multiplication suite:

```sh
python3 -B tests/half_neg/run.py \
  --build-directory /absolute/path/to/current-build \
  --expected-build-sha256 BUILD_RECEIPT_SHA256 \
  --torch-python /absolute/path/to/pinned-python \
  --output /absolute/path/to/new-half-neg-evidence
```

A coordinator may supply a fresh shared reference with both
`--reference-directory` and `--reference-execution`. The reader rebinds its
actual argv, PID, interpreter, complete streams and every physical result blob.
