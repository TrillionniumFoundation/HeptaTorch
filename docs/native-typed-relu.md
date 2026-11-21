# Native I64, F64, F16 and BF16 ReLU

The current HIA source implements public command3 / operation5 for no-grad I64, F64, F16 and BF16 tensors in its admitted rank0–4 layouts. The original F32 route remains unchanged. The full candidate is775 functions in15 compilation units; the implementation and its complete predecessor inverse are bound by the [source review](../validation/current-runtime775/source-review.json).

## Value and layout contract

I64 negative values produce zero, including INT64_MIN. Floating negative nonzero values produce positive zero; negative zero is retained. F64 retains every input NaN bit. Reduced-float scalar paths preserve NaN payloads. For the fixed AVX2 reference path, complete32-element vector chunks quiet Half NaNs and canonicalize BF16 NaNs to `0xffff`; the scalar remainder retains its observed payload. Dtype, shape, admitted layout, fresh ownership, input bytes and complete Generator state are independently checked.

This behavior follows the fixed upstream [ReLU entry](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/Activation.cpp), [clamp kernel](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/cpu/TensorCompareKernel.cpp), [AVX2 double clamp](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/cpu/vec/vec256/vec256_double.h) and [reduced-float conversion](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/cpu/vec/vec256/vec256_16bit_float.h). Expected bytes come from actual execution of the locked official CPU wheel, with all eleven provenance checks, AVX2 dispatch and one Torch thread. Host code does not calculate expected Tensor values.

`typed_relu_value` performs only integer bit operations and has no calls or allocations. `typed_relu_tensor` uses the existing guarded no-AD allocation/iteration structure. The old `unary` body gains one branch for operation5 and dtypes2/4/5/6; its entire preceding route and all other772 parent function bodies are recovered exactly by the source inverse. Bool ReLU, unsupported typed unary selectors, typed AD and unsupported metadata remain refusals. General PyTorch dtype/layout/device/frontend behavior remains outside this finite contract.

## Fixed cases and the four original control requests

The776 literal cases retain the geometries and raw input words of405 wide and371 Half negation cases, with ReLU as the operation. They include every Half encoding, F64 exponent classes, infinities and NaN payloads, negative zero, integer limits, rank0–4 layouts, zero-size inputs, scalar/vector boundaries, all24 dense rank-four permutations and real allocation failures. The case digest is `117a548388a0ca08715e5922abb2ea0bf4b1d3b346ee3212238316400378367f`.

The two original negation suites contain four ReLU refusal controls. The successor keeps their input recipes and original96-byte request envelopes and verifies real numeric results:

| Original suite/case | Dtype | Shared ReLU reference case |
| --- | --- | ---: |
| Half negation369 | F16 | 774 |
| Half negation370 | BF16 | 775 |
| Wide negation403 | I64 | 403 |
| Wide negation404 | F64 | 404 |

Each successor checks full result logical/storage bytes, geometry, a fresh Tensor/Storage/Version, unchanged source and previously computed negation, complete Generator preservation and final release. Each old suite still has46 control slots: two explicit numeric successors and44 other original slots. All original negation comparisons, failed-suffix transactions, owners, mapping checks, allocation recovery and EOF predicates remain. A complete byte-level inverse recovers both original observers and the earlier helper that pins them. Original Half239 and F6487 request partitions contain no ReLU requests and remain151/63 refusals respectively.

## Actual scoped execution

The same actual cold build, source and ELF were used for all three groups. The same newly executed776-case ReLU reference and its actual process receipt feed both old suites and the new group.

| Group | Cases | Frames | Assertions | Owner snapshots | Bitwise elements | Real OOM/recovery pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Half negation | 371 | 11,415 | 37,920 | 2,729 | 272,448 | 2 |
| I64/F64 negation | 405 | 10,522 | 37,490 | 2,971 | 287,279 | 2 |
| Typed ReLU | 776 | 21,863 | 75,060 | 5,672 | 559,719 | 4 |

All1552 cases and original physical readers passed. The original41 report/receipt/stream files comprise6,834,992 bytes; their complete220,544-byte XZ archive, all modes and exact single-stream EOF were read and verified independently. Every literal case, native row, provenance check, aggregate count and four old-request mapping was rechecked. The execution-host readers replayed raw frames, owner/storage snapshots and reference blobs. The local independent review has read all reports; it has not yet independently replayed all execution-host raw frames and blobs. These boundaries are explicit in the [native review](../validation/current-runtime775/native-review.json).

The original source-admission, missing-upload and sparse-dependency setup failures remain failures with zero native cases. The independent report reader's two schema assumptions were repaired against the unchanged originals. No failed report was rewritten as a success, and no numerical expectation or tolerance was relaxed.

## Complete current gate

The [coordinator](../tests/qualify_current_runtime.py) requires29 groups and13,851 cases on a single current main image, with separately linked current-object Generator probes. Prior PR53 passed25 groups/13007 cases and failed three source-admission groups before native execution; that entire run remainsFAIL. Its shared F32 source-helper pin is repaired without changing numerical predicates or the native-execution requirement. A new complete fixed-tree run and independent full artifact readback remain mandatory.

The2,590 native declarations,688 derivative declarations and17 complete subsystem contracts retain their full denominator. This bounded CPU ReLU implementation closes none of those complete contracts by itself.
