# Native Half addition and subtraction

The original765 increment added ordinary, out-of-place addition and subtraction to
the existing binary command when at least one operand is public F16 or BF16.
The arithmetic executes in HIA. Python supplies literal inputs, starts the
fixed reference wheel, and checks retained bytes; it does not evaluate expected
arithmetic or serve as a runtime fallback.

## Admitted contract

| Property | Behavior |
|---|---|
| Public command | Command 2, operation 1 for addition and operation 2 for subtraction |
| Alpha | The existing protocol fixes alpha to 1; no new alpha argument is introduced |
| Operand dtypes | All 20 ordered public dtype pairs containing F16 or BF16, with F32, Bool, I64, F64, F16 and BF16 as the public type set |
| Promotion | Actual tensor rank distinguishes rank-zero operands from dimensioned operands; both inputs are converted to the common dtype before arithmetic |
| Geometry | Existing rank capacity, broadcasting, empty/scalar tensors, and every layout constructible through the existing public factories and conversions |
| Result ownership | A fresh Tensor, Storage and Version; input descriptors, values and versions remain unchanged; no input, view or AD retention edges |
| Gradient boundary | Inputs requiring gradients remain explicitly unsupported, with status 5 and complete rollback |
| Bool subtraction | Either Bool operand causes status 7 before broadcast construction, corresponding to the reference wheel's `NotImplementedError` |
| Other errors | Existing invalid-handle, broadcast and allocation statuses remain 1, 2 and 3 respectively |
| RNG | The operation and all failed transactions leave the complete generator state unchanged |

The public F64 tag is 4. Private wrapped-F64 tag 3 remains protected by the
unchanged command-argument guard and does not become a public operand type.
True division is specified separately in [its current contract](native-half-division.md).
This add/sub contract does not add arbitrary alpha, in-place/out variants,
new Half view APIs, Half autograd, devices or the remainder of the full PyTorch
surface. The full-project status remains **HOLD**.

## Reference semantics and numerical profile

The reference is the repository's fixed Torch 2.15.0+cpu wheel, with all eleven
original provenance checks enabled, `ATEN_CPU_CAPABILITY=avx2`, reported CPU
capability `AVX2`, and one Torch thread. This is an explicit numerical profile;
the result does not certify other CPU dispatch overrides or devices.

The implementation follows the pinned upstream
[add ufunc](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/ufunc/add.h),
[CPU ufunc generator](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torchgen/dest/ufunc.py),
[binary metadata and subtraction dispatch](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/BinaryOps.cpp),
and [Bool subtraction checks](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/BinaryOps.h).
The scalar add expression and vector fused multiply-add use the converted
operands. Subtraction routes through addition with negative alpha. The
multiplication kernel's `original_scalar_value` specialization therefore must
not be applied here. For example, F16[-1] plus a rank-zero F32 tensor containing
word 0x3f801000 yields F16 positive zero after the right operand is rounded to
F16; a dimensioned F32 operand instead produces an F32 result.

The fixed wheel's NaN bytes are part of this contract. For F16, complete 32-lane
inner-loop blocks prefer the right operand's NaN payload and scalar tails
prefer the left operand's payload. For BF16, complete 32-lane blocks produce
0xffff for NaNs and scalar tails produce 0x7fc0. F32/F64 results in the admitted
Half-containing pairs prefer the right NaN payload. Signaling inputs are
quieted, and invalid infinity arithmetic uses the observed indefinite result.
The native implementation derives the inner extent from the actual promoted
operand and result layouts; it does not classify the entire tensor from its
element count alone. The relevant upstream loop mechanics are in
[Loops.h](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/cpu/Loops.h).

## Source and exact predecessor preservation

Only the existing `binary` dispatch body changes. Four functions are appended:
`half_addsub_f32`, `half_addsub_f64`, `half_addsub_element` and
`half_addsub_tensor`. All 760 other predecessor bodies, all 16 multiplication
helpers, all 3 negation helpers and `command_argument` remain byte identical.
The native source still has 15 compilation units including the unchanged RNG
and owner support units.

[The source inverse](../tools/current_runtime/half_addsub_source_transition.py)
and its [complete delta](../tools/current_runtime/half_addsub_source_delta.json)
restore every 761 predecessor source byte and verify a forward round trip.
The inverse composes with the existing761→758→742→740 transitions. It does not
transfer a historical runtime PASS to a new executable.

| Identity | SHA256 |
|---|---|
| Canonical765 source | `b4b2996978dce56c9375ce97b2d2638472cbb154a4c0a7c5de822234e31a9c68` |
| Native executable | `62f74e9232ae30027a64ca61881bb581320b1787529c6ee9e96818a0f8447afe` |
| Pinned source/object manifest | `100f60ce946ac809949d79faf4eca8e8e16d5f4aeacfd907d228edc1ea0c1fd2` |
| Exact761 parent source | `ab3ed0e0898d2006fbdf145357c79cafd599c2be344fcb3cb65fd6ad55efb231` |
| Original1004-case prefix | `553e1a15e63bc66a60faf1276e827410165c9ed8ff0b05d6fb013421930cc6d8` |
| Current1006-case corpus | `07fee181f5b4ea4eff68ccb98c15fbfaf1b81e4b39f0573b21489681443aeea8` |

## Original765 scoped verification

The final native765 image passed the complete **1006-case** actual suite and
pure physical replay. The entire original1004 input prefix is byte-identical;
the two appended cases preserve the original228-byte factory97/FE-add packets.
The qualifier launches neither Torch nor the native runtime and computes no
expected numerical output.

The corpus covers the 20 dtype pairs, rank-zero promotion in both directions,
all 65,536 encodings of each 16-bit dtype with scalar bit patterns and both operand
orders, signed/signaling/paired NaNs, exceptional finite values, vector tails,
permutations, singleton strides, channels-last layouts, empty results, broadcast
errors, Bool subtraction error priority, identical handles and the complete
historical add/sub request mapping. It also checks input preservation, independent
ownership, lifetime after release, invalid/stale/wrong-kind handles, no-AD refusals,
FE transactions, complete rollback, RNG and real allocation failures.

| Actual complete physical proof | Result |
| --- | ---: |
| Cases / native sessions | 1,006 |
| Frames | 31,468 |
| Assertions | 119,842 |
| Owner snapshots | 9,166 |
| Bitwise result elements | 2,899,312 |
| Mapping reads | 12 |
| Real allocation refusals / identical-request recoveries | 4 / 4 |
| Protocol controls | 70 |

The actual native report SHA256 is
`22a62daf77f2a9801105bf63fb5017a91acc6ccaa4999457d9e80f9e7539d968`;
the full physical proof is
`6e1a9f39c0312320f0acf9383a37cc67fe47e78262051a4b53546f12adb7c0b0`.
The actual reference is
`83136ebf1dab33632aa95f023dc543d9345371696a477f2c446be389b2240cde`.
These identify this execution; later runs must bind their own original reports.
An additional independent review passed 7,712 cases and 342,369 bitwise elements.

The final original Half47 also passed actual execution and complete physical
replay: 47 children, 3,513 frames, 17,713 assertions and 94 child EOFs. Its full 239
entry recipe now requires 173 refusals, 22 multiplication successes, 2 negations,
40 single-command add/sub successes and 2 original FE additions. The old 312
multiplication suite was separately rerun with its four explicit add/sub
successors, using the same actual 1006 reference as Half47: 9,664 frames, 37,168
assertions, 2,868 owners and 1,313,606 exact elements.

The FE owner proof validates all 84 words of factory, promotion and result
descriptors, all eight owner tables, slot generations, first-fit data/page
extents and exact promotion reclamation. It preserves every original live
descriptor, Storage payload and Generator. The pure source closure includes
the linked ordinary HIA round_f16/round_bf16 bodies in the fixed rng03 unit;
every other unknown callee is rejected. Four real endpoint pairs and 496
targeted corruption/exception controls independently checked the final rule.

Earlier packet-alignment, omitted-FE, incomplete-temporary-owner and cross-unit
call-graph failures remain preserved with their original observer bytes. They
were resolved before the final runs, without changing the 765 native image.
[The three-file observer inverse](../tools/current_runtime/half_addsub_fe_observer_transition.py)
restores the complete original 1004 cases, reference and native observer with
an exact forward round trip. The original numerical/error/resource predicates
remain; old execution results are not transferred to a new observer.

The complete765 gate subsequently passed24 groups /3,845 cases and independent
original-artifact readback; see [the merged CI review](../validation/current-runtime765/ci-review.json).
The current771 successor requires a fresh26-group /5,446-case run. Its original1006
add/sub cases and four original division-control requests passed actual execution
and independent physical replay:31,532 frames,120,134 assertions,9,186 owner snapshots
and2,899,316 exact elements. The four division controls share the actual1196 reference
with Half47, multiplication312 and division1196; see [the771 review](../validation/current-runtime771/native-review.json).

## Exact old refusal transitions

The immutable source is `frozen737/controls.py` in
`validation/current-half47-source/complete.tar.xz`: 15,168 bytes,
SHA256 `5ea842f9ffcde2bab021773d61e78ea33023b64ad37bbcd7a7a52b6f112bb25b`.
Its exact `entry_controls` function is 7497 bytes including its original final
newline, SHA256 `7b1db2c47be5e94c23455ea49fd4a74fde9184516f1e4c400a9d46538222fc0a`.

The original entry contains 52 single-command add/sub requests and two
same-frame factory/add requests. The complete unchanged entry was audited,
including requests outside the binary loop; no expected arithmetic was
computed. They divide as follows:

| Original requests | Count | Successor behavior |
|---|---:|---|
| Public addition | 22 | Actual-Torch numeric success with complete metadata and ownership checks |
| Public subtraction without Bool | 18 | Actual-Torch numeric success with complete metadata and ownership checks |
| Same-frame factory97/FE-add | 2 | Preserve the original 228-byte request, both responses and complete ownership checks, with separately observed Torch results |
| Bool subtraction, both operand orders and both Half dtypes | 4 | Original status 7 retained and bound to actual Torch `NotImplementedError` |
| Private dtype3, both operations, both operand orders and both Half dtypes | 8 | Original protocol status 7 retained through the unchanged admission guard |

[The input corpus](../tests/half_addsub/cases.py) contains the exact 40 successful
single-command recipes, two FE recipes and four actual Bool errors, including
the original raw input words and same-handle relationships. Each FE recipe
creates F16/BF16 shape [2] from raw words [1,2] and adds it to the pre-existing
F32 words [0x3f800000,0x80000000]. The binary loop uses Half values [1,-0].
The original 239-request entry remains immutable. After the prior 22
multiplication and 2 negation successors, the new 42 add/sub successes produce
66 numeric successors and 173 retained refusals. A complete actual native
status audit executed all 239 original requests and confirmed exactly this
partition and every retained error code. That diagnostic remains separate from
the subsequently completed original Half47 numerical and ownership gate. The
final 47 execution and physical proof verify all original requests and every
other original assertion; the repository-wide 24-group gate is still separate.

The original 312-case multiplication suite also contained four op1/op2 refusal
checks. Its current observer now executes those same requests and compares
their actual results to four explicitly identified rows from the fresh 1006-case
reference. It checks all output bytes and metadata, all 84 input descriptor
words, fresh Tensor/Storage/Version identities, absence of AD/input edges,
input preservation and release. Every original multiplication numerical check,
other refusal, RNG check and lifetime check remains. The old 312 input corpus
and its reference producer remain byte identical.

The final 312 suite was freshly executed and physically replayed with the same
actual 1006-case reference used by the final Half47 run. Its physical proof is
`fc3e7a7565230603fc737df414b1fdd36a61df9f3c89e5cefd610992ee907f77`;
the shared actual 1006 reference is
`82e24eb4db0f0ba76c02ac832b189c1d50efe006dcffe177b3f147b652669ef0`.
The independent main 1006 suite has its own reference identity recorded above.
The earlier 312 execution against the original 1004 reference remains preserved
with its own identity and was not transferred to this changed reference boundary.
[The observer inverse](../tools/current_runtime/half_addsub_observer_transition.py)
restores the exact old 312 runner, SHA256
`49a1d9c4f82ca4c2c44f17866013d670e62f2a0397273cd5977e8d36f8826cf5`,
and binds the four unchanged 371-case negation files. It does not delete or
silently filter the four replaced checks.

## Integration interfaces

`tests/half_addsub/run.py` provides the same `read_reference(directory,
execution, inputs)` and `qualify(report, repository, build, build_sha, output,
inputs)` interfaces used by the other current numerical suites. Its optional
paired `--reference-directory` and `--reference-execution` arguments support a
shared fresh reference. Both actual absolute paths remain in the native report.

The updated multiplication native report uses schema
`heptatorch.half-mul.native-observation.v2`; its unchanged reference retains
schema v1. The optional paired `--addsub-reference-directory` and
`--addsub-reference-execution` arguments bind the same fresh 1006 reference.
The report and physical proof retain its actual hash and the exact four
replaced-control labels. The global coordinator and sealer must validate both
reference custodianships and the raw physical proof, alongside the original
Half47 transition, before admitting the current26-group/5446-case profile.
Both the multiplication and add/sub runners additionally require paired
`--half-div-reference-directory` and `--half-div-reference-execution` arguments.
Their native reports and physical proofs retain `division_reference_sha256`
and all2/4 replaced-division labels. The complete old inverse APIs remain,
with a preceding exact division-observer inverse; no original predicate is omitted.
