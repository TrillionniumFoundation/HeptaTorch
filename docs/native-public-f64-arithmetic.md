# Native mixed public F64 arithmetic

Command `2`, operations `1`–`4`, implements native addition, subtraction, multiplication and true division for admitted no-gradient public F64 tensors paired with F32, Bool or I64, in either operand order. All numerical work, promotion, allocation, ownership and rollback execute in HIA. The interface retains the existing rank 0–4, construction and layout limits.

This extension includes direct signed-I64-to-F64 conversion with round-to-nearest, ties-to-even. It avoids an intermediate F32 rounding step, which would lose integer precision before the F64 calculation. Rank-dependent promotion is retained: a rank-zero F64 operand paired with a dimensioned F32 tensor can produce F32 under the fixed Torch rules. Both original Bool subtraction errors remain explicit.

The current implementation is bounded. Same-F64 arithmetic still uses its original path; exact NaN payload parity for that path is open. F64 AD, `out=`, inplace variants, non-default `alpha`, `rounding_mode`, complete layouts/devices and other CPU architectures are also open. No complete upstream declaration or system contract is closed by this finite extension.

## Source identity and complete inverse

| Current mixed-arithmetic identity | SHA-256 |
| --- | --- |
| Complete HIA source | `4722c53b248780534a846a472784a565c4454ab738db75d655a7801b6623f07f` |
| Native ELF | `5cb190cfc4b920bb03e78803e79296e2e046ec90eb6e07b2d5ba6eebfb534edd` |
| Canonical source manifest | `be3eaf71fe817f7d365f70eee21574fcbefa8d6027babb2a57796764b100e362` |

The [canonical manifest](../tools/current_runtime/source/manifest.json) contains 773 functions and 15 ordered compilation units. Relative to771, only `binary` and `half_mul_cast` change; the other769 complete function bodies remain byte-identical. Two real helpers are appended: `integer_i64_to_f64` and `public_f64_binary`.

The `binary` change redirects its non-Half fallback to the new mixed-F64 dispatcher. The complete original Half branch remains unchanged. The new dispatcher admits only the declared public mixed pairs and operations; same-F64, private dtype3 and unsupported operations retain their predecessor route. The cast change adds an exact I64-to-F64 guard before the unchanged remaining conversion path.

The [source inverse](../tools/current_runtime/public_f64_arithmetic_source_transition.py) checks the entire source, both changed bodies, both appended functions and an exact forward round trip to771. It also checks that the direct conversion is the complete original nearest-even integer algorithm with only the binary64 format constants changed. The [enclosing inverse](../tools/current_runtime/fusion_source_transition.py) composes773→771→767→765→761→758→742→740→736→700. A source relation never substitutes for native execution.

The thirteen core unit files are now named `runtime773_00.hepta` through `runtime773_12.hepta`, with the pinned RNG and owner support units. Every current object is cold-compiled and checked against the canonical manifest. Old partition files and the obsolete partition-generation summary are retired by exact Git identity; their replacement is not counted as removed runtime functionality.

## Actual 2,541-case matrix

The [case definitions](../tests/public_f64_arithmetic/cases.py), [reference producer](../tests/public_f64_arithmetic/reference.py) and [native observer and physical reader](../tests/public_f64_arithmetic/run.py) define the complete matrix. Expected numerical bytes come from the actually executed fixed CPU Torch2.15.0 wheel under CPython3.12, with AVX2 and one Torch thread.

| Matrix component | Cases |
| --- | ---: |
| Allocation refusal and recovery | 4 |
| Broadcast errors | 24 |
| Integer precision | 32 |
| Layouts | 288 |
| Original mixed F64 entry requests | 25 |
| NaNs in layouts | 32 |
| Paired NaNs | 144 |
| Promotion | 192 |
| Retained finite same-F64 behavior | 8 |
| Scalar precision | 512 |
| Single NaNs | 128 |
| Special values | 96 |
| Vector boundaries and tails | 1,056 |
| **Total** | **2,541** |

Actual execution and independent physical replay passed all2,541 cases: **80,200 frames,304,449 assertions,460,792 exact result elements,22,960 owner snapshots,82 protocol controls,four real OOM refusals with identical-request recovery and10 mapping reads**. The case digest is `03f58c4d5ae32ccec0eb3772c6bfcf3f06e868b2bab3fc7ca1d3a196210df239`. The actual reference digest is `e5dc97e5c7bc0ed3eaec334103e91615a4f1a11f57f617972ec753fb0710cc93`; its physical qualification digest is `534472f25c96fd310be67f06a5fe06a787291eb0100d04b3f04322567bb01846`.

The controls retain complete input geometry and all84 descriptor words, fresh Tensor/Storage/Version identities, full owner tables, RNG state, release order, mapping rollback and native process EOF. Original command2 words4–7 remain ignored. Nonzero values in those unused fields must preserve the default arithmetic result and ownership; they do not introduce an `alpha` or rounding-mode parameter.

Two failed attempts are preserved as original evidence. The first stopped during reference generation because the new test expected `RuntimeError` for Bool subtraction, whereas the actual fixed Torch error is `NotImplementedError`. The second completed the numerical comparisons but exposed four incorrect new control expectations that assumed unused command words must be refused. The final run corrects those test assumptions to the established protocol and preserves the native source. Neither failed attempt is relabelled as a pass.

## Preserve all87 original F64 requests

The original F6448 reader still executes33 numerical recipes,12 boundary recipes,one87-request entry recipe andtwo real allocation-recovery recipes. Original request bytes remain unchanged. Its current entry partition is:

| Outcome | Requests |
| --- | ---: |
| Retained refusals, including both Bool subtraction errors | 63 |
| Existing F64 negation numerical successor | 1 |
| New mixed arithmetic single-command successors | 22 |
| New original same-frame FE arithmetic successor | 1 |
| **Total** | **87** |

The negation successor consumes the same actual405-case reference as `wide_neg`. All23 arithmetic successors consume the same actual2,541-case reference as the new arithmetic group. Their literal inputs, original labels, commands, reference rows, process receipts and complete raw bytes are bound explicitly.

The [arithmetic successor](../tests/current_runtime_f64_arithmetic_successor.py) reuses the original87-request code and the complete prior negation numerical closure. The [observer inverse](../tools/current_runtime/public_f64_arithmetic_observer_transition.py) proves that exposing `prepare_entry` preserves the original helper byte for byte after reversal, including the prior default86-refusal/one-negation terminal contract. That predecessor contract is labelled separately from the current complete63-refusal/24-success census.

The original240-byte FE request creates an F64 tensor and adds it to an existing F32 tensor. Its [owner rule](../tests/current_runtime_f64_arithmetic_fe_owner.py) checks all84 words of three descriptors, all eight owner tables, first-free slot generations, first-fit aligned allocation, complete source preservation and temporary-promotion reclamation. It derives these obligations from the HIA source and compares actual endpoint snapshots; it does not fabricate an observed intermediate state.

The actual48-case run and independent physical reader passed **3,259 frames,17,923 checks,626 owner blobs and96 native child EOFs**, including both original real OOM controls. Complete reports, process identity and archive custody are recorded in the [source-bound native review](../validation/current-runtime773/native-review.json).

## Common qualification

The coordinator and CI require **27 groups /7,987 cases** on a fresh cold build of this exact current source. All26 preceding groups and their original controls or explicitly verified numerical successors remain mandatory. Generator ABI and reordered-link probes continue to build and execute their separate current-object images.

The two focused groups above cover2,589 cases on the stated source and image. Their original execution input snapshots remain fixed. They do not establish the final repository's full27-group pass or any full-project closure. The [current plan](../validation/current-runtime773/qualification-plan.json) and [audit](current-audit.md) record those separate obligations.
