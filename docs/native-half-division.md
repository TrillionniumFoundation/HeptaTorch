# Native Half true division

Command `2`, operation `4` now executes true division in HIA when at least one admitted public Tensor operand is F16 or BF16. The implementation supports the current no-grad, rank 0–4 construction and layout contract, reuses the existing rank-aware dtype promotion and broadcasting rules, and creates independent result Tensor/Storage/Version ownership. The canonical implementation is in [runtime.hepta](../tools/current_runtime/source/runtime.hepta); [the manifest](../tools/current_runtime/source/manifest.json) fixes its ordered units and image.

This is a scoped native implementation. Full dtype/layout/backend parity, Half AD, `rounding_mode`, `out=`, inplace operations and other architectures remain open. No complete upstream declaration or subsystem is closed by these finite cases.

## Exact source change

The 771 source changes only the existing `binary` dispatcher and appends four helpers: `half_div_f32`, `half_div_f64`, `half_div_element`, and `half_div_tensor`. The other 766 functions from 767 remain byte-identical. [The complete inverse](../tools/current_runtime/half_div_source_transition.py) checks the exact parent, current bodies, new route, all four helpers and forward reconstruction. The enclosing [source proof](../tools/current_runtime/fusion_source_transition.py) continues through the earlier complete inverses. These proofs never substitute for execution.

| Identity | SHA-256 |
| --- | --- |
| Complete HIA source | `f534c77069abc15ba15879d44492aacbc5f3709f6a861a0f0b9e0e365232fca1` |
| Static native image | `db935c74bf0a2b359038c7be56f5a5721af8f0c2d757a2330527bd751861ae86` |
| Canonical source manifest | `15337cd10313c9cd485e0c1db70b6d6da46d2cd4afa21053db9eb7f86d57301e` |

All 15 units are cold-compiled with the fixed native tools. A preliminary build made before the canonical manifest was pinned is retained as an inadmissible build; it is not reused as the qualification receipt. The actual reference/native run uses the canonical manifest above.

## Numerical behavior

The fixed upstream CPU implementation selects direct division. For reduced floating results, a scalar RHS retains its original F32 opmath value before the result is rounded to Half/BF16. A TensorIterator scalar includes dimensions with zero stride or size one, so this path is checked for rank-zero, singleton and broadcast layouts. The native implementation obtains that value from the original RHS rather than its rounded promoted copy.

The F32/F64 helpers preserve the observed input NaN priority and payload quieting; invalid finite/infinite combinations and final Half/BF16 conversions follow the fixed actual AVX2 reference. Vector-group and scalar-tail conversion behavior is checked independently. The sources used to establish the intended upstream path are the pinned [BinaryOpsKernel.cpp](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp), [CPU Loops.h](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/cpu/Loops.h), and [TensorIterator.cpp](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/TensorIterator.cpp). Actual numerical expectations come from the separately executed fixed wheel, not from a host reimplementation of the kernel.

Input bytes, Versions, RNG state, complete ownership and transport boundaries remain checked. The two allocation controls require a real refusal followed by recovery of the identical request after restoring its resource limit.

## Actual 1,196 matrix

[The cases](../tests/half_div/cases.py), [reference producer](../tests/half_div/reference.py) and [native observer/physical qualifier](../tests/half_div/run.py) define one reproducible 1,196-case matrix.

| Group | Cases |
| --- | ---: |
| All-word coverage | 32 |
| Broadcast errors | 24 |
| Layouts | 180 |
| Original division requests | 22 |
| Paired NaNs | 192 |
| Promotion | 160 |
| Raw NaNs in layouts | 8 |
| Same-handle operands | 6 |
| Original scalar precision | 144 |
| Special values | 144 |
| Vector boundaries and tails | 284 |
| **Total** | **1,196** |

The actual run and independent physical replay passed all 1,196 cases: **38,348 frames, 144,654 assertions, 2,250,592 bitwise result elements, 10,816 owner snapshots, 32 packet controls, two real allocation failures/recoveries and four mapping reads**. Case SHA-256 is `f9499ec57933edbd5344fe462b588ec433beba831b827754f969d6a774eda959`; the actual reference is `1bd2365335d3d09200b143e1654168a36ac85d51e74b8aae95617cab3e75e904`. The physical qualification is `9845bb41f2683873151ff61feffb0bcae2e52537e7d4a43a6a69b33ec608c4b6`.

These scoped results belong to 771. Previously mentioned native 769 source/build/evidence could not be recovered; its reported 1,128 cases are not used. The current 1,196 matrix is newly implemented and actually executed. Full current-image qualification still requires every group listed in [the current build contract](current-runtime.md).

## Preserve original requests and observers

The original Half47 entry executes all 239 request recipes. Its required partition is 151 original refusals and 88 explicit successes: 22 multiplication, 2 negation, 40 single-command add/sub, 2 same-frame FE additions and 22 true divisions. The four private-dtype3 division boundaries retain refusal. Original evidence remains unchanged. Fresh runs preserve the recipes and all predicates; the two stale-capability requests bind their handle words to the actual preceding creation and release. Additional successful allocations can change those two generation tokens, so cross-version equality of all239 packet digests is not a valid requirement.

The 312 multiplication observer also keeps its two original division control requests; the 1006 addition/subtraction observer keeps four. They now require actual division results and complete owner/input/lifetime checks. A dedicated [observer inverse](../tools/current_runtime/half_div_observer_transition.py) recovers the previous full observer files before their older inverses are applied. No former numerical, error, resource or lifetime predicate is deleted.

The coordinator generates a single actual 1,196 reference and shares its complete files, process receipt and interpreter binding among Half47, multiplication 312, addition/subtraction 1,006 and division 1,196. Their native reports and physical proofs must agree on that exact reference. The observed 771 multiplication 312, addition/subtraction 1,006 and original Half47 runs passed with this shared reference. Half47 retained all original requests and four OOM controls, with 3,887 frames and 20,133 assertions. The771 full26-group run and subsequent current gates are reported separately in [the audit](current-audit.md) and [the source-bound review](../validation/current-runtime771/native-review.json).

The current same-F64773 combined gate is **28 groups / 13,075 cases** on one cold-built production image, with the existing separately linked current-object Generator probes. Old 765/767/771 passes, source equivalence, and the sum of focused runs do not establish combined qualification.
