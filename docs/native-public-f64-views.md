# Native public F64 views: candidate800

**Status: local exploratory execution and complete physical replay passed; strict800 reference qualification, the full35-group run and final800 CI remain pending.** This extension implements bounded no-gradient public F64 `as_strided` through opcode111 on Linux x86-64, rank0–4. It does not close the complete view, dtype, AD or upstream operator contracts. Full native/derivative/system closure remains **0/0/0**, with project status **HOLD**.

The [active qualification plan](../validation/current-runtime800/qualification-plan.json), [source summary](../validation/current-runtime800/source-review.json) and [native summary](../validation/current-runtime800/native-review.json) distinguish expected gates, observed local results and historical evidence.

## Current source and representation

| Identity | SHA-256 |
| --- | --- |
| Complete HIA source | `aab2e736ff0ef4a282ff73fb1a69a95d2ceeea5260bbdada0af5b30cc16dcd3c` |
| Native ELF | `083a92ad78977396f6d2cacff7b7f559cec4871dfcfb67624db48b6f8d64c3d5` |
| Ordered15-unit manifest | `a5cf09f191db4887b4452001e0b72e75fffb2147f2476544cc3be49d8ffa2c85` |

The complete [source inverse](../tools/current_runtime/public_f64_view_source_transition.py) restores every byte of the published797 predecessor. It changes five existing bodies (`bind_alias`, `so_tensor_live`, `to_command`, `to_strided_view`, `f64_dtype_audit`), adds `public_f64_empty_strided_view`, `public_f64_empty_descriptor` and `so_f64_empty_offset_live`, and retains the other792 functions byte for byte. All original named backward functions remain unchanged. A separate guarded-path proof preserves the original dispatcher domain, non-F64 constructor arguments, ordinary owner byte limit/span checks and successor arguments. Source proofs grant no numerical qualification.

Views use fresh Tensor identities while sharing the original Storage and Version. The cases cover scalar and rank1–4 layouts, noncontiguous, overlapping and zero strides, exact special-value bits, mutation through aliases, root release, nested views and recreation of a nonempty child from an empty parent. Ordinary storage bounds and negative-stride/offset errors remain part of the observed contract. The extension admits only the signed-empty shape/stride combinations actually accepted by the fixed reference; it does not declare arbitrary negative sizes valid.

Empty F64 views may retain an element offset beyond the old ordinary byte-offset limit. For offsets **134,217,729 through9,223,372,036,854,775,807**, the representation uses bit63 as an empty-view marker and stores the exact element offset in the remaining bits. This word is never used as a memory address or span: the data pointer stays at the actual Storage base and span is zero. A valid tag additionally requires an actual zero dimension, zero element count, F64 dtype, no AD/history, the original13 zero no-AD fields, matching descriptor/owner words, and valid root/Storage/Version identity and incarnation. Forging only a zero count on a nonempty shape is rejected. Offsets at or below134,217,728 retain the ordinary representation; all original legal owner offsets remain outside the tagged path.

The observer explicitly decodes this representation. It still compares every raw metadata13 packet, all84 descriptor words and all8 live owner tables. Complete source inverses bind the observer changes; no observed owner snapshot or expected numerical value is projected, patched or manufactured.

## Independent case and session census

| Family | Tensor cases | Accepted | Actual errors |
| --- | ---: | ---: | ---: |
| Original bounded view set | 188 | 178 | 10 |
| Signed-empty shape families | 841 | 606 | 235 |
| Wide-dimension boundaries | 20 | 13 | 7 |
| Empty element-offset boundaries | 75 | 55 | 20 |
| **Total** | **1,124** | **852** | **272** |

The [original cases](../tests/f64_view/cases.py) retain all188 inputs and observations. Case187's actual accepted negative empty shape is preserved as historical incompatibility evidence and receives an explicit native success with its complete additional observations. It is neither discarded nor counted twice. The [added family catalog](../tests/f64_view/family_cases.py) retains all841/20/75 literal recipes and their hashes; historical names describing earlier capacity gaps do not waive any accepted current case.

The1,175 native sessions comprise1,124 tensor cases,3 alias chains,3 allocation families,1 session containing19 protocol controls, and44 owner controls. Each allocation family requires a real native allocation refusal followed by the identical request succeeding after resource restoration, with256 recovered aliases. Owner controls cover the original8 corruptions,8 empty-view versions,6 additional ancestry controls and22 high-offset/tag controls.

The actual local800 campaign and independent replay produced **55,530 frames,391,786 assertions,30,180 full Owner reads and10 mapping reads**. The consolidated final comparators also replayed every one of the1,175 original sessions and reproduced the same frame/check totals. These are actual local observations from the recorded source/image; their Torch installation lacks complete original provenance. They do not qualify the new strict runner or the final current tree.

## Preserve every old request

The original F64 entry still executes all87 literal requests. Its new partition is **61 retained refusals +24 existing numerical successors +2 view successors**. The24 numerical successors remain one negation,22 mixed arithmetic single commands and one original same-frame FE arithmetic transaction; both Bool subtraction refusals remain. Only the two named original view requests become explicit successes, preserving their complete original bytes. Their companions are original188 reference rows174/175; public_f64_logical192 consumes row176 for its original no-gradient scope request. The other AD scope refusal,192 original numerical cases,18 packet controls andtwo OOM controls remain required.

Both view successors require actual result bits, geometry, shared Storage/Version, root identity, all unrelated live descriptors and owner rows, complete temporary-alias release, unchanged original input/RNG and full lifetime companions. The original87 recipe code is retained by the [successor adapter](../tests/f64_view/legacy_successors.py) and its complete inverse.

A focused old797/new800 measurement executed the same two original requests with identical setup/teardown and then independently replayed all raw evidence:

| Measured scope | Old797 | New800 | Difference |
| --- | ---: | ---: | ---: |
| Frames including common setup/teardown | 15 | 41 | +26 |
| Driver assertions including common setup/teardown | 34 | 180 | +146 |
| Complete original Owner blobs | 5 | 10 | +5 |

The common setup/teardown contributes13 frames and30 assertions on each image. Adding the measured differences to the previous qualified F6448 census of3,259 frames/17,923 assertions/626 Owner blobs yields the **expected3,285/18,069/631**. The original48-case denominator,96 child EOFs andtwo OOM controls remain unchanged. This focused experiment did not execute all87 requests, all48 cases or all35 groups; the full new run must establish these expected totals.

## Strict execution and complete physical qualification

Follow [the build and qualification commands](current-runtime.md). After the actual cold build and pinned-reference preparation, execute [`run.py`](../tests/f64_view/run.py), then [`qualify.py`](../tests/f64_view/qualify.py), both with the same `--build-directory` and `--expected-build-sha256`. Supply that view output to the original coordinator with `--f64-view-directory`. The coordinator replays it against the same admitted build before any old view successor consumes the reference.

[`reference.py`](../tests/f64_view/reference.py) is the sole reference entry point. One actual Torch process observes the original188 and all added families, including full storage blobs, allocation observations, chains and fault setups, then finishes the same custody record. [`family_evidence.py`](../tests/f64_view/family_evidence.py) first rereads the complete strict188 reference and then all families with the same original PID, argv, interpreter, environment, RNG, version and source bindings. All11 original provenance expressions, independently pinned complete official wheel/install bytes, actual executable mappings and both stream EOFs remain mandatory. There is no unverified-reference switch.

The pure qualifier prohibits native/Torch process execution. It rebinds original input bytes and modes, checks every recorded child identity, request/response frame, compressed owner stream and EOF, reruns every numerical/metadata/lifetime/error/resource predicate, and requires each recomputed observation to equal its saved original. It consumes all frames, owner snapshots and mappings exactly. Original source-reader and ordinary-domain proofs are rerun, and all inputs are rechecked after replay. The final archive and sealer must preserve this complete evidence, not only success summaries.

The separate1,124-case view gate adds no cases to the original **35 groups /16,365 cases**. Those groups, all prior controls or their explicit successors, separate current-object Generator probes, current-object F64/RNG consumers, legacy gates and final artifact review remain required. The old independent dtype-f64 workflow remains present. Scalar packet sqrt retains its strict PyTorch differences and RNE **PARTIAL** status; this extension adds no public F64 Tensor sqrt operation. Live Native/Make historical-source consumers and original failed evidence remain retained.
