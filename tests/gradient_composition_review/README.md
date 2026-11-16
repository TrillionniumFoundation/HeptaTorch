# Independent native review of the unified gradient runtime

The final candidate 05 passes the three native compilation/owner gates and the
fixed eight-case extreme-stride comparison. These results are bound to source
`1fed8ea7ba5bf755a9df9ca1e61d34cfd4712cc2fd66e0728493091c81c24dae`
and executable
`6773485896d961b6c2e62784ebb66320290df47128bd1331aecc0c8b7f6611fe`.
The whole PyTorch rewrite remains **HOLD**. This package does not replace the
producer's numerical regressions, the independent gradient/optimizer review,
or the other fixed PyTorch replays.

## Observed results and limits

| Gate | Result | What it establishes |
|---|---|---|
| Native unit reproduction | PASS: 428 complete core bodies, nine rebuilt objects and final ELF byte-identical | Every core body occurs exactly once and byte-identically in the seven core units. Frozen RNG and owner units are checked separately. Actual native compiler and ABI3 linker execute; no producer authoring module or host HIA parser supplies semantic admission. EOF executes and closes successfully; the ELF has no dynamic interpreter or dynamic program header. |
| Owned entry and output capacity | PASS: four cases, 25 raw native processes, 16 interactive frames, 17,475,600 complete state bytes compared | Below-complete output capacity preserves the original owner. Exact capacity agrees with the interactive path. Public gradient assignment/republishing/clear and Adam followed by an explicit late distribution error use real checked owned calls. Ordinary errors roll back the frame; explicit partial errors commit the prefix and stop the suffix. |
| Typed owner handoff | PASS: two actual parent profiles, four directions, 18 frames | Complete namespaced parent/current units compile and link together. Both directions reject incompatible state tags with status 7 while preserving pointer, length and all bytes; the rightful producer recovers the state, and close removes the mappings. Symbol renaming has an exact inverse. |
| Extreme size-one strides | PASS: eight actual reference cases, 80 native frames, 344 exact field comparisons | The frozen actual PyTorch report and native captures agree on admission, all output/input strides, raw float32 words, offsets, storage bytes, Version, requires-grad and opaque identity relations. No tolerance or expected failure count is used. |
| Comparator source change | Source-only PASS: one changed function; 427 bodies unchanged | The final correction scales to wrapped byte strides, handles wrapped zero, then uses sign-bit XOR for signed ordering. The entire authoring edit has an exact inverse. No public stride rejection or case-specific table was inserted. |

The line scanner in the first gate establishes byte custody. It is not an
arbitrary-machine-code verifier. Native admission is limited to the declared
ABI3 producer profile. Copying observed bytes into a real owned qualification
buffer tests that ABI; it is not a foreign-pointer import or authentication of
arbitrary fabricated states. Process-specific Tensor/Storage IDs are compared
through their identity relations, not by matching numeric addresses.

## Reproduce the three native gates

Build the pinned native tools, atomic distribution helpers, two frozen parents
and candidate 05 first. `STACK_BUILD/native` must contain
`stack-module-compiler`, `stack-linker-session` and `stack-linker-runtime`.

```sh
python -B tests/gradient_composition_review/qualify.py \
  --build "$CURRENT_BUILD" \
  --stack-build "$STACK_BUILD" \
  --atomic-tools tools/cpu_distributions \
  --gradient-build "$INCOMING_GRADIENT_BUILD" \
  --adam-distributions-build "$ADAM_DISTRIBUTIONS_BUILD" \
  --output "$REVIEW_OUTPUT"
```

The output directory must not exist. `profile.json` fixes candidate 05, all three
tool hashes and every child validator. The parent builds are the frozen
incoming-gradient 355-function profile and the Adam/distribution 378-function
profile. Test code and expectations for these three gates are byte-identical to
the executions against candidate 04; only the separate profile/unit pins bind
the new source and image. No earlier PASS is transferred to the new candidate.

## Reproduce the strict eight-case comparison

The observation-only PyTorch probe is
`tests/pytorch_extreme_stride_probe.py`. Its hosted actual result is preserved
unchanged as `validation/gradient-composition-review/extreme-stride/torch-extreme-stride.json`
(SHA-256 `2de8fbd82d10fc16b0640684f4b36ae958a5a0455773401f82092210967137c8`).
It includes all eleven required provenance checks. Local native reproduction
does not rerun or substitute that reference:

```sh
python -B tests/gradient_composition_review/observe_extreme_stride_native.py \
  --build "$CURRENT_BUILD" --protocol tools/cpu_distributions/protocol.py \
  --prepared validation/gradient-composition-review/extreme-stride/prepared.json \
  --expected-source 1fed8ea7ba5bf755a9df9ca1e61d34cfd4712cc2fd66e0728493091c81c24dae \
  --expected-image 6773485896d961b6c2e62784ebb66320290df47128bd1331aecc0c8b7f6611fe \
  --output "$STRIDE_OBSERVATIONS"

python -B tests/gradient_composition_review/compare_extreme_stride.py \
  --reference validation/gradient-composition-review/extreme-stride/torch-extreme-stride.json \
  --native-observations "$STRIDE_OBSERVATIONS/report.json" \
  --expected-source 1fed8ea7ba5bf755a9df9ca1e61d34cfd4712cc2fd66e0728493091c81c24dae \
  --expected-image 6773485896d961b6c2e62784ebb66320290df47128bd1331aecc0c8b7f6611fe \
  --output "$STRIDE_COMPARISON"
```

Both output directories must be new. The same observer and comparator bytes
first captured and strictly rejected candidate 04, then accepted candidate 05.
The eight reference rows, plans and raw values are unchanged.

## Preserved failures

Candidate 01's three native gates passed, but an independent numerical test
found a real transposed zero/nonzero gradient merge error: expected logical
`[1, 2, 3, 4, 5, 6]` became `[1, 4, 2, 5, 3, 6]` when the zero contribution
arrived first. Its status remains **BLOCKED**. Its original reports and raw
archives are retained under `candidate-01`; a separate peer archive contains
the full numerical counterexample and original candidate image.

Candidate 04's three native gates also passed. The source review then identified
that an accepted size-one axis can carry an element stride at or above `2^61`
without requiring large storage. Upstream TensorIterator scales strides by four
and compares the signed byte-stride word. Candidate 04 compared unsigned element
strides. The independent actual probe confirmed four real mismatches among eight
cases: the output's trivial-axis stride was `6` instead of `1`. All other
compared fields matched. The strict comparator exited **1**; the original
FAIL report, complete reference, native observations, executed test sources and
stdout/stderr are preserved. The earlier source-only OPEN receipt is retained
as a historical observation, not relabelled as a failure that was already known.

Candidate 05 changes only `pointwise_compare_axes`: it forms the wrapped byte
stride, checks byte-zero ambiguity and compares signed words using XOR of bit
63. Admission, shape tie-breaking, other helpers, tensor addressing, numerical
kernels, ownership and graph code retain their exact candidate 04 bodies.
`comparator-source-review-05.json` records all 428 old/new function hashes and
the precise inverse authoring transformation. The old three-gate package and
its manifest remain immutable.

The earlier command 95 NaN review is separately labelled. Its helper preserves
the fixed backend's observed other-NaN precedence, quiets without discarding
sign/payload, and returns the negative canonical NaN for opposite infinities.
Its source reviewer did not author that helper. Source inspection itself ran
zero native tests and zero live PyTorch tests; those numerical runs belong to
the separate independent review. Both helper and caller are among the 427
unchanged bodies in candidate 05.

## Evidence custody and scope

Original reports are copied without rewriting their paths or observations.
Every raw archive has been decoded and checked member-by-member for length,
SHA-256 and mode, followed by another source-file comparison. Each archive is
small enough for the publishing transport. Originals were not deleted by the
archive tool. After relocation, a group can be verified without its old path:

```sh
python -B tests/gradient_composition_review/archive_evidence.py \
  --verify-manifest validation/gradient-composition-review/candidate-05/evidence/GROUP/manifest.json
```

`--check-original-sources` additionally requires the capture paths still to be
available. The legacy `audit_source_changes.py` applies specifically to candidate
04; `audit_comparator_correction.py` takes both actual build directories for the
source-only 04-to-05 comparison. Neither script replaces execution gates.

The admitted float32, rank-at-most-four domain remains. These tests do not close
other dtypes, devices, all TensorIterator modes, public gradient/optimizer APIs,
all distributions or the entire PyTorch surface. The existing 18 unrepresented
distribution cases and all other recorded subsystem gaps remain separate and
open. The API denominator stays 2,590; this package never changes a whole-project
completion flag to true.
