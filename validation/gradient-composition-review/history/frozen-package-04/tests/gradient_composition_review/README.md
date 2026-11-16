# Independent native review of the unified gradient runtime

This review covers native compilation and the owned-resource boundary of the
unified public-gradient, Adam and CPU-distribution runtime. It is separate from
the producer's tests, the independent numerical cross-review, and the fixed
PyTorch observation replays. The whole PyTorch rewrite remains **HOLD**.

The current source, executable, native toolchain and test identities are bound by
`profile.json`. A successful report applies only to those exact bytes. A previous
candidate's results do not qualify a replacement source or executable.

## Candidate 04 result

All three native gates passed for source
`3e6e879444e2795d824ec074b2bc8ded4ff058c68382c6752e1af13c5a432401`
and executable
`4ed8d68613fcec6b9b236ecdad47ea79b651988a8848b360da1d39fa2f5723be`.
The independent rebuild bound all **428** complete core function bodies exactly
once to seven core modules, alongside the frozen RNG and owned-entry modules.
All nine rebuilt objects and the final executable matched byte-for-byte.
Owned-entry checks compared **17,475,600 bytes** of complete state. Input sources,
tools and executable hashes were rechecked after execution.

The original reports are under
`validation/gradient-composition-review/candidate-04/`; raw observations are in
the three separately verified archives under that directory's `evidence/`.
The top-level native summary is `qualified-combined-04.json`.

**An additional stride boundary remains open.** The source review found that a
size-one dimension can legally carry an enormous positive element stride while
its storage remains small. The upstream iterator compares signed byte strides
after multiplying by four; this candidate compares unsigned element strides.
Byte scaling can cross the signed boundary or wrap to zero. The fixed backend
behavior has not yet been observed for that domain. The eight-case independent
`pytorch_extreme_stride_probe.py` is frozen with a `PREPARED_NOT_RUN` receipt;
it has no native-derived expectations. `source-review-boundary-addendum.json`
narrows the earlier pointwise source conclusion to nonnegative byte strides
representable in int64. This does not alter the three observed PASS reports or
the separately reviewed command 95 NaN fix.

## What the three gates establish

| Gate | Actual checks | Limits of the conclusion |
|---|---|---|
| Source-to-unit/native reproduction | A small independent line scanner verifies that every complete runtime function body occurs exactly once and byte-identically in the core units. The separate RNG and owner units have fixed hashes. The actual native compiler rebuilds all objects; the actual native linker rebuilds the complete executable. Every output byte must equal the producer's pinned output. The resulting ELF has no dynamic interpreter or dynamic program header and exits successfully on EOF. | The scanner establishes byte custody. Native compiler/linker checks provide semantic admission within their declared ABI3 producer profile. This does not prove the numerical correctness of the source bodies or authenticate arbitrary machine code. No producer authoring module or host HIA parser is imported. |
| Owned entry/capacity | A frozen HIA probe creates real owned buffers and calls the checked native `step`. Four cases exercise public-gradient assignment, republishing a still-owned gradient, clearing its final private binding, and Adam followed by a late dropout error. Output capacities below the complete reply must return the original owner byte-for-byte; exact capacities must agree with the interactive native execution. Ordinary suffix errors remain atomic. The explicit late-error command commits its prefix and stops its suffix. | Copying observed state bytes into a real qualification owner is an ABI test, not a foreign-pointer import or a claim that arbitrary fabricated owned state is authenticated. The interactive response is a transport/state comparison, not an independent numerical oracle. |
| Typed owner handoff | Complete real units from both parents and the current candidate are namespace-renamed, compiled and linked together. Renaming has an exact inverse. Both directions must reject the other state format with status 7 while preserving the same pointer, length and all bytes. The correct producer subsequently recovers its state; explicit close removes the owner mapping. | This establishes the tested owned handoff contract between pinned producer profiles. It does not generalize to arbitrary external native owners. |

The entry gate has four cases, 25 raw native probe processes and 16 interactive
frames. The handoff gate has two real parent profiles, four directions and 18
frames. The complete byte count is recorded in each actual execution report.

## Reproduction

Build the pinned native compiler/linkers, the frozen atomic distribution tools,
the two parent runtimes and the current unified runtime first. Then run:

```sh
python -B tests/gradient_composition_review/qualify.py \
  --build "$CURRENT_BUILD" \
  --stack-build "$STACK_BUILD" \
  --atomic-tools tools/cpu_distributions \
  --gradient-build "$INCOMING_GRADIENT_BUILD" \
  --adam-distributions-build "$ADAM_DISTRIBUTIONS_BUILD" \
  --output "$REVIEW_OUTPUT"
```

`REVIEW_OUTPUT` must not already exist. `STACK_BUILD/native` contains
`stack-module-compiler`, `stack-linker-session` and `stack-linker-runtime`.
The atomic tools provide the frozen `entry_probe.hepta`, `state_handoff.hepta`,
`protocol.py` and `owner_client.py`; these are qualification helpers, not the
production numerical implementation. Every target operation executes in the
native HIA executable. The test processes use Python only to encode requests,
observe responses and verify source/output custody.

The three validators can also be run separately; `--help` lists their exact
arguments. `typed_handoffs_gad002.py` is derived from the original frozen handoff
test by changing only the current executable basename, current state tag and
report schema label. The original bytes and the exact inverse transformation
are included. Packet contents and error expectations are unchanged.

## Preserved failure history

Candidate01 used source
`9fa3a32a761d4f177d742ff48aa850a467a6fc5b62d7cc9325f60bbefec47af8`
and executable
`f8446e50cd0cb43b90eb7b7dd133e8edff1f4a8cbad112a395c7e7b98fd43b75`.
All three gates above passed for those bytes. A separate independent numerical
test then found a genuine error: a transposed zero gradient arriving before a
nonzero gradient merged the latter in linear cache order. Logical expected
values `[1, 2, 3, 4, 5, 6]` became `[1, 4, 2, 5, 3, 6]`. Reversing branch creation
order passed. The candidate is therefore recorded as **BLOCKED**, even though
these three narrower checks passed. `historical-candidate-01.json` binds the
original reports and the separate failure report.

The subsequent generic cache correction requires a new source/executable and a
new state contract, `HTGAD002` / `HTOGAD02`. Its qualification must be executed
again. Candidate 04 was actually rerun with this contract as recorded above.
Neither the old results nor a source-only inspection substitutes for that
execution. Intermediate candidates 02 and 03 are not qualified by these gates;
their numerical and source-change history remains in the separate producer and
numerical-review evidence.

`scalar-add-nan-source-review.json` is a separately labelled read-only review of
four fixed official PyTorch files and Intel's published NaN rules. It explains
why scalar/vector double-NaN precedence and invalid Infinity results require
actual backend evidence. It executed no native test and no live PyTorch test;
its architectural predictions are not installed as oracle expectations.

`audit_source_changes.py` and `source-review-combined-04.json` bind a later narrow
review of the five pointwise helpers, the actual emitted binary store, and the
13-line command 95 helper. The helper contributor is a different agent; this
review checks its exact HIA bytes and the one-callee inverse transformation
independently. Its special-value rules agree with the fixed actual backend
observations, using the admitted float32 Scalar bits. This source review itself
executes **zero** native tests and **zero** live PyTorch tests. Its pointwise
conclusion must be read with the stride-boundary addendum above.

## Evidence custody

Original reports are preserved without rewriting paths or observations.
Large raw outputs are stored in separate gzip archives so each can be published
without a large-request transport workaround. `archive_evidence.py` streams the
archive, decodes every member, checks its SHA-256, length and mode, and then
checks the source file again. It never deletes the originals.

An archived group can be independently checked after relocation:

```sh
python -B tests/gradient_composition_review/archive_evidence.py \
  --verify-manifest validation/gradient-composition-review/evidence/GROUP/manifest.json
```

Add `--check-original-sources` only when the original capture paths are still
available. Archive verification itself does not require those old paths.

No report in this package claims full numerical, autograd, RNG, dtype, optimizer
or PyTorch API closure. Those remain separate gates and retain their original
failures and unrepresented cases.
