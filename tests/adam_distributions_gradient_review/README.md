# Gradient / Adam / distribution owner: independent finite review

The final executions bind exactly to core source
`3e6e879444e2795d824ec074b2bc8ded4ff058c68382c6752e1af13c5a432401`
and native image
`4ed8d68613fcec6b9b236ecdad47ea79b651988a8848b360da1d39fa2f5723be`.
This is the 428-function, 576-byte-descriptor, nine-module candidate with internal
state formats `HTGAD002` / `HTOGAD02`. The public request remains `HTORCH03`.

## Executed results

| Suite | Actual result | Evidence |
| --- | --- | --- |
| Six finite combined-owner contracts | 6 PASS; 319 interactive frames, 477 exact assertions, 138 raw binary32 words, 14 complete MT states, 6,990,240 owner bytes checked for rollback, 94 committed descriptor cleanup inspections | `validation/adam-distributions-gradient-review/reports/native-cross-04.json` |
| Two actual Torch mixed-zero/nonzero cases | 2 PASS; 106 frames, 542 exact comparisons, 204 raw words, 30,336 complete MT-state bytes; every recorded forward and gradient layout matches | `validation/adam-distributions-gradient-review/reports/actual-zero-merge-04.json` |
| Twelve actual Torch scalar-add pairs, four paths each | 12 pairs / 48 variants PASS; 408 frames, 864 exact comparisons, 1,848 raw words including before/after NaN representations | `validation/adam-distributions-gradient-review/reports/actual-scalar-add-nan-04.json` |
| Original two-case mathematical reproducer | Both creation orders PASS; 34 frames | `validation/adam-distributions-gradient-review/reports/zero-merge-mathematical-04.json` |

These are finite contracts and reference observations, not new API coverage
counts. The two mathematical cases repeat the defect recipe already covered by
the actual two-case reference. The whole 2,590-schema PyTorch rewrite remains
open. No Torch was installed or executed by these local replay scripts; their
references are the original bytes produced by the pinned hosted wheel with all
11 provenance checks passing.

The six combined contracts exercise real assigned strided gradients, Adam versus
SGD momentum layout, private optimizer roots across public drops and reset,
gradient bindings to actual Adam state, op41 fresh-zero gradients, held-seed
gradient aliasing, partial backward followed by partial Bernoulli, partial
dropout with an overlapping assigned gradient, exact RNG state, and ordinary
whole-frame rollback. Python provides only wire transport, observation, and
independent test arithmetic. All application execution is native HIA.

## Failures are retained

Candidate01 originally passed the six finite controls but failed the new
mixed-zero/nonzero mathematical reproducer. The logical derivative was
`[1, 4, 2, 5, 3, 6]` instead of `[1, 2, 3, 4, 5, 6]` when Bernoulli's fresh
strided zero contribution arrived first. The physical incoming buffer and the
logical work cache had shared storage despite incompatible layouts.

Candidate02 separated those buffers and passed the original mathematical
reproducer and six controls. The subsequent actual two-case reference still
failed both cases because multiplication and the final addition returned
strides `[2, 1]` instead of `[1, 3]`. All eight such before/after stride
differences remain in the report. No field was removed from comparison.

The scalar-add actual reference exposed a separate command95 defect:
candidate02 failed 44 of 48 variants, including NaN payload/sign propagation and
the sign of the invalid result from opposite infinities. Candidate04 implements
the observed rules and retains the original numerical helper for all remaining
inputs. Every NaN word is compared exactly; there is no normalization or
tolerance.

The initial six-control harness also had one test-author error: command58's
completed-node count includes the successful leaf accumulator. Only that count
was corrected from three to four. The original failing execution, executed
script and focused diff remain preserved separately from the real runtime
defects. An isolated helper-harness authoring attempt was also rejected for a
missing real callee body; the corrected harness supplied the actual helper, not
a stub, without changing checker limits.

Both archives retain exact raw transport, source and image identities, executed
test sources, and the historical failures. Their manifests were verified against
every archive member's byte length, SHA256 and mode. Original JSON report bytes
were copied without reserialization.

## Source review and implementation attribution

The incoming physical-cache correction changes four real incoming/work helpers
and four private-format tags from candidate01, adding two reservation helpers.
The final pointwise and scalar-add correction changes only `binary` and
`scalar_add_nograd` from candidate02 and adds six helpers. All other 420 original
function bodies remain byte-identical. Binary authoring reverses exactly to its
frozen parent after removing output-layout selection and restoring its old
linear destination address. The scalar-add caller reverses exactly after one
callee substitution.

The review agent authored the small command95 NaN helper after obtaining the
actual reference. Its isolated real-HIA kernel matched all 924 output words
from the 48 variants. The composition author incorporated that exact helper and
caller, checked by SHA256 and full byte equality. This implementation contribution
is disclosed; the helper unit test is not represented as an independent source
audit of code authored by another person. The official reference, independent
wire replays, source bindings, and peer source review remain separate evidence.

This review does not prove all TensorIterator layouts, all NaN behavior in other
operators, all CPU instruction-set dispatches, hooks, higher-order gradients,
other dtypes/devices, or the entire PyTorch API. The native protocol also retains
its explicit ordinary-frame transaction and structured partial-error contracts.

A separate source-review question remains pending actual observation: an
admitted size-one axis can have an enormous element stride (for example
`2^61`), while upstream TensorIterator compares signed byte strides after
multiplying by the element width. That boundary is being probed independently;
the present finite results neither establish nor dismiss compatibility there.

## Portable replay

The scripts accept explicit source, executable, wire and reference paths. They
do not import production authoring, implementation code, or native-derived
expected values. Output directories must not already exist.

Common arguments for all four scripts are:

```sh
--binary build/gradient-adam-distributions/tensor-gradient-adam-distributions-session \
--source build/gradient-adam-distributions/runtime.hepta \
--protocol tests/adam_distributions_gradient_review/protocol.py \
--expected-native-sha256 4ed8d68613fcec6b9b236ecdad47ea79b651988a8848b360da1d39fa2f5723be \
--expected-source-sha256 3e6e879444e2795d824ec074b2bc8ded4ff058c68382c6752e1af13c5a432401 \
--output NEW_OUTPUT_DIRECTORY
```

Run `test_joint_gradients_v2.py` with the additional argument:

```sh
--distribution-observations tools/cpu_distributions/evidence/torch-bernoulli-dropout.json
```

Run `replay_zero_merge_layout.py` with:

```sh
--observations validation/adam-distributions-gradient-review/actual/torch-zero-merge-layout.json \
--expected-observations-sha256 72150ed1d2a1792f2d16be85b4b3bd323bafa8b20d2a6b7f23218682eda63f43
```

Run `replay_scalar_add_nan.py` with:

```sh
--observations validation/adam-distributions-gradient-review/actual/torch-scalar-add-nan.json \
--expected-observations-sha256 e664c9ff717a104b8c7317dfc55677e0248e58f981cb8315eacb3975049048ff
```

`test_zero_merge_layout.py` needs only the common arguments. The two pure Torch
probes are separately included with their original source hashes; their CLI is
`--reference --lock --repo --output [--prepare-only]`. Prepare-only records
`PREPARED_NOT_RUN` and does not execute Torch.
