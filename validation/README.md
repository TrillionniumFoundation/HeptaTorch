# Validation evidence

The original native implementation is bound by `native.json` to both its HIA source
and executable SHA-256. It includes 123 tensor/integration cases, 17 optimizer
cases, seven spatial/training integration cases, 894 scalar special-value
vectors, and 19 independently reviewed adversarial groups. These counts describe
different sets and units; they are not a count of upstream PyTorch tests.

The native packet corpus contains 275 input/expected-output pairs. Its exact
files, sizes and hashes are recorded in `fixture-manifest.json`. `make test`
executes all of them through the native HIA hexadecimal decoder and tensor
executable with empty PATH. Responses were recorded only after independent
numerical/analytical/finite-difference checks or explicit rejection checks.

`components.json` retains the larger component campaigns, their seeds, reference
methods, numerical errors, and distinct executable identities. They include the
2,314,020 basic arithmetic comparisons, 3,424,359 elementary-function comparisons,
20,025 high-precision controls, and 144 spatial gradient finite-difference
coordinates. Component adapter results are separate from the final integrated
executable's observations.

`bootstrap.json` records a fresh compiler bootstrap. The native compiler's source
and binary pins are in `compatibility/hia-toolchain-lock.json`; the full recipe is
the executable GitHub workflow. The finite bootstrap passes do not claim the
separate upstream full cold-qualification suite or a formal compiler proof.

## Independent review findings

Four concrete implementation issues were found and corrected before publication:

1. Empty gradient reads bypassed nonleaf/requires-grad validation.
2. A detached alias could modify a saved output used by backward.
3. A squared denominator caused avoidable extreme-value division-gradient errors.
4. Adam's early gradient squaring and early variance debiasing could suppress
   valid large-gradient updates.

The final review reran malformed inputs, invalid IDs, shape/payload bounds,
empty tensors, storage aliasing, stale versions, graph isolation, fan-out,
repeated backward, optimizer state/configuration contracts, large gradients,
and descriptor/workspace/output/native-wrapper limits. All reported groups
passed against unchanged source and executable hashes.

## Original hosted PyTorch differential gate

**Observed hosted result: PASS.**
[Run 43738165041](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43738165041)
tested commit `4cf01bee1a6961a423bf2054dc303fb43fa0f33b`. Compiler bootstrap,
native build, all 76 native packet pairs, all 76 live PyTorch differential cases
(2,754 values), and all 11 exact oracle-provenance checks passed. The native
executable was unchanged during comparison. `hosted.json` retains the complete
per-case results, tolerances, wheel provenance and hosted bootstrap receipt.
Documentation-only evidence updates after that run leave the tested runtime,
compiler recipe, oracle code and fixtures unchanged.

The CI workflow installs the pinned **CPU PyTorch 2.15.0+cpu** only as an
independent test oracle. Its inline test program serializes inputs and invokes
real PyTorch forward operations, autograd, and optimizers. It never implements
HeptaTorch's runtime operations. The original gate required exactly 76 cases, zero
failures, the exact wheel version, source revision and artifact hash, and an
unchanged native executable. Numerical tolerances are recorded per case.

`compatibility/pytorch-oracle-lock.json` fixes the official Python 3.12 Linux
x86-64 wheel URL and SHA-256. Pip verifies the downloaded wheel against that
hash. The installed distribution's source URL and archive hash, exact package
version, CPU backend and build Git SHA are checked independently of numerical
comparisons.

The inventory release commit is `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`.
The official wheel was built at its sole parent,
`d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`. The release adds only five lines of
Ruff lint configuration and comments to `pyproject.toml`. Comparing the complete
74-entry root trees established identical Git objects for every other entry,
including every source, generator, test and third-party subtree. Both pins and
the exact source-equivalence evidence are retained in `hosted.json`.

The [first hosted run](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43737347959)
passed compiler bootstrap, 76 native packet pairs and all 76 real-PyTorch
numerical cases (2,754 values). Its overall result remains **FAIL** because the
original gate incorrectly required the wheel build SHA to equal the release tag
SHA. That failed result is retained in `hosted.json`. The exact provenance
correction passed the fresh run linked above without changing the numerical
test definitions, tolerances, seed, native implementation or executable.

Local execution lacked PyTorch and remains recorded as NOT_RUN for that oracle.
A hosted result is tied to its tested GitHub commit and run URL; none of these
scoped checks establish whole-schema or whole-project parity.

## Previous native expansion

`native-extension.json` binds the previous operator-expansion source and executable.
All 221 fixture pairs passed, including the 76 unchanged baseline pairs and
145 new pairs. The final binary also passed 63 matrix/Linear cases (4,593
values, including eight central finite-difference cases), an independent
2,514-case matrix shape/gradient/admission review, and 11 cross-feature cases
(529 exact comparisons). The peer review includes 529 valid matmul cases,
1,263 shape refusals, 720 Linear cases and two capacity refusals.

The 44 seeded-backward checks, 67 index_select cases, 72 gather cases and pooling
checks in the component section were executed on isolated component binaries.
Their original hashes remain attached; their production regression packets
were subsequently replayed on the final integrated executable. The test-only
error-inspection opcode used for three gradient atomicity probes is absent
from production source and fixtures.

The actual runtime now implements commands 29–32 and graph operations 25–27.
The legacy rank-two matrix command delegates to the generalized kernel, and its
superseded forward/reverse functions were removed. The native build has no
Python or tensor-library dependency. The initial external compiler bootstrap
and independent numerical oracle retain their existing, separate dependencies.

That revision of the hosted gate required exactly 153 uniquely named real PyTorch cases,
including the new operations, strided explicit seeds, repeated gradients and
three attention compositions without masking or dropout. It preserves the
original tolerances and test definitions for all 76 baseline cases and adds
explicit tolerances for the new cases. `hosted-extension.json` records its actual
execution status and exact commit; the historical passing run above does not
qualify the changed source. The source/image and complete fixture manifest are
also checked during hosted build.

Observed operator-expansion hosted result: **PASS**.
[Run 43740237504](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43740237504)
tested commit `2bfb4bac21361a5057da1eeee9349fbcb8b3cba7`. Compiler reproduction,
221 native packet pairs, the exact source/image/fixture manifest and all
153 real-PyTorch cases (8,244 values, zero failures) passed. All 11 oracle
provenance checks passed. The native executable hash equals the locally tested
image and remained unchanged during the comparison. All three attention
compositions passed their forward and Q/K/V-gradient comparisons.

The full report and per-case tolerances are retained in `hosted-extension.json`.
This evidence update changes documentation and receipts only; it leaves the
runtime, test workflow, compiler recipe and fixture bytes unchanged.

## Current training graph lifetime

`session-native.json` binds the current integrated source and executable. All
275 native packet pairs pass: 221 previous request inputs remain byte-identical, five previous output files
are independently requalified after the Adam rounding correction, and 54
new pairs cover graph reset, generation-checked IDs and retained state. The new
pairs comprise 14 implementation cases, 12 independent invariant cases and 26
independent public-protocol cases, plus two 512-step rounding regressions. The original isolated component identities
remain attached; replay results are bound to the final binary separately.

The final independent public-protocol review passes 26 cases and compares
40,123 numeric values, including 1,000 SGD and 1,000 Adam updates in single
requests. SGD matches an independent binary32 recurrence exactly; the maximum
Adam discrepancy is 9.089708328e-7 against independently computed state and bias
correction. A retained 20,000-element tensor also survives 20 backward/reset
iterations that allocate 9,600,000 cumulative transient payload bytes within
the unchanged 1 MiB workspace. Both 1,000-step optimizer requests fit the HIA
hex decoder and are durable native fixtures.

Ten further native rejection checks compare all 1,048,568 workspace bytes except
the eight-byte status field. None changes after rejected reset. The scratch-only
entry harness preserves every one of the 120 nonentry HIA function bodies from
the final production source. Boundary-generation injection is confined to that
harness; no debug opcode or artificial test state is shipped in the runtime.

The hosted gate now requires exactly 159 unique real-PyTorch cases. It retains
the previous 153 case definitions and tolerances, adds six graph-lifetime cases,
and checks the exact current source, image and 275-pair fixture manifest. The
six additions cover 1,024-step SGD, 512-step Adam, 512-step AdamW, gradient
accumulation, an empty retained prefix and an empty retained tensor. Their
oracle results must come from a new hosted execution; earlier receipts qualify
only their own recorded runtime identities.

### Adam rounding failure and correction

The first 159-case hosted run failed in two new 512-step Adam/AdamW trajectories,
with maximum absolute discrepancies of 0.000664085 and 0.000812143. Its complete
report, exact source/image identity and successful native/build/provenance
checks remain in `hosted-session-failure.json`; that run is not relabeled.

The runtime correction uses stable lerp for the first moment and software fused
multiply-add for the final product/addition in the second moment. The software
FMA passed 485,163 independent exact binary32 comparisons, and both 512-step
native sessions reproduce all eight originally published PyTorch reference
values exactly. The original 17-case optimizer reference (320 values), 18-step
network reference (56 values), four large-gradient cases, 26 lifetime cases and
ten workspace refusal probes were rerun against the corrected binary. All
passed with their original numerical thresholds.

Six pre-correction golden responses changed; each was independently requalified
before updating. This includes five files from the pre-existing 221-pair corpus
and one added 1,000-step Adam response. Every earlier request input is preserved.
Two 512-step native regression pairs bring the corpus to 275 pairs. The real
PyTorch oracle's 159 cases, hyperparameters, random sequence and tolerances are
unchanged. A fresh hosted run must qualify the corrected source.

### Corrected hosted result

**PASS:** [Run 43742636363](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43742636363)
tested commit `6d21e076f15fdd097a5715f9e58306e93b257ce1`. All 275 native packet
pairs, the source/image/fixture manifest, compiler reproduction, 11 provenance
checks and 159 real-PyTorch cases (8,471 values) passed. The 1,024-step SGD case
agreed exactly; maximum Adam and AdamW absolute discrepancies were
1.4901161193847656e-8 and 5.960464477539063e-8. The binary was unchanged during
comparison. `hosted-session.json` retains the complete result and connects it
to the original failed run. This evidence update changes receipts and prose;
it leaves the runtime, compiler recipe, numerical oracle and fixture bytes
unchanged.
