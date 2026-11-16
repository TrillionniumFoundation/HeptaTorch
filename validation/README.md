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
independent test oracle. Its test program, now in `tests/pytorch_oracle.py`, serializes inputs and invokes
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

## Expanded compiler profile

`compiler-profile.json` records the distinct local compiler configuration with
256 functions, 2,560 total blocks and a 2 MiB compiler workspace. It retains the
upstream compiler/source pins, exact counted adaptation, verified import closure,
complete relocated table map, independent review and native reproduction.
The original compiler evidence remains historical and is also the explicit
predecessor boundary for this configuration.

Four native compiler generations and all packed executables match exactly. A
modified compiler source produces a different stable successor, and the new
compiler reproduces its HIA packer. The current-application subsection binds the
final runtime source `4c1d1724…` and binary `ecca265b…`; both original and expanded
compilers produce identical HeptaTorch and unhex bytes. Earlier application
snapshots in the native qualification subsection keep their own identities.

The portable `tools/compiler-profile/check_capacity.py` gate has 16 pinned-input
cases. It checks real execution at and beyond the old limits, exact new limits,
overflow refusal, retained instruction/SSA/per-function guards and high-index
call cycles. Separate records cover all 256 scalar exports, source/output/work
guards, dirty buffers and the high-index call-graph path. It does not claim to
fill every relocation slot or replace the upstream full cold campaign.

CI regenerates and compares both source-authoring receipts, reproduces the
native profile, runs the capacity gate, requires old/new application byte
identity, then runs the unchanged 275-pair native and 159-case real-PyTorch
gates. A fresh hosted record is required for this changed build procedure.

Observed expanded-profile hosted result: **PASS**.
[Run 43743468220](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43743468220)
tested commit `2bb11186c9e2bffaddd4d0e0fc88cf9997f8b52d`. The original bootstrap,
expanded four-generation native reproduction, exact source receipts, all 16
capacity cases, original/expanded application byte identity, 275 native packet
pairs and all 159 real-PyTorch cases (8,471 values) passed. All 11 oracle origin
checks passed. `hosted-compiler-profile.json` preserves the complete reports and
binds the expanded compiler hash to that run. This receipt update changes no
compiler recipe, runtime source, test definition, tolerance or fixture bytes.

## Tensor composition and indexed accumulation

**Executed native reviews: PASS. New complete hosted differential gate: NOT_RUN
in this receipt.** [Tensor composition evidence](tensor-composition-native.json)
binds the complete per-case records to source
`264e9c5c4e8ea1834724cc0d8d859952016daffe4af330a8daa7745e7a5e5233`
and monolithic executable
`5b992e5f07fe1167aacf31d9fd07a053b3b02fc1bd1149c06f8ccd164ef96bb0`.
The build contains 143 functions, 1,783 blocks and 7,064 instructions. These
reviews cover commands 34–37 (`cat`, `stack`, `scatter_add`, `index_add`) and
their interaction with existing native operators, saved versions, repeated
backward and graph reset in the current CPU binary32/rank-at-most-four profile.

| Evidence group | Cases | Value comparisons | Reference and execution |
| --- | ---: | ---: | --- |
| Independent composition review | 64 | 3,738 | Native monolithic image against independent NumPy composition, analytical VJP and SGD |
| Cat/stack semantics and layout | 81 | 1,673 | 73 semantic cases/107 values plus eight layout cases/1,566 values; independent NumPy and analytical references |
| Cat/stack failure atomicity | 16 | Separate state-preservation assertions | Instrumented native harness, with production functions preserved and a test-only dispatcher |
| Cat/stack hosted layout replay | 8 | 924 | Native monolithic image compared with actual recorded pinned PyTorch values and metadata |
| Scatter/index addition | 100 | 1,553 | Independent coordinate formulas and explicit admission/refusal expectations |
| Indexed-addition supplemental review | 9 | 1,652 | Shared transpose aliases and binary32 special-value arithmetic |
| Index-add alpha replay | 6 | 102 exact binary32 values | Native monolithic image compared with actual recorded pinned CPU PyTorch bits |
| Indexed boundary replay | 12 | 64 | Actual recorded PyTorch forward/backward results, exceptions and gradient shapes |
| Mixed new-operator graph review | 3 | 102 | Independent full-chain forward calculation and central finite differences for eight leaf coordinates, plus rejection and untracked-source controls |

These are separate suites with different case definitions and units. They are
not added into a count of unique upstream PyTorch tests. The 16 atomicity cases
use harness image `56daa05dc3fbd005f6f48e4052b13575fbc2a12646161d19bc70d33a244e6058`:
all production functions except its dispatcher are byte-identical, the original
dispatcher remains available as `command_base`, and the extra opcode 99 is
absent from production. A producer report's historical “pending layout” text is
retained; the later eight-case hosted layout replay supplies that specific
comparison without claiming the new complete main oracle ran.

The [original boundary observations](hosted-boundary-observations.json) retain
their own CI commits, run URLs, exact oracle provenance and original JSON-line
hashes. The twelve indexed observations, twenty-two empty-shape observations,
six alpha observations and eight layout observations remain **OBSERVED** at that
layer. Replaying them on the current native image is a separately recorded
operation. Their signed-i64 payloads were parsed as arbitrary-precision integers;
the value `9223372036854775807` is preserved exactly.

### Retained alpha failure and correction

The first index-add candidate separately rounded multiplication and addition at
every rank. Its [original failure report](tensor-composition-alpha-failure.json)
is preserved byte for byte: three of six cases failed, with 67 mismatching
binary32 values among 102 compared values. The old source and executable hashes
remain in the aggregate receipt; the failed image is not relabeled as the
current one.

The [focused kernel diff](tensor-composition-alpha-kernel.diff) records the
authoring change that emits the native HIA update. The emitted old and current
`indexed_add_forward` functions have hashes
`d6f95963aa6467898b2cb0a44554a46ee5e95af18d8ce2c4ea15d68c995714e1`
and `f5743c8f0dece54ebe8abf574c786cef38a066ba44236fa3decaf307b68404d5`.
Rank zero and one retain separate multiply/add rounding; rank above one uses
`fmaf(alpha, source, current)`. All six recorded cases and 102 exact values then
match the current image using the same observations and comparison rules. The
whole current source includes other integration changes as well; the focused
diff does not claim the complete old and current runtimes differ only by that
function. No old full runtime is vendored in this evidence set.

This rounding rule is qualified for the locked `2.15.0+cpu` wheel and recorded
Linux x86-64 CPU/ISA execution. It does not establish identical rounding across
all upstream devices, CPUs, vector widths, schedules or future builds.

### Empty-stride correction and packet corpus

All 275 requests from baseline commit
`31d5b1483714b32254730ba56787884c5c76e2fb` are byte-identical. Of their expected
responses, 274 are unchanged. The response for
`session-implementation-reset-empty-owned-leaf` changes exactly one decoded
byte, offset 280 from `0` to `1`, correcting shape `[2,0]` strides from `[0,1]`
to `[1,1]`. The request, stored-file hashes, decoded-byte hashes and original
pending-review record are all retained in the aggregate receipt. The correction
follows the pinned [contiguous-stride rule](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/c10/core/TensorImpl.h) of multiplying by `max(dimension,1)`,
also checked against the recorded empty-shape examples. That probe does not
contain a direct `[2,0]` creation row; the evidence does not imply it does.

The staged corpus now has 461 input/output pairs: 275 baseline pairs, 81 cat/stack
pairs, 100 indexed-addition pairs and five additional mixed-composition packets.
The aggregate receipt retains the complete current file inventory and its
deterministic hash. [Build integration](build-integration.json) separately records
the actual canonical 461-pair monolithic gate passing. It also preserves the
initial hex-transport failure and correction: an encoded request exceeded the
decoder wrapper's input capacity; complete byte-pair chunks are now decoded by
the unchanged native decoder, with neither fixture bytes nor runtime bytes
changed for that fix. Packet counts remain distinct from numerical review case
counts.

The new main oracle is configured for exactly **268 cases = 159 existing + 35
cat/stack + 74 indexed-addition**. Its script hash is recorded in the aggregate
receipt. A complete successful hosted result for that changed gate is not
asserted here; historical passing runs and the small probes keep their original
source identities. Linked-module image qualification is recorded separately;
this receipt binds its production numerical reviews to the monolithic image.
The upstream denominator remains 2,590 native schemas and
`full_rewrite_complete` remains false.

## Native borrowed modules and dual build integration

`native-modules.json` binds the native module compiler, native linker, verified
upstream authoring closure, three identical native compiler generations, source
partition, object relocations and actual linked runtime. Its complete reports
record 69 native boundary cases, 204 independent contract cases, 19 independent
alias/effect/lifetime cases, 18 guarded flat-ABI cases and two ordinary-mode byte
comparisons. The current 143 function bodies are preserved in independent units
of 63 and 80 definitions, with 52 borrowed imports and 1,108 symbolic PLT32 call
relocations. The linked image is
`cf78556c3c512227ffe123635b76e5a6af97697606e8ba3535c56f5101f4f80d`.

`build-integration.json` records a fresh rebuild through the canonical portable
scripts and the actual Makefile. Both the default linked image and monolithic
reference image pass all 461 packet pairs through the native HIA hex decoder.
`fixture-manifest.json` binds all 922 text files, the current source and both
executable identities. `oracle-extension-custody.json` records unchanged Python
ASTs for all prior oracle functions and serializer methods; new case families
run after all prior case families.

The module profile supports checked borrowed buffers. Its contract explicitly
leaves owned-resource transfer, allocation, resource results, arbitrary foreign
pointer provenance, machine-code authenticity and whole-program stack cost
unqualified. These finite suites do not close the full resource-ABI gate or any
whole-PyTorch acceptance gate.

## Observed dual-build tensor-composition result

**PASS:** [Run 43788929713](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43788929713)
tested commit `724cbf2eb19d2ef8d400615d7196f5397cb402f6`. The native-linked and
monolithic builds each passed all 461 production packet pairs, all 268 uniquely
named real-PyTorch cases and 12,932 value comparisons, with zero failed cases.
All 11 exact oracle-provenance checks passed for each build, and both executable
hashes were unchanged during the run. Compiler reproduction, the historical
legacy witness, the 16 capacity cases and the 69/204/19/18 module suites passed.

`hosted-tensor-composition.json` contains both complete oracle reports, actual
build/module gate results, step conclusions and exact source/oracle/fixture
identities. Re-serializing each captured report reproduces its hosted file
SHA-256. It adds the actual hosted outcome after the earlier native-review
receipt, which continues to record that its own main-oracle execution was
NOT_RUN at assembly time. Historical failed and successful receipts are not
rebound to this source. This evidence update changes no runtime, compiler,
workflow, numerical case, tolerance or fixture bytes.

## Combined native training through composition and indexed accumulation

**PASS:** [Run 43791020627](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43791020627) executed commit `91665f48c95f3bd8bb9bf4ec725c4f30ca66b511`. Both native-linked and monolithic HIA executables passed 461 native fixture pairs and 270 independent pinned CPU PyTorch cases (13,076 compared values per build, zero failures, all 11 wheel provenance checks). The two new 12-step sessions combine shared differentiable cat/stack, transposed views, duplicate `index_add`, MSE, optimizer updates (SGD/AdamW) and repeated `graph_reset`; 68 and 76 observed comparison values per build matched their real PyTorch references with zero maximum absolute error. `hosted-cross-feature-training.json` preserves both complete 270-row reports, their source/executable identities, per-case tolerances and the original run identity. Older 268-case and 159-case receipts remain unchanged. This is a stronger integrated native regression gate, not full PyTorch schema or subsystem completion.
