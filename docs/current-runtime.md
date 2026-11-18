# Build and qualify the current HIA runtime

The authoritative application source is
[`tools/current_runtime/source/runtime.hepta`](../tools/current_runtime/source/runtime.hepta).
Its [manifest](../tools/current_runtime/source/manifest.json) identifies every
ordered compilation unit, complete core function body and expected native
object/executable. There is one default application profile, named `current`.
The present candidate has 729 core functions and 14 units. Its combined
fifteen-group / 1020-case qualification is pending; no earlier runtime PASS
is transferred to this candidate.

Tensor arithmetic, shape planning, allocation, reference lifetime, gradients,
RNG and command execution run in HIA machine code. Standard Python organizes
builds and independent tests. The produced static Linux x86-64 executable has
no Python, PyTorch, NumPy, libm or libc execution dependency. Bootstrap of the
HIA toolchain remains a separate reproducibility boundary.

## Source validation

```sh
python3 -I -S -B tools/current_runtime/build.py \
  --manifest tools/current_runtime/source/manifest.json \
  --source-root tools/current_runtime/source \
  --check-only --output build/current-source-check
```

The output directory must be new. This verifies source hashes, every complete
function body, imported/exported signatures and effects, acyclic dependencies,
module counts and compiler limits. It runs no native compiler or application;
the result is explicitly `SOURCE_CHECKED_NATIVE_NOT_RUN`.

## Cold native build

The default Make target requires GNU Make 4.3 or newer:

```sh
make HEPTA_SOURCE_ROOT=/absolute/path/to/fixed-hepta-source \
     HEPTA_STACK_BUILD=/absolute/path/to/stack-budget-tools \
     HEPTA_ENTROPY_BUILD=/absolute/path/to/current-entropy-tools
```

The compiler defaults to `HEPTA_STACK_BUILD/native/stack-module-compiler`.
The four-argument session linker defaults to
`HEPTA_ENTROPY_BUILD/generator-session-linker`. Make reproduces this tool
through the fixed two-process native compiler/linker recipe, retaining the
original 40 inputs and raw streams. Its actual `build.json` is mandatory;
a prebuilt linker alone cannot satisfy Generator131 admission. The
[entropy tool source](../tools/current_runtime/entropy_linker/README.md)
and hosted workflow specify its fixed bootstrap inputs.
The application is `build/current-runtime/heptatorch`; the actual build receipt
is `build/current-runtime/build.json`. Make compiles in a fresh staging
directory, then publishes a successful build. A recognized previous local
build and any failed staging attempt are retained for diagnosis. Removing
either paired output causes both to rebuild.

The standalone equivalent is:

```sh
python3 -I -S -B tools/current_runtime/build.py \
  --manifest tools/current_runtime/source/manifest.json \
  --source-root tools/current_runtime/source \
  --compiler /absolute/path/to/stack-module-compiler \
  --linker /absolute/path/to/current-entropy-tools/generator-session-linker \
  --output /new/current-runtime-build
```

Every object is compiled from its complete HIA unit in a new native process.
The builder takes private copies of the two verified tools; it never reads a
historical application ELF or reuses an application object. The native linker
receives the exact new objects through `HIACGS01`, with entry `step` and zero
startup workspace. The current four-argument `step` ABI includes the
read-only host entropy input; its owned runtime and `close` contracts
remain independently checked by the original Generator controls.
The resulting application is not executed by the builder.

| Required native tool | SHA-256 |
| --- | --- |
| ABI3 module compiler | `b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f` |
| Four-argument entropy session linker | `761a941fd7a2d53eb9b45869fcfc20e970640e9dd11d60f7c61a89812b61d74e` |

The fixed Hepta source commit is
`f33d8da316de8fdf27b504371f1a5310997ff548`. Tool reproduction is specified by
the [stack profile](../tools/stack_budget/profile.json) and
[ABI3 contract](../tools/stack_budget/CONTRACT.md). The hosted
[current-runtime workflow](../.github/workflows/current-runtime.yml)
rebuilds the tools through their native fixed points before building the
application. Its initial bootstrap uses Python and GNU ld.

The fixed compiler admits at most 1 MiB per source/object, 256 functions
including imports, 2,560 blocks, 256 blocks per function and 8,192 instructions
per unit. The linker admits 64 objects and 8 MiB input/output. The private
stack has 8 MiB with guard pages; the native path limit is 8,384,512 bytes,
leaving the fixed 4,096-byte reserve. The pinned independent stack inspector
checks object metadata and decoded frames against the actual source call
graph before linking. Actual demand is recorded in each build receipt.

A successful build reports `PASS_NATIVE_BUILD_ONLY`. This records complete
source/tool/object/image identities and actual compiler/linker exit and EOF;
it does not grant runtime or whole-PyTorch qualification.

## Current-image tests

The default test command requires the fixed native tools and a pinned live
PyTorch reference interpreter. All fifteen groups are mandatory:

```sh
make test HEPTA_SOURCE_ROOT=/absolute/path/to/fixed-hepta-source \
          HEPTA_STACK_BUILD=/absolute/path/to/stack-budget-tools \
          HEPTA_ENTROPY_BUILD=/absolute/path/to/current-entropy-tools \
          HEPTA_TORCH_PYTHON=/absolute/path/to/pinned/python
```

The reference environment requires CPython 3.12, the exact CPU wheel in
[`pytorch-oracle-lock.json`](../compatibility/pytorch-oracle-lock.json), and
the original pip installation report at its locked repository path. All eleven
wheel/build/install provenance checks remain mandatory. The interpreter runs
the test harness and the independent oracle; numerical operations in the
application run in native HIA. Use this same CPython 3.12 interpreter for the
coordinator and all original-reader stages. Their strict source-identity proofs
retain Python AST digests, whose serialization can differ across Python minor
versions.

| Group | Cases | Required checks |
| --- | ---: | --- |
| Bool sum | 27 | Original fixed observations, exact dtype contract changes, entry and resource controls |
| View/scalar | 76 | Original view26/scalar32/view14/input4, full owner and RNG checks |
| Integral sum | 24 | Original 260 events and full-width integer/overflow behavior |
| Integral-to-F32 sum | 17 | Cast before reduction, noncontiguous/empty layouts and two allocation-failure stages |
| Original lifecycle | 35 | 6,344 frames, 2,590,274 numeric values, strict owner checks, four positive and eighteen negative controls |
| Original operators/training | 270 | All ten original groups, 284 real frames and fourteen success-prefix/refusal pairs |
| Original F32 sum | 32 | All 626 original events, original comparison classes and complete native EOF |
| Live Tensor/RNG/backward | 32 | 2,132 frames, 17,455 exact binary32 words and 1,137,600 Generator-state bytes |
| Logical reductions | 147 | Any/all/Bool sum, invalid-axis gradient precedence, eighteen packet controls and two real allocation/rollback/retry controls |
| Original resource controls | 7 | Original R4 resource3, backward private-cache1 and R3 preserve3, full 84-word owner/RNG rollback and three actual allocation-limit recoveries |
| Original AsStrided | 16 | Original copy4/reduction4/lanes2/branches6, 1,920 frames, 10,272 checks, 176 complete owner archives, sixteen actual native exits and 32 EOF observations |
| Original cat/stack packets | 4 | Three complete original successful byte streams and one explicit expanded-capacity control, full owner snapshots and actual exit/EOF |
| Cat metadata | 10 | Independently observed Torch metadata and raw data, complete native frame/owner reconstruction, 62,207 repeated empty gradients and 62,208-input forward control |
| CPU Generator | 131 | Original24 + native11 + entry59 + dispatch37; exact Tensor/Generator owner and state, real entropy, unchanged controls, newly linked current-object ABI/reordered probes and actual exit/EOF |
| Public F64 logical | 192 | Original direct dtype4 dense no-gradient any/all/Bool sum comparisons; eighteen packet controls, two real allocation rollback/retry controls, 196 native children and complete frames/owner/EOF; two scope controls remain outside Torch compatibility |

These total **1020 finite comparison cases**. They are not 1020 complete upstream
schemas. Original lifecycle35, operator/training270, live32 and the original
Generator24 subset execute the actual pinned Torch oracle. Ten groups consume exact frozen actual
reference results, including all provenance checks applicable to each package.
Cat4 consumes four fixed original packet fixtures.
Cat4 consumes four complete legacy protocol fixtures. Metadata10 consumes a
separate immutable actual Torch reference with all eleven provenance checks;
its fifteen-member custody package retains original bytes and records explicit
original-path mappings without rewriting the original report.

To choose a new output directory explicitly:

```sh
/absolute/path/to/pinned/python -I -S -B tests/qualify_current_runtime.py \
  --build-directory build/current-runtime \
  --tool-build /absolute/path/to/current-entropy-tools \
  --stack-build /absolute/path/to/stack-budget-tools \
  --torch-python /absolute/path/to/pinned/python \
  --output /new/current-runtime-qualification
```

Make and hosted CI invoke this same coordinator. The thirteen inherited
groups, Generator production cases and public F64 logical192 bind the same actual main source,
image and cold-build receipt. Generator131 additionally compiles/links
and executes separate ABI and reordered images from the actual current
objects. Their identities are checked separately; this is not an
1020-case single-executable claim. Both tool paths are explicit CLI
arguments, and the entropy `build.json` SHA is computed from the actual
bound file rather than copied from a local or historical run. It attempts remaining groups
after an ordinary group failure, retains complete original streams and checks
all reports. A prepared adapter, earlier image's PASS, missing report, timeout,
changed input or incomplete stream cannot satisfy the gate.

[`qualification.py`](../tools/current_runtime/qualification.py) is the shared
build-admission code. It binds the current source and manifest, private native
tool copies, every source/object, native compiler/link input/output bytes and
process EOFs. It reconstructs the exact link request and independently
recomputes the actual object/source stack requirement under the fixed limits.
Runtime comparison values remain independent of this build metadata.

New dtype functionality intentionally supersedes some historical refusal
controls. Their original executions and FAIL reports remain unchanged. The
current admission records the exact changed indices, checks the actual native
wire, and requires separate independent reference comparisons for the newly
successful result. The original numerical cases, error controls outside that
extension and resource assertions remain mandatory. This does not turn an old
failed expectation into a claimed historical PASS.

Current full35 uses one explicit ViewInfo adapter. It binds the real HIA retain,
release and layout bodies, then adds actual base edges to the exact lifetime
and graph-reset equations. It retains the original numerical assertions,
672-byte descriptor, strict report checker and all positive/negative controls.
The original270 package retains all original case bodies and transport; its
adapter only binds the current source/owner identity and complete build.

Live32 retains the complete immutable PR39 source package, including the
original oracle, capture code and owner equations. An explicit, reversible
identity adapter binds the current function inventory, source, ELF and actual
cold build. Its independent readback reopens all 2,132 request/response pairs
and 4,264 raw and decoded 84-word owner snapshots, checks 32 native exits and
64 native EOF observations, and preserves all original partial-write/refusal
controls. The pre-migration 667 run and this later reader
qualification have distinct source and execution receipts.

Resource7 retains the seven original resource recipes, reference values, full
owner equations and allocation thresholds. Its source admission proves exact
retained function bodies and the unchanged allocation path for the actual
F32 nodes. It accepts the actual cold-build receipt supplied by the coordinator;
the source and ELF remain fixed. Each worker and native child must finish with
complete EOF, and each actual OOM case must restore its original process limit.

AsStrided16 preserves all four original readers and owner observers through
eight byte-exact source inverses. The shared admission binds the actual current
cold build, and an independent reader checks every original frame and complete
owner archive. A read-only normal-return observer binds the original finish
function and records actual exit/EOF; its claim is limited to the reviewed
original call graph, which has no early output-close path. No expected values,
reader assertions or process finish implementation are replaced.

The previous published image passed the complete eleven-group 683 gate in
[actual CI](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44294832902).
PR40 was merged with the same executed file tree. The retained 700-function
cat predecessor changed three complete HIA function bodies; the other 697 bodies are constrained by a
whole-source inverse to that exact previous image's source, followed by every
original logical/PR38/AVX2/baseline proof. That 700-function cat image
separately passed Cat4 and the ten fixed metadata comparisons. The 729
candidate retains its exact full-source inverse to that predecessor.
The new complete fifteen-group 1020 gate requires fresh execution; all earlier runs retain their original
denominators and source snapshots.

Metadata10 preserves the full original driver with only its two source/image
identity literals changed. A passive normal-return observer records the exact
original Tensor reader's complete owner bytes and native finish observations.
The separate admission reconstructs every literal command, 13/96/98 metadata
frame, logical Storage byte and Tensor/Storage/Version relation before applying
the unchanged comparator. Original comparisons, cleanup code and native commands
remain intact. The historical seven-case Version mismatch remains a failure
record for its original image.

Generator131 uses the three strict restore APIs in
[`current_runtime_generator_fixtures.py`](../tests/current_runtime_generator_fixtures.py).
The original26 reference inputs, complete source-only 712 predecessor and
portable qualifier are restored with every SHA and mode intact. Its
[pure admission](../tests/current_runtime_generator131.py) reopens actual
raw frames, complete owner/state captures, compiler/link requests,
production and probe exits/EOFs, and the complete current cold/tool
admission. Original case and control bodies are not replaced. The
worker keeps its original `-B` invocation, since its sibling imports and
live Torch oracle require the normal package environment.

Public F64 logical192 is a separate mandatory group. Its six source modules are
byte-identical to the frozen reader, and the outer admission reapplies its
strict 69-input cold-build admission and complete 196-child frame/owner
readback. The sixteen-file reference package preserves the original 192 dense
cases, two allocation observations, six outside-scope Torch observations,
producer sources, all six provenance inputs, actual interpreter entity and
public execution/transport streams. The archived interpreter is evidence and
is not executed by fixture restoration. Two native scope controls require the
current requires-grad and as_strided refusals; neither is counted as Torch
compatibility. Original logical147 remains mandatory because its Bool/I64/F32,
private dtype3, view and gradient constructions differ from direct public F64.
All 1020 cases still require a new combined execution on the admitted build.

The compact view/scalar reference archive contains exactly the eight original
reference/probe files and retains their original bytes and archived modes.
The complete historical native76 archive remains recoverable through its
[location index](../validation/view-reshape-r4-native76/archive-location.json);
current tests consume the compact fixed subset.

CI archives actual source/build inputs, tool bootstrap outputs, native wire,
owner snapshots, original references and reports after success or failure.
Published results are indexed by the [current audit](current-audit.md).
The [full acceptance ledger](../compatibility/full-acceptance.json) remains the
authority for whole-PyTorch closure.

## Reproduce compilation units

The aggregate HIA source is authoritative. Reproduce its units directly:

```sh
python3 -I -S -B tools/current_runtime/partition.py \
  --runtime tools/current_runtime/source/runtime.hepta \
  --name current --prefix runtime \
  --output /new/current-repartitioned
```

The RNG and owner support units default to the current source package.
Dependency traversal and deterministic grouping use at most 65 functions,
3,600 instructions and 1,000 blocks per group, below native tool limits.
Every core function occurs exactly once and its full bytes must match the
aggregate. The fixed RNG and owner support identities are checked separately.

For a byte-for-byte reproduction, add
`--expected-build /absolute/path/to/build.json`. This requires the same source,
unit ordering, unit bytes and native tools, and carries the actual object and
ELF hashes into the generated manifest. A subsequent cold build must reproduce
every hash. No previous application binary is used to produce the new one.
The generated `partition.json` is a local receipt; it is not a second published
source version.

## Implementation changes and retirement

Edit the canonical runtime, regenerate units, review the exact function and
manifest changes, and cold-build in a new directory. A new implementation
first records its actual object/image hashes; all applicable current-image
suites then run against that build. Qualification of an earlier source never
transfers merely because the new manifest was updated.

Current application builds have no historical application-build dependency.
Some original independent validators and separate toolchain/component tests
still bind historical source fixtures. Their dependencies must be migrated
with equivalent checks before retiring them. Unreachable application bodies
and redundant generated current-source versions can be removed immediately
once the complete call graph and actual current-image regressions justify it.
