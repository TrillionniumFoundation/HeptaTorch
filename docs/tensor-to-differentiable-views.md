# Tensor.to recovery and differentiable view history

## Verified result and scope

The recovered 09b Tensor.to source and the R2 differentiable-view parent have
complete HIA build recipes. The current R3 successor has actually compiled all
thirteen units and passed both immutable reference suites: the original
fourteen cases and four additional input-boundary cases. R3 restores the real
incoming gradient geometry and clone contract for overlapping and holed
inputs, and preserves the observed NaN sign and payload during coverage
normalization.

A separate seventy-two-case replay also passes on this same R3 image. It
compares gradient words and verifies exact input construction and clean
session termination; its narrower scope is recorded separately below.

This is a bounded CPU F32 profile. Complete PyTorch parity remains **HOLD**:
2,590 native schemas, 688 derivative declarations and all seventeen full
system gates remain in scope; no complete schema is newly closed. The
[acceptance ledger](../compatibility/full-acceptance.json) continues to distinguish
finite native qualification from full subsystem completion.

## Four distinct native identities

| Profile | Functions / blocks / instructions | Units | State / backing | Runtime result |
| --- | --- | --- | --- | --- |
| Recovered Tensor.to 09b | 614 / 8,078 / 29,033 | 12 | HTTOCP08 / HTOTOCP8 | Original full35 retained; exact cold rebuild |
| Differentiable view R1 | 614 / 8,082 / 29,053 | 12 | HTVAD001 / HTOVAD01 | First complete strict14 replay: 13 PASS / 1 FAIL |
| Differentiable view R2 | 614 / 8,090 / 29,094 | 12 | HTVAD002 / HTOVAD02 | Original14 PASS; complete input4 replay: 1 PASS / 3 FAIL |
| Differentiable view R3 | 627 / 8,245 / 29,636 | 13 | HTVAD003 / HTOVAD03 | Original14 and input4 PASS; separate scoped gradient72 PASS |

All four use complete 672-byte descriptors. Their immutable identities are:

| Profile | HIA source SHA-256 | Executable SHA-256 |
| --- | --- | --- |
| 09b | 4f8f73f463f681b5eb29423d71a35a0b42d807fd92bf976bbf76c1970fbcee33 | 53fb43b869858f020c0b936a65f00f3b14b97d9e37038a74c357e1da471fe083 |
| R1 | 644c1e3e7828c60b1335794469a7ab38f73cb73b9f364d1b7ec9b83fdf7bf3c5 | 526dde8b0e3d1b48511781e0b5d5c5082fd25127cef4df96edfbc41dc99c15ef |
| R2 | 197bd52db4815f575ea3e9055d3f5556900314bd2f4fbb262e80cea76ea16ef2 | 13cee2163e8f62081db35e831885714789ac5117fb80c030a1ccaf7194e0821c |
| R3 | 3c2c0a125bcac6d47075bf58fc3f78c0e7b9a0b1370757981b99dcd3793b21c6 | 7841b19051960b6fa13cb98841dbb8e4d6ead97f03e2ad65af415e1a35b03221 |

The [09b recovered recipe](../tools/tensor_to_native_recovered/),
[R1 historical recipe](../tools/differentiable_view_recovered/),
[R2 parent recipe](../tools/differentiable_view_recovered_v2/) and
[R3 current recipe](../tools/differentiable_view_recovered_r3/) have separate
manifests. The frozen [R3 profile](../tools/differentiable_view_recovered_r3/profile.json)
binds its actual source, executable, all thirteen source/object pairs and the
two strict runtime reports. R3 requires the exact twelve-unit R2 parent,
including its constructor and retained-hook fix. It does not accept a direct
09b input as that parent.

## What the HIA implementation changes

The direct strided-view constructor retains the existing geometry, storage
extent and dtype guards. For a differentiable F32 input it records a real
op44 edge to that immediate input. The ultimate base Tensor remains separately
represented by the view's base fields. These two relationships need different
lifetimes and cannot be substituted for one another.

R1 exposed a second issue under Version-driven history refresh. The general
history capture helper correctly preserves the old history for other nonleaf
operations, but its retained-gradient routing is unsuitable when this view
refresh replaces the current gradient function. In the failing case, backward
through an old copy wrote the current view gradients, backward through the new
copy omitted a retained gradient, and the final public-view drop failed the
ownership audit.

R2 changes only the capture-success branch of HIA function to_view_refresh
and the four state-tag producers. Its new blocks verify the old hook's exact
Tensor identity and incarnation, clear that old weak hook, and enable the
current Tensor's hook. They preserve the current accumulator's low bits and
incarnation, all strong Tensor references, edge geometry and arithmetic.
An old AccumulateGrad node bypasses this hook movement.

The general history_capture function remains unchanged. The patch contains
an exact inverse to the prior function, and the R1 source and failed image are
retained. This behavior follows the hook update order in the pinned upstream
[variable.cpp](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/variable.cpp),
whose complete restored bytes have SHA-256
ca2f1e573307069dee2fdff70ba81f161f48817cd278ad5464b9809cf23ab2c6.

R3 keeps that R2 constructor and hook behavior. It repairs the incoming op44
gradient representation: the shared Storage span, effective offset and input
strides are materialized before the first-gradient contract is evaluated.
Overlapping or holed inputs can therefore take the existing real contiguous
clone path, producing a fresh public gradient with Version 0. No Version is
reset to obtain the reference result.

The private preserve buffer is reserved before backward. On the first
nonempty op44 construction, HIA checks the fresh private buffer, releases its
unused aligned tail, and updates requested and rounded extents consistently.
The actual copy or index-add and, when required, whole-Storage coverage
division each retain their own mutation and Version effects. Subsequent
incoming merges and cache cleanup retain their existing behavior.

The scoped dview_coverage_divide helper preserves the sign and payload of a
NaN numerator and sets its quiet bit when coverage is positive. Other inputs,
including zero-coverage holes, use the original arithmetic. Global division
and addition functions are unchanged. These changes follow the pinned
[AsStrided backward implementation](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/FunctionsManual.cpp)
and [gradient layout contract](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/utils/grad_layout_contract.h);
the public numerical and Version expectations come from the immutable actual
input4 observations. Complete source and compilation records are retained in
the [R3 producer](../tools/differentiable_view_recovered_r3/) and
[build history](../validation/differentiable-view-r3-build-history/).

Python assembles source, submits compiler inputs, drives the transport and
compares observations. Tensor values, gradients, mutation, reference counts
and ownership are implemented and executed in HIA.

## Strict actual-reference qualification

The immutable PyTorch observation has SHA-256
2270d5ba925bcf91c56db94fac663695e69079c74edbf80aff8fa086e7c81665.
It contains fourteen cases and 209 attempted operations: 207 succeeded and two
raised the recorded expected errors. Its provenance retains all eleven pinned
interpreter, package, source-build and installation checks.

The portable reference preparation command verifies all fifty-two original
packet members, their complete bytes and modes, the single XZ stream and
the tar digest. It then runs the original independent admission reader:
41,095 checks, 432 snapshots, 2,539 Tensor records, 14,126 binary32 words and
57,144 physical storage bytes. This preparation performs no new PyTorch or
native runtime execution.

The native comparison reader revision named R3 is unchanged between the
production R1 failure and production R2 success. Its SHA-256 is
52c3b3242a73fb412102c83c5ba47cf8d95a4eb3c01e56f41bfe593066506d05.
The R2 observer differs from its R1 predecessor only in the two state-tag
literals; its whole-file inverse is preserved.

For the production R3 image, the separate profile reader R5 explicitly binds
627 functions and thirteen native units. Only its main entry and profile
arguments differ; the Case class, reference admission and other comparison
functions remain unchanged, with exact inverse records. Its SHA-256 is
deb9f6fafc3f91b2749bdfac030e19cb0ccdb832fa238af91ed9feb396e62eb8.
The current observer has SHA-256
68d0f4657c6cc7062d98cbf033565b16c7b51f3381c6490247e42512c5baf4a2;
its R2-to-R3 change is limited to the two state tags.

The successful R2 run records:

| Check | Actual result |
| --- | --- |
| Cases | 14 PASS, 0 FAIL |
| Protocol frames | 15,135 |
| Strict assertions | 96,461 |
| Exact binary32 words | 14,126 |
| Complete Generator state bytes compared | 2,325,760 |
| Whole-owner rollback bytes compared | 459,258,768 |
| Session EOF and exit status | All fourteen reached EOF and exited zero |
| Input mutation | No input changed |

The report SHA-256 is
0b987fcb1d7fb74fe8a6d1edb994bed53851f07acbc3dfe6af6f754489e598d4.
The [complete native14 evidence](../validation/differentiable-view-r2-native14/)
retains 749 original members totaling 53,715,863 bytes, including the actual
transport, ownership observations, input copies, all twelve HIA units and
objects, compile/link output and execution receipts. Its original archive
SHA-256 is 987313191d99ac245f794162e3e6b416e8b3ff308c9f83aa190a62ad4b17ee5d.

The comparisons cover recorded logical and physical bytes, shapes, strides,
offsets, Tensor and Storage relationships, Versions, immediate and captured
edges, Saved fields and errors, gradient presence, retained-gradient and retry
effects, Weak expiry and the complete 5,056-byte generator state. They preserve
the expected Saved-version and released-Saved failures. Additional native
execution counters are recorded without representing them as observed PyTorch
counters.

Every fixed live identity relation in this reference has same_Python_object
equal to same_Tensor. Qualification covers those actual relations. It does not
establish general multiple-Python-wrapper reference-count semantics or Python
warning-text frontend compatibility.

## Existing full35 evidence and the copied control correction

The recovered parent image retains the original full35 execution:
35 cases, 6,344 frames and 2,590,274 compared values. Original EOF checks,
post-close Version checks and all eleven provenance checks are retained.
The 283 previously recorded state-Version differences are explicit; parameter
Version differences remain zero.

The original combined execution is still classified FAIL because its copied
negative-control run constructed one synthetic node with eighty words instead
of the new complete eighty-four words. The corrected entry changes precisely
that one literal. Its whole-file inverse is exact; control names, order,
expected refusals and production HIA bytes remain unchanged. The later copied
run passes four positive and eighteen negative controls and adds no native or
PyTorch execution.

See the [original and corrected full35 package](../validation/tensor-to-frozen35-remote09b/)
and the [copied672 source provenance](../validation/tensor-to-frozen35-copied672/).
An independent root readmission also reran the unchanged strict full35 checker
and checked all 6,344 archived transitions, 6,321 full snapshots and 231,198
complete eighty-four-word node records. It preserved all seventy-eight input
files and performed no new native or PyTorch execution. Its report is retained
under [recovery review](../validation/tensor-to-recovery-review/pure-full35/).

## Failure history and recovery boundaries

The [R1 native history](../validation/differentiable-view-r1-native14-history/)
distinguishes two actual attempts:

1. The older reader R2 reached native case zero, including 690 frames and clean
   EOF, then failed while serializing a protocol Payload into its diagnostic
   report. The other thirteen cases were NOT_RUN. This host failure is not a
   native numeric classification.
2. Reader R3 corrected only diagnostic Payload encoding. The first complete
   native replay then reported the genuine 13 PASS / 1 FAIL result. The R2 HIA
   hook fix, with that same reader and frozen reference, later passed all fourteen.

The [R1 build history](../validation/differentiable-view-r1-build-history/)
preserves the initial bootstrap failure caused by an omitted real root
pyproject.toml in a sparse checkout, the separate local-promisor clone failure,
the exact source-layout repair, and the successful ten-stage fixed-tool cold
build. No empty project sentinel or resolver change was used.

The successful fixed tools have the original module-compiler and session-linker
hashes. All twelve parent objects and the original parent executable were
actually recreated, then the R1 and R2 children were actually compiled.
[R2 build evidence](../validation/differentiable-view-r2-build-history/) records
that its parent and fixed tools were reused from that verified cold build.

Earlier Cloud full integration attempts ended with SIGKILL and no admitted
complete result. A later reconnection recovered the exact original archives:
all 309 producer members and all 795 members of the first failed cold run were
verified, including complete bytes, sizes, modes and single-stream XZ EOF.
The original exit -9 and outer FAIL remain unchanged; their cause is not
reclassified as a proven out-of-memory event.

The separate [original source recovery](../validation/tensor-to-original-source-recovery/)
also verifies all 78 source-packet members and recovers the full 96,402-byte
coordinator, with SHA-256
a6f91b1d6134255652a7ac2fea5538dc084e96f521866063abafe88a9ac35c62.
Its original 31-file package manifest and factory-shape failure history remain
available. This resolves the missing-original-source obligation without
reclassifying its historical execution. The frozen-HIA recovery recipe has a
distinct build identity.

The [original16 recovery](../validation/tensor-to-original16-recovery/)
restores four unchanged replay programs and their actual references: copy4,
reduction4, lanes2 and branches6. Its prior same-image observations retain
1,920 frames, 10,272 checks, 21,892 exact words, 1,860,608 full generator-state
bytes and all sixteen clean EOF results. Complete original reference packets
and native transcripts are preserved. Restoring these bytes performs no new
native or Torch execution; the added CI targets run them on the newly rebuilt
parent image.

Partial immutable Git object uploads are not treated as a published source
tree or an admitted hosted result.

## Additional input boundaries: actual observations and R2 failures

The [input4 reference packet](../validation/differentiable-view-input4-reference/)
preserves one additional real PyTorch invocation, using the fixed probe
cf569335f48624c1edf2f97a649200f2e792ebcef5b6f1df6bac73293ce02efc.
Its actual report is
f4abe6a750ecc791b3a7bc7aa507b4c77eb0a2fcccf45b0c6a828db0e01490d3:
four observed recipes, all 103 attempted operations accepted, seven backward
operations, all eleven provenance checks and 25 complete input bindings.
The independent original byte admission performs 64,059 checks over 2,019
Tensor records, 210 snapshots and 10,120 logical binary32 words.

The portable preparation command restores all 42 packet members with their
original modes, checks the full tar digest and single XZ stream through EOF,
and binds the 25 inputs to the original execution receipt. It preserves and
checks the original independent admission; it does not claim to rerun those
64,059 checks or to execute PyTorch again.

The [R2 input4 history](../validation/differentiable-view-r2-input4-history/)
contains both actual native attempts. The first reader stopped early because
it had not mapped a second publication of the same seed StorageWeakRef key.
The corrected event mapping creates the new Weak reference before releasing
the old one, matching the recorded Python evaluation order. Every inherited
comparison method remains unchanged.

That complete second R2 replay reports **1 PASS / 3 FAIL**, with 12,353 frames,
77,432 checks, all 10,120 raw words, 374,676,864 whole-owner rollback bytes,
four clean EOF results and unchanged inputs. Its report SHA-256 is
c7c745e00788b6961a289dec59c0453acffe98838bc9b33693b80b43ce98e8cd.

| Fixed input recipe | Complete R2 result | Observed difference |
| --- | --- | --- |
| Zero-stride overlapping input | FAIL | Public input gradient Version 1 instead of 0; later NaN sign/payload lost |
| Nonzero-stride overlapping input | FAIL | Public input gradient Version 1 instead of 0; later NaN sign/payload lost |
| Holed logical input, output offset outside part of its logical footprint | FAIL | Public input gradient Version 1 instead of 0 and consequent Version relations |
| Two distinct retained sibling views joining one graph | PASS | All forty recorded events matched |

The source diagnosis identifies two separate implementation defects. Incoming
op44 gradients were described as contiguous even when AsStrided backward
returns input geometry, which selected gradient reuse where the layout contract
requires a fresh clone. The normalization division also used the existing
canonical-NaN division path, losing the actual numerator sign and payload.
R3 repairs both paths and has actually passed the unchanged semantic
comparisons for the original fourteen and new four cases. The R2 report
remains 1 PASS / 3 FAIL, and its fourteen-case qualification remains limited
to that original set.

## Current R3 strict qualification

The [original14 report](../validation/differentiable-view-r3-native14/report.json)
and [input4 report](../validation/differentiable-view-r3-input4/report.json)
come from separate executions of the same frozen R3 image. The input4 reader
keeps its corrected Weak replacement mapping and all twenty-one inherited
comparison methods. Neither execution regenerates the PyTorch reference.

| Check | R3 original14 | R3 input4 |
| --- | --- | --- |
| Cases | 14 PASS / 0 FAIL | 4 PASS / 0 FAIL |
| Protocol frames | 15,135 | 12,353 |
| Strict assertions | 96,461 | 77,432 |
| Exact binary32 words | 14,126 | 10,120 |
| Complete Generator state bytes compared | 2,325,760 | 1,102,208 |
| Whole-owner rollback bytes compared | 459,258,768 | 374,676,864 |
| Session EOF / exit | All fourteen, zero | All four, zero |
| Bound inputs | All unchanged | All unchanged |

The original14 report SHA-256 is
c3147b161f32a85a87ccf00da47f3d2fc8bc2a5c47e159cf1f7ad1f8938606a1;
its [757-member evidence package](../validation/differentiable-view-r3-native14/package-manifest.json)
retains 47,725,967 original bytes. The input4 report SHA-256 is
59851d848c383305d8e827d0ef3f1ef1b5fba6f513f5ecfda088fa9ced544d68;
its [439-member evidence package](../validation/differentiable-view-r3-input4/package-manifest.json)
retains 43,115,466 original bytes. Both preserve the raw commands, responses,
owner observations, all thirteen units, executable and normal EOF records.

These fixed input4 cases now cover the two observed input-overlap geometries,
the holed input footprint and the distinct retained-sibling join. They do not
establish arbitrary input geometry, arbitrary graph fanout or a complete
AsStrided schema. The [R2 failures](../validation/differentiable-view-r2-input4-history/)
remain separately recoverable.

## Separate multiaxis72 gradient evidence

The immutable seventy-two-case reference has SHA-256
2512acdfeaf98f0aa8d8f44ce37dec48b926244ae274066e6e5b86e919a806e9.
It uses a contiguous 256-element input and twenty-four output geometries with
three seed modes each. The original R1 replay used all-one input values where
the actual probe constructed arange values 0 through 255. Its original
gradient result remains recorded, with that input-mapping limitation.

The R3 successor reader corrects only that input construction alongside
explicit source, image and profile bindings. The gradient comparisons and
termination checks remain unchanged, and an exact inverse preserves the
original reader. All seventy-two first request packets were independently
checked for the exact binary32 arange input.

The [R3 report](../validation/differentiable-view-r3-multiaxis72/report.json)
passes **72 / 72**, with **1,224 frames**, **21,312 exact gradient words**,
all seventy-two clean EOF/exit-zero results and all fifty-eight bound inputs
unchanged. Its SHA-256 is
d4ab2a822f0ab78595ebf9006770b507502e75a3031f9ca09a6fb149d9c9c70d.
The [execution receipt](../validation/differentiable-view-r3-multiaxis72/execution.json)
has SHA-256
0495041aee6666f1947a6c15773b1f0415320f851c103905e630af1edc84ed10.

This reader qualifies base/view gradient presence, lengths and logical
binary32 words, successful protocol operations, exact input construction and
session termination. Gradient Versions are recorded without comparison.
It does not qualify complete Tensor/Storage identity, layout, Saved behavior,
full Generator state, general owner accounting or every CPU dispatch.
Those limits are separate from the stricter original14 and input4 readers.

The [complete package](../validation/differentiable-view-r3-multiaxis72/package-manifest.json)
retains 423 members and 30,475,622 original bytes, including source, all thirteen
units, raw transcripts, reference, exact reader inverse and the original R1
history. Its archive SHA-256 is
330a859ced04cb0ebecdae5b02c86567daccd8e79bb477da5056067363367c8b.
This R3 run adds no new PyTorch execution.

## Reproduction and CI

From a fresh worktree, after building the pinned Hepta stack tools:

~~~sh
make check-tensor-to-recovered-source
make tensor-to-recovered \
  HEPTA_STACK_BUILD=/absolute/path/to/hepta-stack-budget
make test-tensor-to-recovered-tombstone \
  HEPTA_STACK_BUILD=/absolute/path/to/hepta-stack-budget
make test-differentiable-view-reference14
make test-differentiable-view-input4-reference
make differentiable-view-r2-parent \
  HEPTA_SOURCE_ROOT=/absolute/path/to/pinned-hepta \
  HEPTA_STACK_BUILD=/absolute/path/to/hepta-stack-budget
make differentiable-view-recovered \
  HEPTA_SOURCE_ROOT=/absolute/path/to/pinned-hepta \
  HEPTA_STACK_BUILD=/absolute/path/to/hepta-stack-budget
make test-differentiable-view-native14 \
  HEPTA_SOURCE_ROOT=/absolute/path/to/pinned-hepta \
  HEPTA_STACK_BUILD=/absolute/path/to/hepta-stack-budget
make test-differentiable-view-native-input4 \
  HEPTA_SOURCE_ROOT=/absolute/path/to/pinned-hepta \
  HEPTA_STACK_BUILD=/absolute/path/to/hepta-stack-budget
~~~

The unchanged live32 and original full35 require the separately pinned PyTorch
oracle installation:

~~~sh
make test-tensor-to-recovered-live32
make test-tensor-to-recovered-frozen35
make test-tensor-to-recovered-copied18
make test-tensor-to-recovered-original16
make test-differentiable-view-input4-reference
~~~

These commands retain failed output directories. Use a new output path or a
fresh worktree for a new execution. The self-check uses Python's standard
library and its sibling rebuild module; its invocation intentionally omits
isolated-import mode. The native reader and reference preparer use explicit
pinned dependency paths and can run with isolated imports.

The current build chain reconstructs the exact R2 parent before compiling
the thirteen-unit R3 child. The original16, live32 and full35 targets above
retain their separately identified recovered-parent scope; their historical
success is not transferred to R3.

The additive workflow retains the original steps and ninety-minute job
limit. It separately checks source reconstruction, compilation, tombstones,
live32, full35, copied controls, immutable reference admission and the
original14/input4 replays. Each command preserves its exit status, and
always-run artifacts retain generated source, reports and runtime transcripts.
Hosted success must be evaluated for the actual PR source head; these local
R3 records do not assert that hosted result. The scoped multiaxis72 handoff
has its own exact command and bindings in the archived execution receipt.

## R3 finite preserve-resource evidence after the profile freeze

[The separate resource packet](../validation/differentiable-view-r3-preserve-resources/qualification.json)
qualifies three finite controls on the same R3 source and executable. The
first invocation remains FAIL: tail release/reuse and the Saved-error prefix
passed; OOM preparation requested a dimension above the existing 262144
limit. Source02 caps the filler dimension and its sole successor invocation
runs only the OOM case. Neither earlier successful case was reexecuted.

Across the successful controls there are 113 frames and 243 checks. These
exclude the failed preparation's 13 frames and 17 checks; the complete
retained history has 126 frames, 260 checks and four EOF exits of zero.
The successful OOM case checks the complete owner, its pointer/length,
5056 RNG bytes and mappings through the failed request, then restores the
address-space limit and succeeds with the identical backward request.
The Saved failure follows two real op44 nodes and retains result (6,2,0,3);
its atomic57 variant preserves complete rollback. Tail reuse checks both
live gradient generations and final Weak expiry.

These are three bounded resource compositions, not general allocation or
owner qualification. The frozen producer profile's historical NOT_RUN
fields remain unchanged. Run `make test-differentiable-view-preserve-resources`
after the same R3 build and input4 reference restoration. The driver uses
Python `-S -B` because it imports sibling modules; all tensor operations
execute in the native HIA image. The CI step preserves its status, Make
stdout/stderr and the complete output directory even after failure.
Running all three source02 controls together on the hosted source head
still requires that head's actual CI and raw artifact admission.

## Remaining implementation and qualification boundaries

| Boundary | Current evidence and remaining limit |
| --- | --- |
| R1 retained-hook refresh defect | Corrected in R2 and preserved in R3; original14 passes on R3 |
| Input with overlapping strides | The two fixed input4 geometries pass on R3, including public G Version and NaN sign/payload; arbitrary geometries remain unqualified |
| Holed input and output outside part of its logical footprint | The fixed input4 recipe passes on R3; general geometry is not closed |
| Different view siblings joining a graph | The fixed retained-sibling recipe passes on R3; arbitrary fanout remains unqualified |
| Multiaxis output views from the fixed contiguous input | 72 gradient-word/EOF cases pass on R3 within the narrower reader scope above |
| Original16 and broader inherited regressions on R3 | NOT_RUN at the immutable R3 profile freeze; parent-image results remain separate |
| R3 allocation/OOM and general owner controls | Three finite preserve-resource compositions now have separate actual evidence; historical profile NOT_RUN is retained, and general owner/OOM behavior remains unqualified |
| General view/reshape, Tensor.to overloads, all dtypes/devices/layouts and copy modes | Incomplete |
| Higher-order and forward AD, hooks, custom functions and transforms | Incomplete |
| Python frontend, accelerator, distributed, export and data subsystems | Separate full-project gates remain open |
| Full schema parity and all-system acceptance | 0 / 2,590; HOLD |

Each named suite retains its actual input and comparison scope. Overlapping
suites are never added together as a unique API or complete-schema count.
