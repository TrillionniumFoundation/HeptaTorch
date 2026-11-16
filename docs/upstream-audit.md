# PyTorch upstream audit and completion criteria

HeptaTorch implements a native HIA tensor and training runtime. The compatibility
target is the pinned PyTorch release below. Completion is tracked against that
release's behavior, including runtime services and backend semantics, as well as
its operator declarations.

## Pinned baseline

| Field | Value |
| --- | --- |
| Upstream | `pytorch/pytorch` |
| Stable release | `v2.15.0` |
| Release date | 2026-10-28 |
| Commit | `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457` |
| Audit date | 2026-11-15 |

The GitHub release API and tag reference were read independently. The tag points
directly to the commit above. The release is neither a draft nor a prerelease.
The lock file preserves the source blob and subtree identities used for the
inventory. Future work must update that lock deliberately instead of silently
comparing against a moving `main` branch.

- [Release](https://github.com/pytorch/pytorch/releases/tag/v2.15.0)
- [Pinned source](https://github.com/pytorch/pytorch/tree/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457)
- [Local upstream lock](../compatibility/upstream-lock.json)
- [Complete native schema inventory](../compatibility/upstream-schemas.json)

## Measured inventory

| Source | Count | Meaning |
| --- | ---: | --- |
| `aten/src/ATen/native/native_functions.yaml` | 2,590 | Exact lines beginning `- func:` |
| Distinct names before `(` | 2,590 | Includes overload suffixes |
| Distinct names before `.` | 1,558 | Base names, not a public API count |
| `tools/autograd/derivatives.yaml` | 688 | Exact lines beginning `- name:` |
| `torch/` | 4,761 | Git blobs |
| `aten/` | 2,757 | Git blobs |
| `c10/` | 441 | Git blobs |
| `test/` | 11,920 | Git blobs, including fixtures |
| `torchgen/` | 89 | Git blobs |
| `functorch/` | 83 | Git blobs; excludes `torch/_functorch` |
| `caffe2/` | 49 | Git blobs |
| `third_party/` | 225 | Git blobs, plus 35 submodule gitlinks |

Every queried subtree returned `truncated=false`. These counts do not include
the contents of submodule gitlinks. They do not measure implementation effort,
executed tests, public API coverage, or equivalent behavior.

The schema inventory contains every one of the 2,590 exact declaration strings.
Its initial `unimplemented` status means that the complete upstream schema has
not been qualified. A restricted native implementation can be useful and tested
without covering all of that declaration's dtype, device, layout, shape, aliasing,
mutation, and differentiation behavior. Scoped capability evidence is tracked
separately. The full-schema status must not be promoted by counting a function
name or by passing only a contiguous-vector example.

The denominator also excludes generated schemas, custom and other namespaces,
higher-level modules, Python and C++ frontend behavior, compiler facilities,
distributed services, serialization, and backend-specific kernels. Therefore
even qualifying all 2,590 records would not by itself complete the whole project.

Sources:
[native declarations](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/native_functions.yaml),
[derivative declarations](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/tools/autograd/derivatives.yaml).

## Native implementation boundary

The initial executable target is an HIA CPU runtime using binary32 values,
software floating-point primitives, tensors of rank at most four, reverse-mode
automatic differentiation, and SGD. The current HIA toolchain's lack of native
floating-point instructions makes numerical primitive qualification part of the
runtime work. This target establishes an executable training path and leaves
the complete upstream target in the inventory.

HIA sources must perform the actual tensor operations, gradient propagation,
and parameter updates. A host PyTorch call, a precomputed answer, or an unchecked
constant success result is not an implementation. Upstream tools may be used
outside the runtime to produce independent reference data. Compiler bootstrap
requirements and test-oracle dependencies must be stated independently from
runtime dependencies.

For every supported operation, evidence must identify:

- The HIA entry point and tested source revision.
- Device, dtype, rank, layout, shape and capacity constraints.
- Forward, backward, aliasing and mutation behavior.
- Unsupported inputs and the resulting explicit error.
- Reference input and output values, tolerances, and how the reference was made.
- The command and observed result of the native executable.

A native HIA API and unchanged execution of existing `import torch` programs are
different compatibility surfaces. Full frontend compatibility also requires a
specified language boundary and the corresponding object, error, override, and
extension protocols.

## Semantic requirements for the CPU training path

### Tensor storage and shape

A tensor descriptor must distinguish logical shape from physical strides and
storage offset. Multiplication when computing sizes and address ranges must be
checked before allocation or access. Zero-element tensors and scalar rank-zero
tensors need separate test cases. Invalid dimensions, incompatible shapes,
capacity exhaustion, and out-of-bounds views must return an error before data
access.

Transpose and view operations share underlying storage in PyTorch. A copied
transpose can implement the numerical transformation but cannot qualify aliasing
semantics. `view` requires compatible strides; `reshape` may copy when required.
Basic indexing returns a view and advanced indexing returns a copy. Test writes
through a view against its base to verify sharing.

Source: [Tensor views](https://docs.pytorch.org/docs/2.15/tensor_view.html).

### Broadcasting and reductions

Broadcast trailing dimensions, including missing dimensions and singleton
dimensions. Test `[2, 1, 3]` with `[1, 4, 1]`, rank-zero scalar inputs, incompatible
shapes, and empty axes. In particular, a size-zero dimension broadcast with a
singleton remains size zero; a plain maximum of the two dimension sizes is
incorrect for this case. In-place operations must not grow the target's shape.

Backward must sum contributions over every expanded or missing input axis.
Reduction tests must cover empty input, explicit axes, negative axes, duplicate
axes, keepdim, NaNs, and dtype-dependent behavior. An empty sum and an empty mean
have different results. A full upstream schema also includes dtype promotion;
the binary32 slice must state its narrower dtype contract.

Source: [Broadcasting semantics](https://docs.pytorch.org/docs/2.15/notes/broadcasting.html).

### Software binary32 arithmetic

Qualify the actual software arithmetic independently before relying on training
loss as evidence. Include positive and negative zero, both signs, subnormals,
minimum normal values, overflow, underflow, infinities, NaNs, cancellation,
division by zero, halfway rounding, and guard/round/sticky-bit cases. Record the
rounding mode and distinguish numerical equality from exact bit-pattern
requirements. NaN payload propagation needs a stated contract if it is tested.

Floating-point reductions and matrix products can have different rounding from
other valid implementations because operation order affects the result. The
test oracle should compare elementwise primitives at the appropriate bit or
numerical level and use justified tolerances for composite calculations.

Source: [Numerical accuracy](https://docs.pytorch.org/docs/2.15/notes/numerical_accuracy.html).

### Automatic differentiation

The runtime must record actual executed operations and propagate gradients in
reverse dependency order. Multiple paths to the same input must accumulate:
`x * x + x` is a small decisive case. Broadcasting requires reduction to each
input's original shape. Matrix multiplication requires both input derivatives.
Test ReLU's selected derivative at zero independently of smooth finite-difference
tests.

Saved values must remain valid until their backward operations run. A mutation
policy must either retain snapshots or detect invalidated saved tensors.
Upstream uses version counters and rejects changed saved values when necessary.
Gradient accumulation across calls, graph retention, no-grad mode, detach, and
higher derivatives are separate behaviors and must be accounted for separately.

Source: [Autograd mechanics](https://docs.pytorch.org/docs/2.15/notes/autograd.html).

### Optimization and training

An end-to-end test must execute forward computation, scalar loss, backward,
gradient clearing, and a parameter update. Use deterministic affine regression
and a small network with at least one nonlinear operation. Assert gradients and
updated parameters against independent values, then recompute the new loss.
Falling loss alone does not establish correct gradients.

Vanilla SGD can close independently of momentum variants. If momentum is added,
PyTorch initializes the buffer to the first gradient; dampening applies starting
with later steps. Nesterov, weight decay, maximize, skipped gradients, parameter
groups and optimizer state restoration require distinct qualification.

Source: [Pinned SGD](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/optim/sgd.py).

## Whole-project completion matrix

| Component | Required implementation and evidence |
| --- | --- |
| Runtime and dispatcher | Storage ownership, allocators, dtype promotion, schemas, dispatch, meta execution, errors and extension registration |
| Tensor operations | All declared operations and generated variants across supported shapes, dtypes, devices and layouts |
| Views and mutation | Strides, offsets, aliasing, overlap rules, in-place operations and saved-value correctness |
| Sparse, nested and quantized tensors | Representations, invariant checks, operators, conversions and differentiation |
| Autodiff and transforms | Reverse and forward mode, higher derivatives, hooks, custom functions, checkpointing and batching transforms |
| Neural networks | Parameters and buffers, recursive module state, train/eval semantics, convolution, pooling, normalization, attention and recurrent layers |
| Optimizers | Parameter groups, optimizer state, adaptive and momentum methods, schedules and checkpoints |
| Accelerators | Device discovery, transfer, allocation, streams/events, execution, numerical modes, and CUDA/ROCm/MPS/XPU kernels |
| Distributed systems | Rendezvous, process groups, collectives, asynchronous work, timeouts, failures, DDP, FSDP, DTensor and distributed checkpoints |
| Compiler and export | Graph capture, symbolic shapes, decompositions, transformations, lowering, fusion, AOT execution and export formats |
| Data and reproducibility | Data loading, worker lifecycle, RNG state, deterministic modes and checkpoint resumption |
| Ecosystem and packaging | Serialization, profiling, frontend APIs, extension protocols, supported platforms and reproducible builds |

The completion matrix is an engineering work inventory. No row is marked
complete merely because the repository contains a corresponding directory or
symbol. In particular, CPU results cannot establish GPU or distributed behavior.
The durable [full-acceptance ledger](../compatibility/full-acceptance.json) has
17 gates. Every gate remains open or in progress; the full upstream schema
qualification count remains zero out of the 2,590 frozen declarations.

## Upstream test sources

The pinned upstream files below are semantic references for independent tests:

- `test/test_binary_ufuncs.py`, `test/test_unary_ufuncs.py`
- `test/test_reductions.py`, `test/test_tensor_creation_ops.py`
- `test/test_view_ops.py`, `test/test_shape_ops.py`, `test/test_indexing.py`
- `test/test_autograd.py`, `test/test_ops.py`
- `test/test_nn.py`, `test/test_optim.py`, `test/test_serialization.py`
- `torch/testing/_internal/common_methods_invocations.py`
- `torch/testing/_internal/opinfo/definitions/`

Link: [Pinned test tree](https://github.com/pytorch/pytorch/tree/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/test).

Tests must record which of these behaviors were reproduced. An exact arithmetic
fixture, finite-difference check, independently implemented reference, and live
PyTorch differential run are different evidence types. They must not be reported
interchangeably. The initial audit environment's default Python could not import
`torch`; reference data must state its actual provenance.

## Attribution and repository cleanup

The extracted native schema strings are upstream reference material. The pinned
PyTorch `LICENSE` and `NOTICE` are preserved under `third_party/pytorch/` with
their original contents. This directory supplies attribution for reference data;
it does not install or embed the upstream runtime or all dependencies named in
the original notice.

Keep one implementation path for each supported behavior. Remove generated
build products, temporary probes, abandoned implementations and duplicate
reports from the tracked source tree after their replacement is verified.
Retain source, reproducible build and test entry points, required notices,
the pinned compatibility inventory, and evidence necessary to reproduce the
reported results. Missing features stay visible in the inventory rather than
being removed from the target during cleanup.

## Native extension audit

The follow-up started from HeptaTorch commit
`fed7c077e62f4c8447099927c00da178b691424b`. The complete tracked tree, single
branch and four-commit history were inventoried, the pinned HIA compiler was
checked, and all 76 original packet fixtures were reproduced before editing.
The runtime, protocol, build recipe, oracle provenance and compatibility inventory
were reviewed with targeted independent adversarial execution of the numerical,
storage, differentiation and spatial paths. This is not a formal verification of
all native machine instructions or an execution of the upstream PyTorch suite.

The implemented expansion adds broadcasted rank 1–4 matmul, Linear over arbitrary
supported batch dimensions, explicit vector-Jacobian seeds, index_select and
gather with exact integer index payloads. Matrix backward reduces broadcast
axes and accumulates both paths when operands share a leaf. Index backward
scatter-adds repeated indices. The final source and all new evidence identities
are recorded in `validation/native-extension.json`.

| Finding | Resolution or remaining dependency |
| --- | --- |
| Legal pooling requests had no computation bound | Fixed: reject more than 16,777,216 conservative kernel terms before computation; exactly-at-limit and empty-output cases retained |
| Separate rank-two matrix kernels duplicated the generalized path | Replaced by a rank-two admission wrapper; three superseded forward/reverse helpers removed |
| Non-scalar outputs lacked a direct cotangent interface | Implemented exact-shape seeded backward, strided seeds, repeated accumulation and error preflight |
| General matrix shapes and integer selection were absent | Implemented scoped native matmul, index_select and gather; remaining dtype/overload coverage stays open |
| One native compilation unit is approaching fixed compiler limits | Capacity expanded by the separate profile below to 256 functions and 2,560 blocks; the later borrowed-module bridge implements a bounded cross-unit path, while general ownership and public resource ABI remain open |
| Graph/data arena grows for the duration of one request | Resolved within one request by the subsequent graph-lifetime change below; cross-request persistence and scalable allocation remain open |
| Only software binary32 and one CPU target exist | Open: other dtypes, SIMD/device code generation, device memory/stream services and independent backend qualification |
| The wire protocol is separate from upstream frontends | Open: Python/C++ object, dispatch, module/state, override and extension interfaces |

The original pooling reproducer used only 25,824 request bytes yet admitted
41,990,400 window terms (`[1,1,80,80]`, kernel 80, stride 1, padding 40). It
exceeded a one-second audit deadline. The same request now returns capacity
status 3. The new matrix kernel has the same term limit, including the legacy
rank-two command. Whole-request CPU deadlines and scalable allocation remain
separate service concerns; per-operation bounds do not imply those facilities.

The native compiler's limits are specified in the pinned
[support profile](https://github.com/TrillionniumFoundation/Hepta/blob/f33d8da316de8fdf27b504371f1a5310997ff548/docs/HIA_SELFHOST_CORE_SUPPORT.json).
Its [object-build contract](https://github.com/TrillionniumFoundation/Hepta/blob/f33d8da316de8fdf27b504371f1a5310997ff548/docs/HIA_PROGRAM_SELFHOST_BUILD.md)
admits scalar public exports, not a general resource-bearing tensor library
interface. Scaling the full rewrite therefore requires language/runtime module
work alongside operator implementation. The full target and all 2,590 original
schema records remain intact.

That extension expanded the oracle to 153 actual PyTorch cases, including its new operators and
three composed attention paths without masking or dropout. Those attention tests
qualify composition of the implemented primitives, not the complete attention
API. That revision's fixture manifest contained 221 native packet pairs. Historical
baseline, isolated component, final integrated and hosted results retain their
own source and executable hashes in the validation records.

## Training graph lifetime audit

Starting from merged commit `297f34be8f0e9f2a2283042db513abafc3f1e7dd`, command
33 now reclaims transient graph descriptors and arena storage while preserving a
fixed initial prefix of owned leaves. Parameters, optimizer states, gradients
and versions survive reset. Transient handles combine a 56-bit generation and
an 8-bit slot, and validation compares the complete handle. A discarded handle
cannot become valid when its slot is reused. Graph traversal still follows slot
order, converting slots to the current generation's IDs at each visit.

The audit covered every numeric ID consumer (`valid`, `dptr`, `descriptor`,
`mark_graph`, `backward_seed`), owner/version equality checks and workspace
header offsets. Independent review specifically checked the `keep=255` endpoint:
clearing uses a raw descriptor address, avoiding a masked `dptr(256)` wrap to
slot zero. Invalid prefix ranges or changes, retained aliases/nonleaves and
generation exhaustion are rejected before state mutation.

The final binary passed 275 native packet pairs, preserving all 221 earlier
request inputs and independently requalifying five changed Adam output files. Its independent 26-case review compared 40,123 numeric values, including
1,000 SGD and 1,000 Adam updates within individual requests. Another case
reclaimed 9,600,000 cumulative transient bytes in the unchanged 1 MiB workspace.
Ten native refusal probes compared all 1,048,568 workspace bytes other than the
error word and found no changes. Their scratch entry harness preserves all 118
nonentry production function bodies byte for byte and is not a production
debug interface. Exact identities and original isolated-component records are
in `validation/session-native.json`.

The graph-lifetime revision kept the compiler profile unchanged and used 121
functions, 1,439 blocks and 5,751 instructions. Graph reset does not increase request/response
capacities, provide cross-request persistence, general reference counting or
full upstream graph-lifetime compatibility. The new hosted gate requires 159
uniquely named real-PyTorch cases, with long SGD/Adam/AdamW sessions and preserved
gradient tests added after the unchanged previous 153 cases. A fresh hosted
receipt is required for this runtime revision.

The first 159-case hosted execution found two 512-step Adam trajectory failures,
retained with their original identities in `validation/hosted-session-failure.json`.
The correction adds stable first-moment lerp and software fused multiply-add
for the second moment. Independent FMA qualification passed 485,163 exact bit
comparisons. Six expected-response files were updated only after the original
independent optimizer/training references passed; two new 512-step regression
pairs make 275 total. The oracle inputs and thresholds remain unchanged.

The corrected [hosted run](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43742636363)
passed all 159 cases and 8,471 numerical comparisons on commit
`6d21e076f15fdd097a5715f9e58306e93b257ce1`, with 275 native packet pairs and all
oracle provenance checks. `validation/hosted-session.json` retains the full
passing report without replacing the failed record.

## Compiler capacity profile

The 121-function runtime at this stage fit the original profile, but left only
seven of its 128 function slots and 97 of its 1,536 basic blocks unused. To continue adding HIA
implementation, the repository now supplies a separately identified compiler
configuration, `heptatorch.hia.compiler.f256-b2560-w2m.v1`. It preserves the
upstream source identity and does not relabel its original limits.

| Resource | Original profile | Expanded profile |
| --- | ---: | ---: |
| Functions | 128 | 256 |
| Total blocks | 1,536 | 2,560 |
| Compiler workspace | 1 MiB | 2 MiB |
| Relative relocation entries | 12,000 | 14,336 |
| Instructions | 8,192 | 8,192 |
| Blocks per function | 256 | 256 |
| SSA names per function | 2,048 | 2,048 |
| Source and generated image | 1 MiB each | 1 MiB each |

The source adapter verifies its upstream import closure before executing it,
requires exact reproduction of the original HIA source, and applies counted
changes to private capacity/layout sites. The internal function count and
ordinal use existing 32-bit header accessors so that the exact count 256 cannot
wrap at a byte boundary. The new tables occupy bytes 4,096 through 1,515,023,
leaving 582,128 bytes at the end of the 2 MiB compiler workspace. All table
intervals, call-graph strides, dirty-workspace clear spans and relocation writers
were independently checked. The conservative relocation bound is 13,826, below
14,336; the canonical Core byte bound remains below its unchanged 8 MiB gate.

The adapted compiler remains small enough for the original pinned native
compiler to build it. A separately generated HIA packer gives only the compiler
a larger workspace. Four successive native images and executables agree byte
for byte, and a deliberately changed source propagates into a different stable
successor. The packer also reproduces identically through old and new compilers.
HeptaTorch and unhex retain their exact original executable bytes and ordinary
1 MiB application buffers. No upstream compiler or generated packer source is
vendored; source adaptation and HIA generation occur in build scratch.

The durable 16-case gate executes programs beyond the old boundaries and at the
new limits, and rejects count overflows, excessive per-function blocks or SSA
names and high-index call cycles. Separate native checks exercise all 256
explicit scalar exports, guarded buffers and a high-index 256-function graph
with a non-first entry block under both clean and dirty workspaces. The full
per-edit, fixed-point and independent records are in
`validation/compiler-profile.json`. These finite checks are not the separate
upstream full cold-qualification campaign or a formal compiler proof.

This expands one compilation unit. The upstream project already contains a
[native static ELF linker](https://github.com/TrillionniumFoundation/Hepta/blob/f33d8da316de8fdf27b504371f1a5310997ff548/docs/HIA_NATIVE_ELF_LINKER.md),
but the pinned compiler does not admit general external function imports or
resource-bearing public tensor interfaces. Its
[Core module route](https://github.com/TrillionniumFoundation/Hepta/blob/f33d8da316de8fdf27b504371f1a5310997ff548/docs/HIA_CORE_MODULE_GRAPH.md)
combines modules into one limited Core program. The later borrowed-module
profile below adds a bounded cross-unit signature/effect and linking path.
General ownership contracts, resource allocation and returns, foreign ABI,
and the remaining operator, dtype, frontend and device work still require
independent implementation and qualification.

The expanded profile also passed the complete fresh
[hosted build](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43743468220)
on commit `2bb11186c9e2bffaddd4d0e0fc88cf9997f8b52d`: source-receipt equality,
four native generations, all 16 capacity cases, original/expanded application
byte identity, 275 native packet pairs, 159 real-PyTorch cases and all oracle
provenance checks. `validation/hosted-compiler-profile.json` retains the complete
result; the original runtime and failed Adam records remain separately bound.

## Tensor composition and indexed addition audit

This batch adds cat, stack, scatter_add and index_add to the native binary32
runtime. The merged authoritative packet source is
`264e9c5c4e8ea1834724cc0d8d859952016daffe4af330a8daa7745e7a5e5233`,
with monolithic reference executable
`5b992e5f07fe1167aacf31d9fd07a053b3b02fc1bd1149c06f8ccd164ef96bb0`.
It contains 143 functions, 1,783 blocks and 7,064 instructions. The previous
compiler, graph-lifetime and operator receipts retain their original source
and executable identities; this batch does not relabel them as fresh runs.

| Audited behavior | Implementation and qualification |
| --- | --- |
| Variable parent lists | Cat/stack save every full generation-coded ID and owner version; repeated parents accumulate gradients; list and output allocation are preflighted together |
| Cat's `(0,)` exception | Only exact rank-one `(0,)` inputs are skipped for shape matching; all-skipped lists accept every signed 64-bit axis, including both endpoints |
| Strided composition | Logical copying and VJP support transposed inputs and seeds; output channels-last selection changes actual physical writes as well as metadata |
| Stack layout | Last-axis insertion for rank-three inputs follows the virtual-unsqueeze memory-format rule; other admitted paths use ordinary contiguous output |
| Empty metadata | Generic descriptor strides multiply by `max(dim,1)`; every dimension is checked before zero-aware numel admission, including late zeros after large prefixes |
| Indexed accumulation | Exact i64 configurations, repeated indices, shared parents, source/self tracking, noncontiguous inputs and graph-reset reuse are implemented |
| Delayed source refusals | Scatter's returned source-gradient shape and empty multidimensional index_add's deferred index bounds are checked during graph preflight, before gradient/work changes |
| Alpha arithmetic | All binary32 encodings are admitted with zero upper word bits; rank-above-one index_add uses native software FMA, while rank-zero/one retains separate multiply/add for the pinned measured CPU profile |

The empty descriptor correction is global, not a cat-specific metadata
workaround. A shape containing zero does not allocate the product of its earlier
dimensions, but an invalid dimension still fails. With rank at most four and
each dimension at most `2^18`, the product of `max(dim,1)` for an empty shape is
at most `2^54`. Both the address arithmetic and canonical empty strides remain
within u64. Exactly one earlier fixture response changes: byte 280 of
`session-implementation-reset-empty-owned-leaf` changes from zero to one,
correcting strides for `[2,0]` from `[0,1]` to `[1,1]`. Its original request and
the earlier failed comparison remain preserved.

Pinned source review covers
[TensorShape.cpp](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/TensorShape.cpp),
[MemoryFormat.h](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/c10/core/MemoryFormat.h),
[TensorImpl.h](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/c10/core/TensorImpl.h),
[FunctionsManual.cpp](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/FunctionsManual.cpp),
[ScatterGatherChecks.h](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/ScatterGatherChecks.h),
[TensorAdvancedIndexing.cpp](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/TensorAdvancedIndexing.cpp),
[input_metadata.cpp](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/input_metadata.cpp),
and [ufunc/add.h](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/ufunc/add.h).
The [protocol](protocol.md#concatenation-and-stacking) records exact command
fields and error phases; [numerics](numerics.md#indexed-addition-and-alpha-rounding)
records the arithmetic distinction and its CPU/ISA qualification boundary.

Actual locked-wheel probes observed 22 empty-tensor cases, 12 indexed boundary
cases, six alpha-rounding cases and eight layout cases. Their original JSON
observations, CI runs and source identities remain in
[hosted boundary observations](../validation/hosted-boundary-observations.json).
The wheel is `2.15.0+cpu` at build commit
`d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`; it is not relabeled as the inventory
commit. The [oracle lock](../compatibility/pytorch-oracle-lock.json) separately
records the verified lint-only release-source difference.

The merged monolithic reference executable passed these independent checks:

| Check | Executed cases | Compared numeric values |
| --- | ---: | ---: |
| Cat/stack shape, copy, VJP, error, capacity and lifetime checks | 81 | 1,673 |
| Cat/stack failure atomicity | 16 | Gradient/work bytes and allocation cursors |
| General composition, including matmul and repeated graph reset | 64 | 3,738 |
| Scatter/index_add coordinate, error and lifetime checks | 100 | 1,553 |
| Indexed addition special values and shared transpose views | 9 | 1,652 |
| Replay of actual indexed boundary observations | 12 | 64 |
| Replay of actual alpha observations | 6 | 102 |
| Replay of actual layout observations | 8 | 924 |
| Cat/scatter/stack/index_add cross-feature chains | 3 | 102 |

These are distinct evidence sets with some overlapping behaviors; their counts
are not a percentage of upstream schema coverage. The complete regression
collection is 461 pairs: 275 preceding pairs, 81 cat/stack pairs, 100 indexed
pairs, three cross-feature pairs and two additional matmul-composition pairs.
The default linked image has also passed all 461 pairs. Its executable hash
and module qualification remain separate from the monolithic review above.
The original 81 production and 16 scratch atomicity request/response pairs
remain byte-identical after merging. Reports in
[tensor composition evidence](../validation/tensor-composition-native.json)
bind the actual final replay, component identities and preserved failures.

The main hosted oracle is configured for 268 uniquely named cases:
159 preceding cases with unchanged inputs/tolerances, 35 cat/stack cases and
74 indexed-addition cases, including late-zero shape admission. The same cases
passed separately against both runtime images in
[hosted run 43788929713](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43788929713),
which tested commit `724cbf2eb19d2ef8d400615d7196f5397cb402f6`. Each build passed
268 cases, 12,932 value comparisons and all 11 exact oracle-provenance checks,
with no failures and no executable change during comparison. Both 461-pair
native suites, compiler reproduction and module gates also passed.
[The complete receipt](../validation/hosted-tensor-composition.json) records
both source-bound results and their original hosted report hashes. Historical
boundary observations and earlier runtime receipts retain their identities.

This batch retains conservative version checks, first-order reverse AD,
binary32 CPU values, rank at most four, bounded one-request storage and the
explicit packet frontend. It does not qualify full cat/indexed schemas across
all dtypes, layouts, devices, in-place/out variants or upstream frontends.
All 2,590 schema records and all 17 full-project gates remain in the target.

## Native borrowed-module bridge

A separately identified compiler/linker extension now demonstrates real
independent compilation of the complete current runtime. It derives a
63-function `ht_kernel` unit and an 80-function `ht_runtime` unit from the
143 authoritative function bodies, preserving those bodies byte for byte.
The runtime unit imports 52 declared functions from the kernel unit. Native
HIA admission and ELF relocation produce executable
`cf78556c3c512227ffe123635b76e5a6af97697606e8ba3535c56f5101f4f80d`,
which has passed all 461 packet regressions with an empty PATH and no host
numerical runtime. The independent boundary checks and exact execution receipts
belong to [native module evidence](../validation/native-modules.json).

Imports are real external declarations, not placeholder function bodies.
The admitted module boundary carries exact scalar types or borrowed `readbuf`
and `mutbuf` descriptors, with zero or one scalar/unit result. Native compiler
checks retain effect and borrowing constraints; the linker checks interface
metadata, symbol visibility, ABI agreement, relocations and an acyclic module
dependency graph. Mutable aliasing and expired loans are refusal cases, while
shared read borrowing remains admitted. The public executable still uses the
same bounded three-buffer packet entry.

This is a profile extension to the pinned Hepta source, not a claim that the
unmodified upstream compiler already had these external import capabilities.
The canonical packet source remains the only hand-maintained implementation;
generated module units and objects are derived build artifacts. The default
`make` target produces the linked `build/heptatorch` using
`HEPTA_MODULE_COMPILER` and `HEPTA_MODULE_LINKER`; `HEPTA_COMPILER` still builds
unhex and the `make monolithic` reference executable. The corresponding
`test` and `test-monolithic` targets replay the same 461 fixtures. Reproduction
commands and the host source-organization/transport boundary are documented
in [Build](build.md#reproducing-the-borrowed-module-tools).

Owned `buffer`/`chain` values, ownership transfer, allocation across the public
boundary, borrowed-resource returns and arbitrary foreign pointer provenance
remain outside this profile. The linker is not a machine-code authenticity
proof; source/compiler/object identity and build custody remain necessary.
General module ABI, scalable allocation, cross-request persistence and the
full 17-gate acceptance target therefore remain open.
