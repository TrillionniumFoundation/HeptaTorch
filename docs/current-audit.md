# Current HIA rewrite audit

## Result and scope

The default application is a complete **701-core-function HIA source** with
fourteen independent compilation units. Whole-PyTorch acceptance remains
**HOLD**: the pinned PyTorch2.15.0 target contains2,590 native declarations,
688 derivative declarations and17 whole-system gates. Ten gates are IN_PROGRESS
and seven OPEN; no complete upstream schema or whole-system gate is qualified.

PR37 merged the first direct current700 build into main
`b22622b371d79478864cb80154b1782e708a5a81`. Its tested tree was
`dbbbc4b0ad0a95fd01b7ae369532bf7a10c3f0f7`; all ten exact-head checks and five
workflow runs passed. That original current700 qualification remains distinct
from every new701 execution described here.

`make` now selects the complete current source; `make test` and the focused
workflow share eight mandatory groups. The application builder consumes no
parent application ELF or historical overlay chain. The final HIA compiler and
linker are fixed; their original bootstrap and compiler witnesses remain
separate toolchain dependencies.

| Current build property | Actual value |
| --- | --- |
| Profile | `sum-f32-cat-empty-01` |
| Core functions / blocks / instructions | 701 / 9,352 / 33,473 |
| HIA units / compiler-linker processes | 14 / 15, all newly executed with exit0 and both EOFs |
| Tensor descriptor | 672 bytes |
| ELF size | 3,624,960 bytes |
| Maximum call path / with reserve | 131,824 / 135,920 bytes |
| Complete source SHA-256 | `82360ce87dae6f9a004ef50be4f991b6f83905a83f4b17ca8d6634827b7bd313` |
| ELF SHA-256 | `80445a8c433f20314d3bc7b00ebb68f6a70fa200ee2e6aba193cff21aabfbc17` |
| Original cold-build report SHA-256 | `ff065dcd35698ffbf4b456fed0740049cbd5804fec324c4a305b6ee81810c0d8` |

The builder and independent stack inspector are unchanged. Source admission
checks complete function-body partitions, imported signatures/effects,
dependencies, fixed compiler/linker limits and every actual object. A build
reports `PASS_NATIVE_BUILD_ONLY`; it cannot grant application qualification.

## Current same-image acceptance

The required total is513 cases across eight groups. Their complete original
reference/comparison contracts and separate control denominators remain intact.
Actual completion and original report hashes are recorded in
[the execution index](../validation/current-runtime701/evidence.json); a group
still in progress is not admitted as PASS.

| Group | Required cases | Relevant independent checks |
| --- | ---: | --- |
| Bool27 | 27 | Complete owner/RNG, seven original dtype observations, ten entry controls and real allocation rollback/recovery |
| Original view/scalar76 | 76 | Original26+32+14+4,82,503 frames and481,580 assertions |
| Original F32 Sum32 | 32 | Original626 reference events,64,415 frames,460,075 assertions and full owner state |
| Original Integral24 | 24 | Original260 events, full-width integer results, overflow, empty/strided inputs and lifetime |
| Bool/I64-to-F32 Sum17 | 17 | Exact original cast witnesses/results,17 cases, four axis controls, seven dtype controls, ten entry checks and two actual allocation failures |
| Original live35 | 35 | 6,344 frames,2,590,274 values, eleven provenance checks and independent4/18 copied controls |
| Original live270 | 270 | All ten original groups,270 actual processes,284 frames,13,076 values and fourteen true refusals |
| Original live32 | 32 | 2,132 frames,17,455 exact binary32 words,1,137,600 Generator-state bytes,32 native children and all original structural/partial-error/EOF checks |

Old current700 controls are not rewritten. Bool27 retains two original FAIL
results at dtype indices2/3. Integral24 retains four original FAIL results at
indices0/1/2/3. The strict outer qualification requires these exact, explained
extension results, all original numeric cases and all remaining refusals. The
new Sum17 dtype/entry/OOM suite tests the implemented F32 output contract.

The live270 numerical oracle was previously exercised only on the old143-function
application. All143 names exist in the new source, but79 whole function bodies
had changed; old PASS results could not qualify this image. The current bridge
uses genuine result references and returned typed Tensor identities. It never
predicts native slots from the old client's logical handle counter. Four
transport comparison functions have complete inverse proofs; all other case
functions, Torch inputs/expectations, tolerances and seeds are unchanged.

For14 original refusal cases, the current transactional owner emits only a
32-byte count0 failure. The bridge executes the genuine success prefix and
terminal refusal in the same process, retains both real frames and requires
complete owner rollback. No old error record is fabricated. It does not claim
an independent complete private-allocation transition ledger for each operation
inside a long successful multi-reset request.

## Actual failures found and repaired

The first701 source added `sf_sum_f32`: Bool/I64 input is converted element by
element to F32 with its format preserved, then reduced by the shared F32 kernel.
The private conversion Tensor/Storage/Version is reclaimed on success and on
failure. Integer summation followed by a final cast is not an equivalent
implementation and is not used.

The first17 native run passed15 and failed2. A separate real diagnostic confirmed
that command108 conversion matched the original Torch cast bytes while the
subsequent F32 sum reproduced the numerical difference. The fixed wheel
advertises AVX512, but its pinned SumKernel registration supplies no AVX512
implementation and dispatches to AVX2 with8 F32 lanes. Six constants in
`sd_vector_inner` and `sd_loop2d` were corrected to that actual accumulation
order; references and tolerances were unchanged. The corrected r2 image passed
all211 then-connected cases. Its separate raw17 audit checked31 children,
8,118 frames,2,598 complete owner blobs and62 EOFs without executing native or
Torch again. These r2 results retain their own source/image identities.

The first live270 run on r2 passed269 and failed `cat-skipped-only-grad`, with
a genuine `status=9,count=0` response. The skipped one-dimensional empty cat
parent incorrectly set incoming mode1. Its second duplicate edge then reached
the shared-cache protection instead of the ordinary accumulation path. The
final source changes only this branch to mode2, preserving return1 and every
guard. The other700 complete r2 function bodies are unchanged. A new actual
live270 run on the final image passed all270 cases and284 frames.

The complete source proof reverses final701→r2→first701→original700. Relative
to original700,696 bodies are unchanged, four are modified and one is added.
The separate reader proof first binds actual new physical source files, then
executes the original proofs on explicitly verified inverse bytes. It does not
pretend reconstructed bytes were physical file observations.

Original failed execution reports, the first Integral path-admission failure,
diagnostics and all available raw evidence remain unchanged. The r2 complete
seal contains36,712 files /936,547,734 payload bytes; its152,885,096-byte archive
has SHA-256 `c87dac4b653d41cab69c3517445dce2b3467b75f87ba8c42ad1b5dc4c77f7cdb`.
Its original270 FAIL is still FAIL. An earlier local first701 Sum32 run lost
during an environment reset remains `UNKNOWN_AFTER_ENVIRONMENT_RESET`; later
runs are distinct executions, never recovery of that lost run.

The original699 Native24 report was recovered and admitted from its original
bytes without rerunning native or Torch. That earlier pure-recorded admission
and the first current700 local/hosted results remain at
[`validation/current-runtime-local/`](../validation/current-runtime-local/).
Shared original Sum32/Integral24 references remain the exact113-file pack
documented by [its manifest](../validation/current-runtime-references/manifest.json).
The new17 reference is its original actual Torch production, not regenerated
from native output.

## Remaining whole-system gates

The following table summarizes the fixed ledger. “Implemented foundation” means
finite source-bound work exists; it does not assert completion of every dtype,
layout, device, overload or error contract in that row.

| Gate | Ledger state | Implemented foundation | Required remaining work |
| --- | --- | --- | --- |
| module-abi | IN_PROGRESS | Native modular compilation, owned resource interfaces, native linking and stack admission | General public Tensor ABI, foreign-resource provenance and unrestricted module/resource graphs |
| storage-lifetime | IN_PROGRESS | Owned Tensor/Storage/Version records, strong/weak/saved lifetime, growing pools and tested transactional failures | General external/device/concurrent allocation, all alias/lifetime modes and restart durability |
| dtype-layout | IN_PROGRESS | Qualified binary32 operations, separate scalar dtype kernels, scoped I64/Bool Tensor operations and layouts | Complete dtype promotion/casting, complex/quantized/sparse/nested layouts and device variants |
| native-schemas | IN_PROGRESS | Arithmetic, reductions, composition, indexing, selected NN and optimizer kernels | Full contracts and executable qualification for all 2,590 frozen declarations |
| views-mutation | IN_PROGRESS | R4 view/reshape/scalar paths, shared versions, tracked history and failure observations | General alias/mutation rules across all layouts, ranks, dtypes and devices |
| autodiff-transforms | IN_PROGRESS | First-order reverse AD, saved versions, public/retained gradients and finite training regressions | Higher-order/create_graph, forward AD, hooks, transforms and all differentiable schema cases |
| frontend-dispatch | OPEN | Bounded native binary command interface | Full frontend, dispatcher, overload, context and public API compatibility |
| neural-networks | IN_PROGRESS | Selected Linear, losses, normalization, convolution and pooling kernels | Complete modules, state, training/evaluation modes and all operator/backend variants |
| optimizers | IN_PROGRESS | Scoped SGD/momentum and owned Adam/AdamW paths | Complete optimizers/schedulers, state dictionaries, modes and dtype/device behavior |
| rng-determinism | IN_PROGRESS | Native CPU MT19937, persistent Generators, selected Tensor distributions and dropout | All distributions/devices, shared/concurrent state and deterministic execution controls |
| serialization | OPEN | No complete checkpoint implementation | Formats, state/storage alias preservation, compatibility, interrupted/distributed restoration |
| accelerators | OPEN | Current runtime is Linux x86-64 CPU | CUDA/ROCm/MPS/XPU allocation, transfers, streams/events, kernels and backend errors |
| distributed | OPEN | No native distributed subsystem | Rendezvous, process groups, collectives, work/failure semantics, DDP/FSDP/DTensor |
| compiler-export | OPEN | Native HIA source compilation | PyTorch graph capture, symbolic shapes, compile/export, transformations and artifact compatibility |
| data-pipelines | OPEN | No native worker/data-loading subsystem | Dataset/dataloader, sampling, worker lifecycle, multiprocessing, cancellation and reproducibility |
| platform-packaging | IN_PROGRESS | Reproducible fixed Linux x86-64 native profile | Other required platforms/architectures, installation, extensions and profiling interfaces |
| final-integration | OPEN | Source-bound component/regression CI and a current application build | Full application/upstream acceptance, resource/performance/error review and verified retirement |

The separate binary64 foundation still retains its original strict Torch sqrt
differences; its independent rounding qualification does not turn those old
strict failures into PASS. Likewise, preserved distribution/optimizer reports
retain their original OPEN or partial classifications, even when later images
have additive evidence. Current-source evidence must be explicitly joined;
historical report fields must not be overwritten.

The authoritative contracts remain
[`full-acceptance.json`](../compatibility/full-acceptance.json),
[`upstream-schemas.json`](../compatibility/upstream-schemas.json) and
[`upstream-lock.json`](../compatibility/upstream-lock.json).
`make acceptance-status` verifies consistent accounting.
`make full-acceptance` deliberately fails while complete qualification is absent.

## Repository cleanup and retirement

The repository already contains HIA runtime implementations; it is not carrying
an entire second PyTorch C++ runtime that can simply be deleted to finish the
rewrite. The vendored PyTorch material is a small licensed reference subset.
Python files include builders, fixed-reference producers and independent
observers, and their existence is not evidence that Python executes Tensor
operations in the native image. Removing them would not implement the missing
subsystems listed above.

The initial main tree contained 5,578 blobs totaling 741,310,752 path bytes,
including 214 `.hepta`, 951 `.py` and no `.rs` files. There were 645 same-blob
path groups with 1,100 repeated path references totaling 56,976,602 path bytes.
These are tree/path counts, not an estimate of Git object-store savings; Git
already shares identical blobs. They were measured on the initial main tree
`ac57b80733defcd76feb16cac1f2a32f351f79d6`, before adding the current integration.

Current production selection is consolidated: one complete source package,
one default build and one focused qualification coordinator. Historical
application ELFs and overlay rebuilds are removed from that dependency graph.
The current reference package shares original inputs used by both Sum suites.
The first audited branch inventory had main, 36 already merged branch heads and
one unmerged continuation archive. Merged branch ancestry and current ref leases
must be checked again before any remote ref cleanup.

Older builders, source packages, original failed/unknown receipts and test inputs
are still required by the historical workflow or the independent acceptance
readers. Their removal is conditional on replacing those dependencies and
passing the corresponding complete current-image contracts. Identical blob
content alone does not prove that a path is redundant: original manifests and
recorded input identities can intentionally refer to both paths.

Retirement proceeds by an explicit dependency inventory: move a needed original
input into a hash/mode-bound fixture once; point current readers at its verified
location without changing comparisons; run the replacement on the current
image; preserve original failures and provenance; then remove the obsolete
production path. Deleting evidence, emptying the schema inventory, changing
statuses or disabling a gate is not retirement. The request to retain only the
final native stack remains an open final-integration obligation until the full
replacement meets those conditions.

### Verified retirement boundary in this integration

The twelve old `sum_bool_00..11.hepta` files contain exactly the700 original
core bodies, all byte-identical to the preserved original aggregate. They are
replaced by the current twelve `cat_f32_00..11.hepta` files; fixed `rng03` and
`owners` remain unchanged. Five old Bool helper files move into the verified
single baseline fixture alongside the original comparison/aggregate inputs.
The Bool compatibility entry remains explicit. This removes obsolete default
derived files while retaining the exact original input identities once.

The original live32 also passed a separate new execution on the same source
and image: 32 native children, 2,132 frames, 17,455 exact binary32 words,
1,137,600 serialized Generator-state bytes, 238 absolute-Version and 166
gradient-presence comparisons. All eleven live-Torch provenance checks,
original comparisons, full structural owner observations and child EOF checks
passed. Its complete evidence is recorded separately in the execution index.
It is now the eighth mandatory group. Thirteen old Native steps and the
corresponding step in three observer profiles are retired. The other steps,
independent qualifiers and required build dependencies remain byte-identical.

Whole historical retirement remains blocked by real consumers: the fixed
compiler bootstrap packages,461 old protocol fixtures,
Generator24, source-bound dtype/optimizer/resource suites, and the F64 oracle's
whole-source proof against `src/heptatorch.hepta`. The original76 gate now uses
the eight original reference members in the 131,848-byte compact fixture
([manifest](../validation/current-runtime76-reference/manifest.json)). Its
12,561,956-byte complete historical packet remains recoverable through the
[archive index](../validation/view-reshape-r4-native76/archive-location.json);
the sealed member was read in full and matched the original bytes before
removing the Git working-tree copy. Other remaining consumers must be migrated
and executed on the replacement image before their source paths are removed.
No whole gate was disabled and no schema denominator was reduced.
