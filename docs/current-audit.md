# Current HIA rewrite audit

## Result and scope

The application now has a direct, complete **700-core-function HIA source** and
a default build for that source. The entire PyTorch replacement remains
**HOLD**. The fixed target is PyTorch 2.15.0 at
`31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`, with 2,590 exact native declarations,
688 derivative declarations and seventeen whole-system acceptance gates.
Ten gates are IN_PROGRESS and seven are OPEN; no complete upstream schema or
whole-system gate is qualified.

This audit covers the repository/branch state, implementation and build paths,
fixed upstream inventory, acceptance accounting, original execution evidence,
current-source compilation, actual finite regressions and retirement dependencies.
It does not equate an inventory entry, source preparation or component PASS with
implementation of the complete corresponding PyTorch contract.

The initial main revision was
[`deaddeebc1cbd0509048f085017b3f63fb8b9fc0`](https://github.com/TrillionniumFoundation/HeptaTorch/commit/deaddeebc1cbd0509048f085017b3f63fb8b9fc0),
the merged R4 Scalar07b work. Its latest qualified combined runtime had 658
core functions. The unmerged continuation archive at
[`8579ae2fe98fa3abcb4ec3439cb21b7192a6351a`](https://github.com/TrillionniumFoundation/HeptaTorch/commit/8579ae2fe98fa3abcb4ec3439cb21b7192a6351a)
added 108 source/evidence files without modifying production runtime files.
The 699-function integer-sum successor and 700-function Bool-output successor
therefore needed actual recovery, compilation and integration, not a change to
the full-project status flag.

## Build and integration defects addressed

The old default `make all` still selected the 143-function borrowed runtime.
The README, first build instructions and opening continuation checkpoint also
described older defaults. A green historical native workflow did not execute
the pending 699/700-function successor.

The current entry is now `tools/current_runtime/source/runtime.hepta`, with an
ordered manifest and fourteen complete compilation units. `make` builds it;
`make test` uses the same four-suite coordinator as the focused current-runtime
workflow. The older protocol's fixtures and compiler witnesses have explicit
`legacy` and `test-legacy` targets. Historical source-bound claims retain their
own identities.

The application builder compiles every unit with the final native HIA ABI3
compiler and links newly produced objects with the final HIA session linker.
It does not consume a prior application ELF, historical application objects,
or a succession of generated overlays. The compiler bootstrap remains a
separate process; its original Python/GNU-ld bootstrap and historical compiler
witnesses are not Tensor execution dependencies.

| Current build property | Actual value |
| --- | --- |
| Core HIA functions / blocks / instructions | 700 / 9,339 / 33,425 |
| HIA compilation units | 14 |
| Native compiler/linker processes | 15, each exit 0 with both EOFs |
| Tensor descriptor | 672 bytes |
| Native ELF size | 3,620,864 bytes |
| Maximum native call path | 131,824 bytes |
| Call path including unchanged reserve | 135,920 bytes |
| Complete source SHA-256 | `b83a8346b162e12c5ed77e0287d626619a18f84bf7a1ab02a2514d30ae4295b4` |
| Native ELF SHA-256 | `c6afeedc14ba8cc408a62d6fbef0abb5b78dff0039b8099f3449d813e290a90e` |

The same direct builder also reproduced every original object and final ELF
for the 658- and 699-function profiles. Eight source controls rejected altered
core bodies/support code, missing units/exports, mismatched imported effects,
cyclic imports, path traversal and incorrect hashes. Default Make publication
was exercised with a real cold build, a rejected compiler and a missing output:
failures preserve their staging evidence and the prior successful build;
grouped outputs rebuild together.

See [the complete build contract](current-runtime.md),
[the source manifest](../tools/current_runtime/source/manifest.json) and
[local execution receipts](../validation/current-runtime-local/manifest.json).

## Evidence recovered and current-image qualification

The original 699-function Native24 process was already complete. It was not
restarted. Its full 5,722,831-byte original report was recovered with SHA-256
`5f8f9a7b4dbf6455c628a78b392c79d06ff83beecb9eca0caa940f91b85e700b`.
Of its 460 original input files, 457 were still present with exact identities.
Three missing application ELFs were restored with exactly their original
bytes: two from original link output and R4 from a fresh direct cold build
that reproduced all thirteen original objects and its original ELF. The first
incomplete recovery attempt remains a recorded failure.

A new **pure recorded admission** then recomputed all 24 original case records,
260 reference events and five bounded controls without starting native Tensor,
Torch or compiler execution. It reconstructed the complete original top report,
compared all original fields and verified all 460 inputs. Its status is
`PASS_PURE_RECORDED_ADMISSION`; `full_general_transition_recomputed=false` and
the whole-project HOLD remain unchanged. This is distinct from the following
fresh 700-image executions.

| Current-image suite | Actual local execution and comparisons |
| --- | --- |
| Bool27 | **27/27 PASS**, 6,123 native frames; fixed actual-Torch cases; seven dtype controls; ten entry refusals; real output-Storage allocation failure, full rollback and successful retry after restoring the resource limit |
| Original view/scalar76 | **76/76 PASS**, 82,503 frames and 481,580 assertions; all original 26 + 32 + 14 + 4 cases and comparison bodies; actual ownership/weak-lifetime, Version, RNG, numeric and rollback checks |
| Original F32 Sum32 | **32/32 PASS**, 626/626 events, 64,415 frames and 460,075 assertions; complete original reference admission and comparison class; no case filter or substituted expected arithmetic |
| Original Integral24 | **24/24 PASS**, 260/260 events and 14,512 native frames; unchanged comparison and reference admission, full-width integer data, overflow, empty/noncontiguous layouts and lifetime checks |

The four case counts total 159. Control counts, frames, assertions, exact words,
RNG bytes and rollback bytes have different denominators and must not be added
or relabeled as complete schema coverage. The current CI actually starts all
four suites against one cold-build identity; missing, skipped, prepared-only,
failed or incomplete reports cannot produce its focused PASS. The local receipt
manifest and hosted run state specify which executions have actually completed.

The new Bool reader's first run recorded 12 passing and 15 failing cases. It had
used old view command 69 instead of current command 111 and mapped IndexError
to cause 7 instead of cause 2. Only those two new reader choices were corrected;
the HIA source, ELF, Torch reference and all tensor/owner assertions stayed fixed.
The original failure, original reader files and correction record are retained.

Two of the original five integer-sum dtype controls expected explicit Bool
output to be refused. That expectation is intentionally superseded by this
native implementation. Their actual original result is still FAIL with
three unchanged refusals passing and indices 0/2 reporting the new successful
operation. The current seven-case dtype suite tests the new contract. No report
claims the original five all passed on the changed contract.

The original 39,045,926-byte F32 reference was missing at its old standalone path.
It was recovered from the previously sealed archive member
`inputs/0192-actual.json` and matched SHA-256
`c14e4b18f657429ad5dac5498c3abcae303b6a7e7550db7d15a3c9179bdcd728`.
The shared reference pack preserves 113 original files, 47,548,385 payload bytes,
all original file modes, source manifests and environment records in a 536,900-byte
archive. It includes no prior application executable. Both original Sum32 and
Integral24 reference validators still run with explicit, verified path mappings.
See [the pack manifest](../validation/current-runtime-references/manifest.json).

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
