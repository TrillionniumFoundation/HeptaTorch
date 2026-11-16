# Numerical Tensor objects over owned Storage, version 3

This profile connects the existing binary32 tensor, autograd and optimizer
implementation to actual native Storage and Version objects. Tensor handles,
Storage lifetimes and mutation counters are independent. Views and detach share
Storage and Version; clone creates new identities; a data alias shares Storage
with a fresh Version. Saved values keep the data required by the derivative,
while graph metadata can survive without unnecessarily retaining a nonleaf's
payload. Released interior allocation extents can be reused without making old
public handles valid again.

The native process reads a frame, returns its response while stdin remains open,
and retains one HIA-owned state buffer for the next frame. Native HIA performs
allocation, migration, reference updates, numerical execution and rollback.
Python files in this directory author checked HIA or perform transport and tests;
they are not a production numerical or storage runtime.

This is a separate, finite CPU protocol. It does not complete the full PyTorch
rewrite or change the frozen version-1/version-2 runtimes and their evidence.
The upstream denominator remains 2,590; full-rewrite acceptance stays open.

## Rebuild from the authoritative numerical source

The only authoritative original numerical source remains
`src/heptatorch.hepta`, SHA256
`264e9c5c4e8ea1834724cc0d8d859952016daffe4af330a8daa7745e7a5e5233`.
The build reads it, the previously qualified v2 overlay, the original
`tools/storage_objects` HIA authoring, and the separately qualified
`tools/owned_tensor_versions/adam_versions.hepta` correction. This package does
not vendor a second full copy of the numerical source.

The generated core has 237 functions, 2,785 blocks and 10,863 instructions.
Ninety-nine of the original 143 function bodies are byte-identical; 44 have
explicit overlays. The three-function owned transaction module is separate.
The total application is split into five independently admitted native modules,
ordered so that both the function graph and module dependency graph are acyclic.
The largest unit has 930 blocks and 3,586 instructions. Compiler admission limits
are unchanged.

Build the frozen v2 baseline into `build/owned-tensors` first, following
`tools/owned_tensors/README.md`. A fresh v3 ABI2 output directory is then built as
follows:

```sh
python3 -I -S -B tools/tensor_objects/build.py \
  --repository /absolute/path/HeptaTorch \
  --source-root /absolute/path/pinned-hia-source \
  --baseline /absolute/path/HeptaTorch/build/owned-tensors \
  --compiler /absolute/path/owned-module-compiler \
  --linker /absolute/path/owned-session-linker \
  --output /absolute/path/HeptaTorch/build/tensor-objects-abi2
```

The pinned upstream HIA checkout is
`f33d8da316de8fdf27b504371f1a5310997ff548`. The builder verifies its imported
authoring dependencies before importing them. It checks all repository
dependencies, generated unit sources, actual native object files, the link
request and final executable, and rechecks input/producer snapshots afterward.
The native tools run with an empty `PATH`.

The independent ABI3 profile applies the native whole-program stack admission
gate and starts on a guarded 8 MiB private stack. Use the same inputs with
`--stack-profile`, the pinned `stack-module-compiler` and `stack-linker-session`,
and a different fresh output directory. The linker request magic is explicitly
`HIASTKS1`; ABI2 uses `HIAOWNS1`. Both request profiles validate the typed
`step(readbuf,mutbuf,buffer)->(buffer,u32)` and `close(buffer)->()` exports.

| Identity | SHA256 |
| --- | --- |
| Generated v3 core | `4e2cc7f7612c826ec35280432be527cf98075af7d0f2c35a560df67e35c3904e` |
| ABI2 compiler | `0ccdb524ea327ab290b2e70feaac49a4789abb2fee6da8746a095f435c17af73` |
| ABI2 linker | `55a2b0a762c8db574a65d0d0e6a18e3cedc668c01d20080a11c26cf1ccb601b3` |
| ABI2 executable | `4bd22ab115a5428be6ee1ab77fd31a1e20baf14a80756eb38f000574cbad531f` |
| ABI3 compiler | `b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f` |
| ABI3 linker | `aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16` |
| ABI3 executable | `2b39ff1a5a258c5fae7be1fbca5dcf0764787b43fa72ad8fdc3ab6fd119e2a91` |

Both executables are 1,093,632 bytes. Their application bytes from offset 4,096
onward are identical; startup and interface metadata distinguish the two
profiles. `profile.json` binds every source/object hash. `build.json` records the
actual chosen profile. `--source-only` generates checked source without claiming
that native compilation or execution occurred.

## Ownership and differentiation

The allocator uses six separately typed record tables and a movable data pool.
Tensor, Storage, Version, Weak, Saved and StrongHold handles carry kind, slot and
generation; no handle contains a process pointer. Every data address within the
numerical kernels is a pool-relative offset resolved against the current state.
Growing tables or the pool therefore preserves existing public identities.

The state starts with 64 slots per record kind and a 128 KiB pool. Tables grow to
at most 65,536 slots per kind; the pool grows to at most 1 GiB. Node metadata is
paged separately inside that owned pool. The numerical node sequence still has
a 65,535-node bound until graph reset. Per-kind table limits and live Saved/Hold
requirements also constrain how many tensors can coexist. The profile does not
promise 65,535 simultaneously live gradient leaves regardless of other objects.

One checked `buffer` owns all records and payloads. Each frame executes against
a newly allocated trial state. Success drops the original owner and returns the
trial. Failure drops the trial and returns the exact original owner; a canonical
32-byte error response exposes no trial IDs. Only explicit allocator capacity
hints permit grow-and-replay. Shape, work-budget and output-capacity refusals do
not repeatedly increase memory. This transaction strategy is valid because no
external heap owner is hidden in the copied state.

Public Tensor release revokes the handle immediately. A graph's required leaf
keeps its private Tensor/Storage obligation for gradient accumulation. Other
graph nodes retain shape and routing metadata separately from payload ownership.
Actual derivative Saved records retain the required Storage and Version. Two
uses of the same input can create two Saved obligations. Public Saved objects
also retain their own layout, so logical reads and unpack remain valid after the
original Tensor is released or the graph is reset.

The reported strong/weak/Version reference counts are this protocol's explicit
ownership obligations. **They are not claimed equal to PyTorch's internal
`intrusive_ptr::use_count()`.** PyTorch may retain a TensorImpl where this runtime
directly retains Storage and Version. Qualified compatibility concerns identity,
required liveness, stale-handle rejection, mutation checks and numerical results.

The Saved policy follows the pinned derivative definitions and code generator,
not only the minimal mathematical inputs. ReLU saves/reads its output. Gather
and pooling save their input; index-add saves source when its derivative is
needed. Add, clone and transpose retain gradient metadata without saving input
values. Dense matmul distinguishes a saved view from a materialized fold/reshape
copy. Materialized copies are real private Storage objects; backward reads their
old values after a permitted mutation of the original tensor.

Backward retains the graph until explicit public-root release or reset. This is
the protocol's existing repeated-backward policy, not PyTorch's default
`retain_graph=False` lifetime. Public Saved unpack restores shared Storage,
Version and layout into a non-gradient Tensor; it does not reconstruct all
SavedVariable autograd metadata.

## Qualification and preserved failures

The final ABI2 candidate passes the unchanged independent object model,
private-handle/migration boundaries and six replayed hosted matmul observations:
22 cases, 572 interactive frames, 3,771 exact scalar/binary32 fields and
35,080,752 bytes of whole-owner comparisons across failures. The independent
model recomputes obligations from roots and graph reachability instead of
copying the runtime's incremental reference algorithm.

An additional 11 auxiliary-lifetime cases cover convolution, pooling, saved
indices, concatenation, scatter/index-add and saved outputs: 140 frames and 87
exact binary32 values. They release public outputs before backward and require
all expected payloads/auxiliaries to return to the original free-byte and live
record counts after the final graph root is released.

The result-reference suite includes same-frame aliases, Saved and optimizer
states, typed rejection, rollback, and all 16,383 legal command slots. It uses
18 frames and 16,410 exact numeric payload values. The unchanged canonical
`numeric-specials` fixture additionally contributes 894 byte-exact scalar
results. Production numerical arithmetic is never provided by the test client.

```sh
python3 -I -S -B tools/tensor_objects/test_aux_lifetimes.py \
  --executable /absolute/path/tensor-object-session --output /fresh/aux-review
python3 -I -S -B tools/tensor_objects/test_request_references.py \
  --executable /absolute/path/tensor-object-session --output /fresh/ref-review
python3 -I -S -B tools/tensor_objects/test_scalar_compatibility.py \
  --executable /absolute/path/tensor-object-session \
  --input-fixture tests/fixtures/numeric-specials.input.hex \
  --output-fixture tests/fixtures/numeric-specials.output.hex \
  --output /fresh/scalar-review
```

These test scripts require the separately frozen sibling
`tools/owned_sessions/client.py`; its hash is checked before import. Own-child
`/proc` reads are read-only test observations and are not an execution ABI.
The separately delivered independent model/replay scripts and the actual
hosted-observation file retain their original source and report identities.

Earlier results are not relabelled. Candidate06 failed empty transposed clone
stride preservation and two materialized-matmul mutation cases; the unchanged
assertions pass in its successors. Candidate08 passed those suites but the new
reference boundary test found scalar command 14 misclassified as a Tensor
command. Candidate09 removes exactly two classification bits; the complete
failing receipt and two-constant HIA diff are preserved. The canonical pool work
guard is retained; its historical manual SSA names differ from re-emission, so
the build checks exact alpha-equivalence rather than claiming raw byte identity.

The true PyTorch 28-case v3 adapter is an independent CI task. This package's
local receipts do not claim that this new adapter has run. Six matmul tests
replay an already executed pinned Torch observation, which is different from
executing Torch locally.

## Explicit remaining scope

The public protocol and its finite bounds are in `PROTOCOL.md`. Rank is 0–4,
data is CPU binary32, each dimension is at most 262,144 and numel at most
16,777,216. Existing conservative work limits still apply. Version counters are
checked u64; upstream inference/disabled-version and atomic-uint32 behavior are
not fully reproduced. Optimizer state buffers are supplied explicitly; their
creation/replacement lifecycle is not a full `torch.optim.Optimizer` object.

This work does not add Python Tensor dispatch, general device/dtype dispatch,
all view/storage-resize APIs, foreign allocators/pointers, concurrent sessions
sharing one object graph, higher-order AD, saved-tensor hooks, or automatic
PyTorch-default graph disposal. Gradients remain the existing leaf-gradient
buffers rather than independently exposed public Tensor objects. Those limits
remain visible instead of being counted as complete upstream schema coverage.
