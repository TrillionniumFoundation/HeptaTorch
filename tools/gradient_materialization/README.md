# First gradient materialization and input-gradient accumulation

This component corrects first-gradient assignment in the frozen partial04
Tensor service. It builds native HIA over that exact predecessor, preserves
the command58 execution and exception contract, and retains command59's real
gradient-presence flag. It adds no wire opcode or numerical host runtime.

The preceding image produced 4 passes and 9 failures in the actual pinned
13-case signed-zero probe. This image passes the unchanged 13-case replay:
166 interactive frames, 131 metadata comparisons and 70 exact binary32 word
comparisons. The same 16-case actual partial-backward reference also passes,
including leaf readiness, committed partial gradients, `None`, and exception
order. Both the predecessor's failures and the successor's results retain
their own source and executable hashes.

## What changes

The original implementation initialized every work array to `+0` and added
every incoming value to that array. It also added the first leaf gradient to
an allocated zero array. Consequently, a first `-0` contribution became `+0`.
Copying every first scalar contribution globally would be incorrect for
gather: its backward result uses a zero-initialized scatter reduction, and the
actual reference expects `+0` even when the selected incoming value is `-0`.

The new implementation separates derivative construction from accumulation
into a parent's input buffer:

1. Before node execution, each unique differentiable parent receives a
   temporary staging region. Repeated parent slots share that region.
2. For a direct gradient, a per-element byte records whether that derivative
   has written the element. The first write copies the original binary32
   word; later writes add. No floating operation is introduced merely to copy
   a value.
3. For a scatter or reduction gradient, the staging values start at `+0` and
   the existing reduction arithmetic remains in use.
4. After a node successfully applies its derivative, the complete staged
   gradient is copied to an empty input buffer, or added to an existing input
   buffer. If application fails, staged output is discarded and original
   work pointers are restored.
5. A leaf with no existing gradient copies its incoming gradient and marks
   presence. A leaf with an existing gradient adds the incoming values. Empty
   gradients still become present.

The existing numerical derivative bodies are unchanged. The shared `gadd`
helper implements the per-element first-write decision only while a direct
staging region is active. This behavior is used by command58 and by the three
compatible backward entrypoints 7,29,57.

The pinned upstream CPU [`InputBuffer::add` implementation](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/input_buffer.cpp)
also distinguishes an empty slot, which takes the incoming variable, from an
existing slot, which accumulates. The independently verified Git blob and file
hash are preserved in `evidence/input-buffer-custody.json`. The actual hosted
13-case observations supply the behavioral reference for the zero-sign fix.

### Derivative staging classes

| Class | Current operations | Initialization and write rule |
| --- | --- | --- |
| Direct, complete gradient | Unary functions, clone, view, transpose, backward expansion for sum/mean and axis reductions, softmax outputs, cat/stack slices, indexed-add parent gradients | First write copies the word, further contributions add |
| Elementwise binary without shape conversion | Binary input shape exactly equals output shape | First write copies the word, further contributions add |
| Reduction or scatter construction | Broadcast binary inputs, spatial operations, matmul, index_select, gather | Zero-initialized arithmetic accumulation |

This table defines this native implementation. The 13 actual reference cases
qualify their precise recipes; they do not establish all broadcast, NaN,
backend or autograd semantics. The additional native copy test covers raw
signaling/quiet NaN payloads, infinities, subnormals and signed zeros only for
leaf, clone, view and transpose first-gradient copies. It does not claim
payload equivalence for arithmetic operators.

## Resources and state compatibility

Staging reserves five bytes per parent element: four for the derivative word
and one for its first-write bitmap. The bound for a graph is the maximum sum
across a single node's differentiable parent slots. Duplicate slots are
counted conservatively when sizing and deduplicated when used. One staging
region is reused for all nodes in the execution. The parent-size sum is
bounded before allocation; the existing owned-pool capacity and allocation
failure contracts remain enforced.

All work and staging allocation occurs before node application. Command58
uses the same temporary owned allocation for its queue and staging region.
The atomic legacy executor uses a separate temporary Hold and releases it
before returning success. A failed admission still discards the entire trial
owner. Runtime errors in command58 preserve the already completed nodes under
the frozen partial04 contract; this component does not reinterpret them as
frame-admission errors.

Node offsets336,368,376 are temporary input-presence, staging-bitmap and saved
work-pointer fields. They are zero in every committed state, including after
a command58 runtime exception. Temporary work pointers are restored before
Saved or work resources are released. The old descriptor constructor already
clears these fields; no new persistent state invariant is required. The
internal state tag remains **HTSTATE5**, and the request/response magic remains
**HTORCH03 / HTORSP03**. The owned `step`/`close` functions remain byte-identical
to partial04. Node360 continues to mean actual persistent gradient presence.

## Source custody and native limits

| Item | SHA256 |
| --- | --- |
| Frozen partial04 core | `302677af52d3e04520e6f132aaf17b98cbe5d2cf404aec040bf5144e324d2d4d` |
| Successor core | `9feab47afea6723284e15a2ad6c9071dc8c5bdc27b572d2eef4ab4628ed1aea9` |
| Successor session ELF | `4a62b796d38317e630381b7e2e57e8e7824d78c1f721e3d69bd59f59e04ca894` |
| Unchanged owned entry module source | `993651c88e69be93dea49ef473b8409b9f609e5b226c6fcec10de230ef4af99e` |
| Native ABI3 module compiler | `b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f` |
| Native ABI3 session linker | `aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16` |

The complete core contains 261 functions, 3,110 blocks and 12,038 instructions.
It is actually compiled as five core modules plus the unchanged three-function
owned entry module. The compiler's per-unit caps, object contract and linker
admission gates are unchanged. Of 253 partial04 functions,249 are byte-identical.
Only `gadd`, `backward_execute`, `partial_apply_node`, and `partial_execute`
change. Eight small native helpers are added. `source-map.json` records the
per-function identities, and `partial04-to-materialization05.diff` is the
focused generated-source difference.

The canonical 143-function numerical source remains the single underlying
authority. This package contains the new authoring and checked overlay builder;
it does not maintain a second hand-edited numerical implementation. Python is
used for authoring, wire tests and build orchestration, while runtime tensor
arithmetic, gradient handling, ownership and transport execute native HIA.

## Rebuild

First reproduce frozen `tools/backward_partial` candidate04 from its pinned
atomic-release and Tensor-object predecessors. Then run:

```sh
python3 -I -S -B tools/gradient_materialization/build.py \
  --repository REPO \
  --source-root PINNED_HIA \
  --baseline PARTIAL04_BUILD \
  --compiler ABI3_MODULE_COMPILER \
  --linker ABI3_SESSION_LINKER \
  --output FRESH_MATERIALIZATION_BUILD
```

The output includes `runtime.hepta`, the exact checked overlay authoring,
six native module sources/objects, the native link request,
`tensor-partial-session`, and `build.json`. Before and after the build, the
builder checks all selected producers, source inputs, actual unit snapshots
and emitted object bytes. The final core and native image must match the
frozen profile hashes.

## Qualification

The independent reference replayers remain in the frozen partial04 package.
Their inputs, expected values, comparison rules and tolerances are unchanged.
Pass this image and source with the hashes above through their existing CLI.

| Suite | Actual result on this image |
| --- | --- |
| 13 actual signed-zero cases | 13 PASS;166 frames;131 metadata and70 binary32-word comparisons;0 differing words |
| 16 actual ready-node/partial-failure cases | 16 PASS;606 frames;1,587 exact comparisons |
| 8 actual gradient-presence/optimizer cases | All parameter values, absolute versions, presence and skip behavior qualified;880 frames;2,088 exact and180 Adam tolerance comparisons;actual Adam bit differences0 |
| Existing partial native controls | 11 groups PASS;136 frames;2,380,928 committed-owner rollback bytes compared |
| Existing gradient-presence/bounds controls | 8 groups PASS;216 frames;10,340,848 rollback bytes;includes real allocation failure and large gradients |
| Independent legacy57 controls | 14 groups PASS;207 frames;107 exact fields;11,309,408 rollback bytes |
| New copy/temporary-state controls | 4 groups PASS;104 frames;204 exact gradient words;897 temporary fields observed zero |
| Independent committed-owner controls | 4 groups PASS;65 frames;720 temporary fields observed zero |

The independent reviewer separately reran all 13,16,8 actual-reference suites
on the same image. Those receipts remain separate from the implementation
replays and are not summed as additional unique reference coverage.

Run the new narrow qualification with:

```sh
python3 -I -S -B tools/gradient_materialization/test_materialization.py \
  --binary FRESH_MATERIALIZATION_BUILD/tensor-partial-session \
  --expected-sha256 4a62b796d38317e630381b7e2e57e8e7824d78c1f721e3d69bd59f59e04ca894 \
  --source FRESH_MATERIALIZATION_BUILD/runtime.hepta \
  --expected-source-sha256 9feab47afea6723284e15a2ad6c9071dc8c5bdc27b572d2eef4ab4628ed1aea9 \
  --protocol tools/tensor_objects/protocol.py \
  --output FRESH_QUALIFICATION_OUTPUT
```

## Remaining boundaries

The actual eight-case optimizer reference still records eight momentum-state
version observations and65 explicit-state-versus-lazy-creation observations.
They remain open. Optimizer state Tensors and step counters are still supplied
explicitly; this is not yet an Optimizer object with lazy-owned state. There is
no `zero_grad(set_to_none=True)` API in this component. Arbitrary higher-order
autograd, all broadcast accumulation paths, all dtype/backend combinations,
and full PyTorch API compatibility are not established by these cases. The
2,590-schema rewrite denominator and repository-wide unresolved gates remain
in force.
