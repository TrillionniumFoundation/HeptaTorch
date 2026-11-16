# Independent numeric Tensor object review

The current tested successor is candidate 08, native SHA256
`f300613bdbcc15fb493b94451b3be9b210c4dc3ac61b8ee12660ddee469fac01`,
with HIA core SHA256
`e39137c399dee247163fe9197cd18a5134aa4c1e948f13c82e2e09704314f1a9`.
All three independent native suites pass. The earlier candidate 06 failures are
retained with their original inputs, outputs, executable identity, and executed
test sources.

| Evidence | Candidate 06 | Candidate 08 | Meaning |
|---|---:|---:|---|
| Declarative object/numerical model | 12/13 pass | 13/13 pass; 433 frames; 3,252 fields | Exact rational small fixtures and explicit protocol lifetime obligations |
| Private handles, migration, allocation rollback | 3/3 pass | 3/3 pass; 68 frames; 93 fields | Native process, own-child read-only observation, actual address limit |
| Replay of six hosted matmul observations | 4/6 pass | 6/6 pass; 71 frames; 426 fields | Previously executed pinned PyTorch results; no new Torch execution in the replay |
| Hosted PyTorch lifetime probe | 22 observations, zero probe errors | Same frozen report | Real pinned Torch; all eleven provenance checks pass |

Fields include scalar metadata, counts, and exact binary32 words. They are not a
count of upstream tests or a count of unique numerical values from PyTorch.
Whole-owner comparisons across failed frames total 11,011,792 bytes in the model
suite, 23,176,112 bytes in the boundary suite, and 892,848 bytes in the matmul
replay. These totals sum repeated comparisons of committed state; they are not
unique allocated memory.

## What is independently checked

The reference model recomputes ownership from sets of public roots, reachable
gradient nodes, saved-value obligations, Weak objects, and Holds. It does not
copy the runtime's retain/release updates or free-list algorithm. Its numerical
reference uses exact rational arithmetic for small dyadic fixtures; every
expected value is checked to be exactly representable as binary32 before it is
serialized. No test model is part of the HIA execution path.

The checked relationships include separate Tensor, Storage, and Version
identities; shared Storage/Version for transpose and detach; new Storage and
Version for clone; shared Storage with a distinct Version for the data alias;
two distinct saved-value obligations for the two inputs of `x*x`; and public
Saved objects whose independent logical layout survives graph reset.

Public Tensor release revokes its typed handle. A leaf Tensor needed by the
gradient graph can still have an internal row and live Storage while every
public read, write, metadata query, and repeated drop using the revoked handle
fails. A nonleaf payload not needed by any derivative can expire while its
gradient routing metadata remains usable. A Weak object does not resurrect an
expired payload, and an expired Storage can retain weak metadata without
retaining its independent Version.

The native counts tested here belong to this explicit object protocol. They are
not asserted equal to PyTorch's internal intrusive-pointer counts. A PyTorch
SavedVariable may retain an existing TensorImpl, whereas a native saved-value
record directly contributes a Storage and Version obligation. The tested
compatibility is identity, required liveness, mutation invalidation, and
numerical behavior in the stated cases.

The A/B/C case keeps A and C live, drops B's public handle and then its alias,
Saved, and final Hold, and allocates D into B's exact interior extent. In both
tested layouts, A's relative extent is 98,304, B and D use 98,336, and C uses
98,368. B and D have distinct typed identities. Stale public and weak handles
cannot address D.

The migration case grows the object table from 64 to 512 slots, creates 300
simultaneously live empty tensors, and grows the data pool to 4 MiB for a 2 MiB
Tensor. Original typed identities, transposed Saved reads, raw Hold reads,
nonuniform VJP, and SGD remain correct. A request for an in-range 64 MiB Tensor
under a real 64 MiB process address-space limit fails without publishing trial
handles or changing any byte of the original owner, including an earlier
mutation in that frame.

The private-handle controls first read the child process's actual state to
identify existing private AD Saved and private page/gradient/layout Holds. They
then submit those integer handles through public commands and require rejection,
including zero-length reads. The observer never writes state or supplies a
native pointer/capability to the runtime.

## Actual upstream observations and corrected failures

The upstream inventory pin is PyTorch commit
[`31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`](https://github.com/pytorch/pytorch/tree/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457).
The exact wheel/build relationship and all eleven installation/provenance checks
are preserved in `hosted-lifetime-observations.json`. Its SHA256 is
`43aadde8aef7f6406b6d0bd96f7b72fc724583e9bdcee823fa971ef8558f4d0c`.
The originating probe SHA256 is
`15c3f5f230a4cd30a42fe987b71078aa74c422b7e14b099e269bd80fede6c93a`.
The hosted run is 43801012881, job 133081011984. The initial local attempt was
actually made and recorded `NOT_RUN` because Torch is absent locally.

Three failures were measured in candidate 06 before the correction:

1. Cloning an empty transposed `(0,3,2)` Tensor changed its `(6,1,3)` strides to
   contiguous `(6,2,1)`. The hosted Torch observation preserves the original
   strides. Candidate 08 now passes the unchanged metadata assertion.
2. A noncontiguous folded matmul input was saved by original Version identity.
   Pinned Torch materializes an internal copy on this path and accepts a later
   mutation of the original input. Candidate 06 returned status 6.
3. A broadcast/reshape matmul input had the same issue. The actual hosted
   gradients use the pre-mutation values held in the internal copy.

The two matmul cases now pass the unchanged six-case replay. The HIA correction
saves a real private copy with independent Storage/Version and reads that saved
data during backward. Removing a version check while reading changed source
data would fail the numerical comparisons. Direct noncontiguous matrix multiply
and folded-view cases still reject the original-input mutation when Torch does.
The replay reproduces one Torch whole-tensor mutation with multiple native
single-element writes, so that replay does not compare the absolute number of
mutation-counter increments for those two different interfaces.

## Source basis

`upstream/custody.json` binds 20 inspected source files. Sixteen have fresh
GitHub blob identities verified against the saved bytes; four retain the prior
pinned retrieval receipts. `save-policy-review.json` covers the 30 implemented
AD opcode kinds and their generator evidence. This is a table of the current
runtime's AD kinds, not 30 complete upstream APIs.

The main source relationships are:

- [TensorImpl and VariableVersion](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/c10/core/TensorImpl.h)
  distinguish version identity from Storage identity, including view/detach and
  data-access behavior.
- [SavedVariable](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/saved_variable.cpp)
  records and checks versions, sometimes retains the original Tensor, and has an
  explicit saved-data release operation.
- [AccumulateGrad](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/functions/accumulate_grad.cpp)
  owns the gradient leaf Variable. Its
  [resource release](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/functions/accumulate_grad.h)
  resets that ownership.
- [StorageImpl](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/c10/core/StorageImpl.h)
  and [intrusive_ptr](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/c10/util/intrusive_ptr.h)
  separate last-strong payload release from surviving weak metadata.
- [Derivative definitions](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/tools/autograd/derivatives.yaml),
  [saved-variable extraction](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/tools/autograd/load_derivatives.py),
  and [generated gradient guards](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/tools/autograd/gen_variable_type.py)
  determine which actual Tensor values and versions are saved. Mathematical
  value-independence alone does not justify dropping a generated SavedVariable.
- [Dense matmul implementation](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/LinearAlgebra.cpp)
  and [dispatch registrations](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/native_functions.yaml)
  show the composite dot/mv/mm/bmm routes and materialized reshape cases.
- [Clone implementation](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/TensorFactories.cpp)
  supplies the preserve-format behavior, and the
  [storage weak API](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/StorageSharing.cpp)
  supports the real hosted payload-lifetime observations.

## Portable reproduction

Keep `reference_model.py`, `scenarios.py`, `reference-traces.json`,
`review_native_objects.py`, `review_object_boundaries.py`, and
`replay_matmul_observations.py` together. They contain no workspace-specific
runtime imports. Supply the qualified transport/own-child observer explicitly.
Use a fresh output directory for every run.

```sh
python review_native_objects.py \
  --binary /absolute/path/tensor-object-session \
  --expected-sha256 f300613bdbcc15fb493b94451b3be9b210c4dc3ac61b8ee12660ddee469fac01 \
  --transport-observer /absolute/path/client.py \
  --output /absolute/path/model-review

python review_object_boundaries.py \
  --binary /absolute/path/tensor-object-session \
  --expected-sha256 f300613bdbcc15fb493b94451b3be9b210c4dc3ac61b8ee12660ddee469fac01 \
  --transport-observer /absolute/path/client.py \
  --object-state-base 139264 \
  --output /absolute/path/boundary-review

python replay_matmul_observations.py \
  --binary /absolute/path/tensor-object-session \
  --expected-sha256 f300613bdbcc15fb493b94451b3be9b210c4dc3ac61b8ee12660ddee469fac01 \
  --transport-observer /absolute/path/client.py \
  --observations /absolute/path/hosted-lifetime-observations.json \
  --output /absolute/path/matmul-replay
```

Candidate 06's debugger state base is 8,192. Candidate 08's is 139,264 because
request-local result references have an independent 128 KiB scratch region.
This is an explicit observation-layout parameter; old reports and bytes are not
relabelled. The frozen expected traces and all numerical/refcount expectations
are unchanged between the two native candidates.

## Remaining boundaries

The runtime still exposes a bounded, versioned protocol for CPU binary32,
rank 0–4 numerical tensors. It does not implement Python Tensor objects,
arbitrary dtype/device dispatch, foreign storage, all alias/view operators,
concurrent access, or inference tensor rules. The numeric core retains its
private monotone node sequence until reset; public typed slot reuse does not
remove that independent graph capacity limit.

Backward in this protocol retains saved graph state until explicit root release
or reset. The hosted probe confirms that PyTorch's default backward releases
saved data earlier; that policy difference remains explicit. Public Saved unpack
creates a non-grad Tensor sharing Storage/Version and a copied layout. It does
not claim to restore every PyTorch SavedVariable autograd metadata field.

The native Version counter is checked u64. Full equivalence to upstream atomic
uint32 behavior, inference-disabled versions, and every optimizer state creation
lifecycle is not claimed. The pool and table are real native allocations with
checked growth and reuse, but their bounded allocator policy is not a universal
allocator/device/foreign-pointer interface.

This work does not close the full PyTorch rewrite. The upstream denominator
remains 2,590 and `full_rewrite_complete` remains false.
