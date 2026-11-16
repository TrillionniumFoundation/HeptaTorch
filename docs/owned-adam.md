# Owned Adam and AdamW

This native profile adds real optimizer-owned state to the frozen SGD object
runtime. It supports dense CPU binary32 Adam and AdamW in the nondifferentiable,
noncapturable, non-foreach, nonfused path. Each participating parameter gets
actual `step`, `exp_avg` and `exp_avg_sq` Tensors, each with its own native
Tensor identity, Storage and Version. No host numerical runtime supplies the
updates.

The implementation and source mapping are frozen under
[`tools/owned_adam`](../tools/owned_adam/README.md). The complete command layout
is in [PROTOCOL.md](../tools/owned_adam/PROTOCOL.md); build and qualification
commands are in [the build guide](build.md#owned-adam-and-adamw-state-tensors).

## State is created by a participating update

Creating an Optimizer or querying its inventory does not create state. A None
gradient skips the parameter update, all Versions and state creation. Defined
zero and empty gradients participate. An existing defined gradient remains
usable when its parameter's current `requires_grad` flag is false.

Before each group's functional loop, participating entries are initialized.
Duplicate occurrences of the same Tensor share one real state immediately and
then run in occurrence order. Different Tensors sharing Storage remain
different optimizer keys. Adam moments use eligible dense or empty parameter
layout; they do not inherit an unrelated gradient's clone layout.

The step is a rank-zero float32 Tensor initialized to zero and incremented by
native binary32 arithmetic on each participating occurrence. The two moments
start at zero with fresh S/V identities. Per occurrence, step Version grows by
one, `exp_avg` by one and `exp_avg_sq` by two. Parameter Version grows by one
in Adam; AdamW adds another increment when weight decay is nonzero. Empty and
zero-learning-rate updates still observe the logical mutator counts.

## Real identity, mutation and lifetime

Commands 75 and 77 publish the actual parameter or state Tensor. They do not
return copies or reset Versions. An Optimizer privately holds its parameter
Tensor/gradient binding and realized state Tensors. Closing a public token
therefore does not invalidate those private references. Republishing obtains
a fresh public token while stable incarnation, S/V identities and data remain
the same.

Dropping an Optimizer releases its obligations. Public aliases, computation
graphs and Saved objects independently retain the objects they need. A
state alias can outlive its Optimizer; a Saved reference detects later
mutation of its Version. The internal logical reference counts are checked
against these obligations and are not advertised as the exact C++ intrusive
pointer reference counts.

Command 95 provides real no-grad scalar in-place addition for the exposed
binary32 representation. It uses logical Tensor indexing and physical strides,
then increments the shared Version once, including an empty Tensor. The
fixed actual-reference mutation recipes use this native operation. Complete
public `Tensor.add_` autograd history/rebasing semantics remain open.

Ordinary failures roll back the complete frame, including earlier successful
mutations and lazy allocations. Existing structured partial-backward command
58 retains its separate commit contract. This does not claim arbitrary Python
optimizer exception-side-effect equivalence.

## Source and executed evidence

Candidate02 consumes the exact Optimizer08 core. Of 315 previous functions,
308 remain byte-identical; seven dispatcher/tag/audit functions change and
37 functions are added. The owned numerical loop is the qualified
`adam_element` with a new name and four controlled accesses: actual gradient
read, maximize-before-decay dispatch and strided addressing for both moments.
The inverse edit restores the original body byte for byte. The floating-point
operations and their previously qualified rounding behavior are unchanged.

Eight complete HIA modules build the executable
`fa92356954dbb9e0c454f2a58c2f78aa33950103d377c57e7b659ec9904183af`
from generated core
`253438648a437a75ef2dbc77138eca74cb5fe6a6e2ea322c7c38e5a535e5d2a6`.
The compiler and linker are the unchanged guarded ABI3 tools. Actual complete
modules and the linked call graph pass their native checks; aggregate
host-authoring admission beyond its existing budget is not claimed.

| Evidence | Actual result |
| --- | --- |
| [Root build and qualification](../validation/owned-adam-root-native.json) | Exact cold reproduction and 19 native reports passed. |
| [Independent new Adam recipes](../validation/adam-object-review/adam21-24.json) | 4 exposed-scope recipes, 908 frames, 3,469 exact comparisons; 210 parameter/moment values bit-identical under the unchanged numerical policy. |
| [Combined reference ledger](../validation/adam-object-review/report.json) | 22 exposed-scope qualified, one partial retained-nonleaf recipe, one public-gradient-object recipe open. |
| [Producer controls and regressions](../validation/owned-adam/qualification.json) | Ownership, 900 state Tensors, real allocation failure, float32 step boundaries, old numerical/RNG suites and legal owner transfer qualified. |
| New final-ELF live32 and frozen35 CI | Configured; actual hosted results pending. |

The 24-case reference was executed by the pinned real PyTorch profile with
all 11 provenance checks. Local replay executes native HIA against those
fixed observations; it does not claim a second local Torch run. The original
18/1/5 report and older explicit-state lifecycle observations remain unchanged.
Repeated regressions are not counted as new schema coverage.

The private state tags are `HTADAM01` and `HTOADAM1`; outer framing and the
`HTORCH03`/`HTORSP03` wire remain unchanged. A qualification-only native
controller passes real old and new owners between the actual checked steps.
Both wrong-format directions reject while preserving all owner bytes, then
resume correctly through the matching step. Cross-profile close releases the
allocation while the process is still alive. No arbitrary native pointer or
fabricated owner is accepted in that test.

## Explicit unfinished scope

Bias correction currently admits an actual float32 incremented step only
when it is a positive integer below 2^32. Assigned +0 or -0 resumes at one;
the normal 2^24 float32 counting stagnation still increments Version. Invalid
incremented fractional, nonpositive, nonfinite or out-of-range values cause
status 7 with whole-frame rollback. Arbitrary real-exponent bias correction
remains unfinished; values are not silently truncated.

The profile retains limits of 1,024 groups and 4,096 occurrences per Optimizer,
65,535 descriptor ordinals, 16,777,216 values per Tensor and a 1 MiB frame.
AMSGrad, other dtypes/devices, sparse/complex, differentiable/capturable and
foreach/fused modes, hooks, serialization, arbitrary state replacement and
general optimizer exception partial effects remain open. The independently
developing public `.grad` Tensor implementation has not been composed into
this frozen profile. Full PyTorch coverage and the existing 2,590-schema /
17-gate acceptance plan remain unfinished.
