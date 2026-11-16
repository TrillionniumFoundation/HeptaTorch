# Mixed integer autograd candidate 10

This is a new, unqualified successor of immutable I64 candidate 07. It keeps the
HTORCH03 packet, typed public IDs, 640-byte descriptor, and numerical commands
1–106. Internal formats become HTMXAD02 / HTOMXD02; old owners are refused.

Command 103 admits differentiable F32/I64 add, sub, mul, and true division.
The F32 operand may require gradients; I64 can never require gradients. Forward
promotion creates real temporary F32 tensors, while the graph saves the original
arguments and their actual Storage, Version, dtype, and Tensor identities.
Command 104 remains a no-autograd integer mutation interface and rejects a
requires-grad input. No higher-order derivative or Python exception frontend is
claimed.

Command 107 is `[107, root, argument_slot, 0, 0, 0, 0, 0]`, where argument_slot is
0 (self) or 1 (other). For a binary graph node, it publishes the real saved input
Tensor token, or returns zero for an input that was not saved. An absent slot
remains absent after release. An actually saved but released slot raises runtime
status 10; a still-saved input with a changed shared Version raises status 6.
Malformed arguments raise 7 and stale/wrong public Tensor handles raise 1.
107 uses ordinary frame rollback for rejection. Reading the accessor creates a
real public reference if necessary; callers must release it when their observer
holds only primitive metadata. It does not create a Tensor, Storage, or Version
copy, reset a counter, or substitute a promoted value.

For binary mul/div nodes with Saved inputs, descriptor +584 contains a private
Hold ID (strictly above the internal-node range) and +592 its pool-relative
address. That private 32-byte Storage records `(node0, incarnation0, node1,
incarnation1)`, with zero pairs for absent Saved slots. Each entry owns a real
strong Tensor reference through +600 and +208. Existing dropout and integer-view
roles at +584/+592 remain internal node IDs at most 65535. The audit independently
recomputes all roles, checks Saved S/V/offset identity, and clears its +616 scratch.
Saved release and graph deletion clear the owning fields before releasing the
two Tensor roots and their private Hold. Ordinary failure rolls back the complete
owner; command 58 retains its existing partial-commit behavior.

Revision 09 adds release of the owned Saved Tensor roots to the existing
gradient-detach path, after its original admission guard and before graph-edge
and Saved-data release. Revision 10 separates an I64 view's Tensor lifetime
from its remaining graph metadata: when its real Tensor row is released, its
ultimate-base strong reference is released too. A rowless metadata descriptor
keeps its layout, view flag and graph reference count, but cannot retain a
base role. A live I64 view Tensor row must retain its base role. These distinct
invariants require the revision-02 internal format barrier.

Reference: immutable real Torch 36 observations, SHA256
98f24165401f9aadfd86147113454104ae1c8e969dcc105a16b6cb83d605bdfb.
This document is an implementation contract, not a runtime PASS report.
