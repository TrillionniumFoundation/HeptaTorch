# Command70 and nullable gradient storage

This new isolated component derives from the frozen materialization05 core
`9feab47afea6723284e15a2ad6c9071dc8c5bdc27b572d2eef4ab4628ed1aea9`.
The source, toolchain and wire behavior of that predecessor remain frozen.
Commands60 through69 remain reserved for the independent Tensor RNG component;
this component does not implement or accept them.

## Command70

The command is eight little-endian u64 words:

| Field | Meaning |
| --- | --- |
| 0 | Opcode70 |
| 1 | Public differentiable leaf Tensor handle, or an existing request-local Tensor result reference |
| 2..7 | Reserved; must be zero |

The command clears the leaf's gradient to `None`: it releases the private
gradient Hold, clears the gradient address/Hold fields, and clears gradient
presence. Success returns scalar1 and no payload, including if already absent.
It does not alter the parameter data, parameter Storage/Version, graph Saved
variables, optimizer state Tensors, or work data. Invalid handles are status1;
nonleaf or non-differentiable Tensors are status5, as with existing command8;
nonzero reserved words are status7. Frame rejection remains transactional.

Command8 retains its zero-existing-gradient behavior. On `None` it is a no-op
and never reads or writes a null gradient address. Commands11 and40 retain
their existing compatibility zero-value readout when no gradient is present;
command59 remains the authoritative presence query. A frontend must use59
to distinguish `None` from a defined all-zero gradient.

## New private-state invariant

The internal state tag is **HTSTATE6**. Requests and responses retain
HTORCH03 / HTORSP03 framing; command70 is available only in this component.
Old HTSTATE5 and new HTSTATE6 nonempty owners must be rejected by the opposite
`step` with status7, before any clone or mutation. Empty live ownership may
initialize either component. `close` consumes the unique owner independent
of the tensor-state version, as in the existing owned entry contract.

At every committed frame boundary:

- A leaf with no gradient has presence0, gradient address0 and gradient Hold0.
- A leaf with a defined gradient has presence1 and a valid private gradient
  Hold/address, including an empty gradient's zero-length Storage.
- Nonleaf gradient storage is absent; their transient backward work remains
  separate from a user-visible leaf gradient.

Tensor construction does not allocate a gradient buffer. Backward admission
reserves the reachable leaves' missing gradient buffers before any node
executes. The ready leaf's first materialization uses the exact word-copy rule
from materialization05. A command58 runtime exception keeps completed leaf
gradients and releases reserves for leaves that remain `None`. Ordinary frame
errors discard the trial owner. The established node336/368/376 staging fields
remain zero at every committed boundary.

This component does not expose `.grad` as a Tensor object or assign a new
Version counter to an existing object. External gradient aliases and true
Optimizer-owned lazy state remain separate work. Actual pinned Torch
observations for this interface will be recorded separately from the frozen
13,16,8 reference reports; no unexecuted comparison is marked passed here.
