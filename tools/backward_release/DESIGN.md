# Explicit default backward and Saved release

This isolated successor reads the frozen numerical object candidate09. Its
original package, source, executable and all previous receipts remain unchanged.

New command57 takes `output,seed_or_zero,flags`. Flags0 requests default backward
with Saved release; flags1 requests `retain_graph=True`. Other flags fail7.
The existing command7 and command29 retain their previous graph behavior. The
request/response envelope remains version3, so prior requests can be replayed
unchanged against this separately pinned successor.

The owned numerical state has a separate internal tag, `HTSTATE4`. The frozen
candidate09 uses `HTSTATE3`. Both step functions accept a live empty buffer, but
reject the other's nonempty state with status7 before cloning or mutation. This
separation is necessary because default backward can leave work fields160/256
empty and records released Saved roles at offset328. The old implementation does
not understand those invariants. A qualification-only checked HIA controller
links both complete implementations and moves the unique owner through their
actual step and close exports; it does not serialize or synthesize a pointer.

The implementation separates these lifetimes:

1. Graph topology, shape/stride metadata and leaf accumulation references remain
   while public graph roots exist. Cat/stack parent lists remain topology.
2. Actual AD Saved Storage/Version obligations are released after each successful
   node application when flags0. Public Saved objects are independent and remain.
3. Private index payloads are released. Index-select/gather always need their
   saved index; scatter/index-add need it only for the source gradient. A pure
   self-gradient route can repeat after its unused index allocation is freed.
4. Temporary work allocations are released after the full successful pass. Every
   backward entry performs full Saved preflight, then ensures work storage before
   zeroing/using it. Nonleaf persistent gradient buffers, which the public API
   cannot read, are released; leaf accumulated gradients remain.

Node offset328 records a released Saved role. A reachable released role fails
status10 before a version check; it cannot be resurrected by a later flags1.
Nodes with only metadata do not acquire this flag, so add/view/clone/transpose
graphs can run again where the actual derivative needs no released Tensor.
The no-save path reallocates its temporary work as necessary.

The outer native owner transaction remains stronger than PyTorch's failure
side-effect contract: a failed application frame rolls back all gradients,
releases, versions and preceding commands. Upstream can have released Saved data
on already-executed nodes before another node raises. This difference is explicit
and is not claimed to be native/PyTorch failure-side-effect parity.

The existing v3 matrix-copy save policy, scalar arithmetic and numerical
assertions are preserved. The pinned upstream Engine and generated Node release
logic and independent true-Torch boundary probe are audited separately by the
scope reviewer. No local Torch execution is claimed.
