# Direct differentiable view R2: retained hook on lazy refresh

This is a new production source revision. R1 and its two original native
attempts are immutable. R1/R3 completed 13 passing recipes and one failing
recipe; it is not runtime-qualified.

## Actual failure and causal source

The fixed 14-recipe Torch observation is
2270d5ba925bcf91c56db94fac663695e69079c74edbf80aff8fa086e7c81665.
R1/R3 aggregate 2b4256989dfad08b49b0b5f48a253bf564687b878bc46372a55a37d4d054a82b
shows case 11's old captured chain writing current view gradients after
Version refresh, and the new current chain omitting its retained gradient.
Its final public view drop then reaches a real audit status 9. All original
requests/responses and comparison failures remain preserved.

Pinned PyTorch 31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457,
torch/csrc/autograd/variable.cpp, original Git blob
63e2ef57c05b19fd17479fe67abd36e49fc598e6, SHA256
ca2f1e573307069dee2fdff70ba81f161f48817cd278ad5464b9809cf23ab2c6:
lines 177-207 move a retained-gradient hook from the old grad_fn to the
new grad_fn. Lines 755-758 do this after lazy ViewInfo Version refresh.
The retained hook uses a weak Tensor reference. The official whole file is
included as upstream-variable.cpp; no claim is made from a source URL alone.

## Production change

The direct constructor remains exactly R1. In to_view_refresh only, the
successful history_capture branch now validates and removes the weak hook
route in the old nonleaf history snapshot and re-enables the current
Tensor's retained hook. The old snapshot's parents, captured geometry,
Saved payloads and Version semantics remain native state.

An old op0 captured AccumulateGrad bypasses this change. A nonleaf snapshot
with no weak hook must have both hook fields zero and leaves the current
disabled state untouched. A moved hook must match source id plus mode1 and
source incarnation exactly. Current hook state must equal preserved low
accumulator bits plus mode2. Only then are old608/624 cleared and current608
reduced to those same low bits. Current624 is preserved.

This is a weak-route move. It neither increments nor decrements 600/208,
and it does not release the strong captured AccumulateGrad. Malformed state
takes the existing terror9 rejection and surrounding frame transaction.
All previous original blocks stay byte-identical except the single success
branch. Eight appended native HIA blocks implement the move and guards.
Exact inverse proof restores the original function byte-for-byte.

General history_capture, every gradient arithmetic function, ViewInfo
640/648 ownership, slot layout and all reader comparisons remain unchanged.
The new isolated profile uses HTVAD002 / HTOVAD02 with 672-byte descriptors.
The observer changes only those two tag literals. The R3 fixed14 reader
52c3b3242a73fb412102c83c5ba47cf8d95a4eb3c01e56f41bfe593066506d05 is unchanged.

## Qualification boundary

Source preflight and inverse checks are not a native build or runtime pass.
Autograd is the unique native compiler executor; scope is the unique
fixed14 native replay executor. All 12 units must be newly compiled and
linked before the unchanged R3 replay is run. No Torch is executed here.
