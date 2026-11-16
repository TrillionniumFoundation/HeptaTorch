# Owned Adam / AdamW profile — native implementation contract

This successor reads the exact frozen Optimizer08 numerical source
`7c68a96b08aa9a21a52e2fd12be912c5f47dbde889e462f9a0dbb38cb3c09edf`.
It leaves that package and its SGD evidence unchanged. External frames remain
`HTOWNR01` / `HTOWNS01`, and inner requests/replies remain `HTORCH03` /
`HTORSP03`. The new nonempty numeric/backing state tags are `HTADAM01` and
`HTOADAM1`; legitimate states from the previous profile are not interchangeable.

The implemented modes are nondifferentiable, noncapturable, nonfused,
non-foreach dense CPU binary32 Adam (`algorithm=1`) and AdamW (`algorithm=2`).
Algorithm 0 retains the exact SGD wire layout and functional implementation.
AMSGrad, sparse/complex/other-dtype state and optimizer hooks are not provided
by this component. Runtime qualification is recorded separately from this
interface description; compilation alone is not a numerical or lifecycle pass.

## Commands and configuration

Every command has eight little-endian u64 words `[opcode,a,b,c,d,e,f,g]`.
Unused words must be zero. Float words carry binary32 bits with zero high
32 bits. Payload offsets are aligned absolute byte positions beyond the
command table. Existing checked same-frame result references can appear in
membership lists and command handle fields.

| Command | Adam/AdamW fields | Result and payload |
| --- | --- | --- |
| 71 | `algorithm(1/2), defaultsOffset, groupsOffset,0,0,0,0` | Kind-8 Optimizer; four-word inventory. No state Tensor is created. |
| 72 | `Optimizer,0,0,0,0,0,0` | Scalar 1; ordered groups and occurrences, lazy state, then updates. |
| 73 | `Optimizer,set_to_none(0/1),0,0,0,0,0` | Scalar 1; True is the frontend default. None remains None under either choice. |
| 74 | `Optimizer,groupOrdinal,0,0,0,0,0` | Occurrence count; Adam group payload below. |
| 75 | `Optimizer,groupOrdinal,occurrenceOrdinal,0,0,0,0` | Public token for the same retained parameter Tensor. |
| 76 | `Optimizer,groupOrdinal,occurrenceOrdinal,0,0,0,0` | Scalar 0/1 for absent/present state; eight-word inventory below. Read-only and does not insert state. |
| 77 | `Optimizer,groupOrdinal,occurrenceOrdinal,field(1/2/3),0,0,0` | Actual step/exp_avg/exp_avg_sq Tensor, respectively, or zero when absent. |
| 78 | `Optimizer,0,0,0,0,0,0` | Scalar 1; drops membership/state-key/state-Tensor obligations. |
| 79 | `Optimizer,oneGroupOffset,0,0,0,0,0` | Scalar 1 plus updated four-word inventory. |
| 95 | `publicTensor,scalarF32Bits,0,0,0,0,0` | The same public Tensor; explicit no-grad scalar in-place addition. |

The creation/add inventory remains `[groups,occurrences,uniqueParameters,
duplicateOccurrences]`. It counts opaque Tensor identities, never Storage
identities. The Tensor/Storage/Version/Weak/Saved/Hold/Generator/Optimizer
tables retain kinds 1 through 8 and their frozen relative-offset layout.

Adam defaults are exactly eight words:

`[lr,beta1,beta2,eps,weight_decay,amsgrad,maximize,reserved0]`.

The first five are float bits; `maximize` is Boolean; `amsgrad` must currently
be zero; the final word must be zero. Default-constructor checks use the
pinned Adam predicates: `0 <= lr`, `0 <= eps`, `0 <= weight_decay`, and
`0 <= beta < 1`. NaNs fail these default predicates; nonnegative infinity
passes the first three but not either beta check. Per-group override admission
does not reapply constructor value predicates; width, flag and reserved-word
checks always apply.

The groups payload begins with `groupCount`, followed by concatenated groups.
Each group is `[parameterCount,overrideMask,8ConfigWords]`, followed by its
`parameterCount` public Tensor handles. Mask bits 0 through 6 select overrides;
unselected words and the reserved eighth word are zero. Command 79 contains
one group without a group-count prefix. Defaults are merged in native HIA.

Command 74 emits `[groupOrdinal,parameterCount,8ConfigWords]`, followed by
`[parameterIncarnation,currentPublicTokenOr0]` for every occurrence. An
incarnation is an opaque monotonic native Tensor identity, not an address.

Command 76 emits exactly eight words:

`[parameterIncarnation,fieldMask,stepIncarnation,stepPublicTokenOr0,
expAvgIncarnation,expAvgPublicTokenOr0,expAvgSqIncarnation,expAvgSqPublicTokenOr0]`.

The mask is zero when state is absent, with all three state pairs zero. It is
14 when the three real Tensor fields exist: bit 1 step, bit 2 exp_avg, bit 3
exp_avg_sq. The record result is Boolean 0/1, not the mask. Publishing state
does not clone it, reset its Version, or change its stable incarnation.

SGD still uses six-word configuration, eight-word group headers, four-word
state inventory and field 0 momentum. A format is selected by the actual
Optimizer algorithm row, not guessed from payload length.

Wrong/stale typed handles are status 1. Nonleaf parameter membership is 5.
Invalid config words, reserved fields, ordinals, field selectors, cross-group
duplicates or top-level group count are 7. Invalid payload offsets are 100.
Allocation or aggregate occurrence capacity failure is 3. Version exhaustion
is 8; inconsistent private state is 9. The bounds remain 1,024 groups and
4,096 occurrences per Optimizer, with the existing request, table, descriptor
and allocation bounds. An explicit empty group is valid. Duplicate occurrences
within one group remain ordered; duplicate membership across groups fails.

## State and update semantics

Only parameters with a defined gradient participate, including defined zero
and empty gradients. Current `requires_grad=False` does not suppress an
already defined gradient. None skips parameter updates, all Versions, state
construction and the step Tensor. Each group's participating entries are
initialized before that group's numeric loop. A repeated parameter creates
exactly one three-Tensor state, shared by every occurrence during the loop.

The actual CPU binary32 scalar step Tensor is initialized to zero. Its value
is read, incremented with native binary32 arithmetic and stored on each
participating occurrence. Its Version increments once even if rounding leaves
the value unchanged. There is no substitute integer counter. The two moments
are actual zero-initialized Tensors with new Storage and Version; eligible
dense/empty parameter layout is preserved for their physical storage.

Each occurrence mutates exp_avg once and exp_avg_sq twice. The parameter
increments once in Adam. AdamW adds one increment when weight decay is nonzero,
including empty/zero-learning-rate operations. Those increments use the
already qualified native helper. State aliases therefore observe live data
and the real Version progression; Saved-variable checks see later changes.

The scalar numerical body is the frozen `adam_element` with a new function
name and exactly four access changes: gradient reads go through
`owned_optimizer_gradient_value -> tgrad`; the gradient helper applies
maximize before the frozen coupled-decay helper; both moment addresses use
logical `taddr` rather than contiguous assumptions. The exact inverse edit
restores the old function bytes. Native lerp, fused multiply-add safeguards,
second-moment ordering, bias corrections, division and parameter arithmetic
are unchanged. This retains the existing qualified binary32 numerical
profile, rather than claiming every CPU ISA implementation is bit-identical.

The current bias-correction helper admits positive integral step values below
2^32 after the actual float32 increment. It derives the exponent from the
actual step Tensor bits; it does not truncate fractional values. Normal
float32 counting, including its 2^24 stagnation, is represented. If the
incremented result is not a positive integer below 2^32, the participating
update causes status 7 and rolls back the frame. An externally assigned zero
or negative zero therefore legitimately resumes at one. Other Tensor values
can still be stored/read/mutated. General real-exponent bias correction is
explicitly unfinished; the four fixed actual object recipes do not resume an
optimizer after their fractional external step mutation.

## Native ownership and internal format

Kind-8 row fields remain: +16 configuration Hold, +24 group count, +32
occurrence count, +40 unique count, +48 duplicate count, +56 algorithm.
SGD's original 64-byte header / 64-byte group / 48-byte occurrence remain
unchanged. Adam's blob has a 96-byte header, then 80-byte group rows, then
80-byte occurrence rows:

| Record | Byte offsets |
| --- | --- |
| Header | 0 groups; 8 occurrences; 16 algorithm; 24 reserved zero; 32 eight defaults |
| Group | 0 eight configuration words; 64 first occurrence; 72 count |
| Occurrence | 0 parameter node; 8 incarnation; 16 first key index; 24 mask; 32/40 step node/incarnation; 48/56 average node/incarnation; 64/72 square node/incarnation |

Each occurrence retains the actual parameter Tensor/gradient binding. A
realized state key adds one parameter obligation, and each state Tensor has
one private strong obligation. Only the first occurrence row owns a key;
other occurrences retain zero state fields and point to it. The audit
recomputes these obligations against existing node +312/+208 and validates
each actual state Tensor and its shape. No owner descriptor or host pointer
is copied into user-visible bytes.

Dropping an Optimizer releases its obligations. Public state/parameter
aliases and Saved objects can independently keep the relevant objects alive.
Graph reset, private-root migration, current/fresh public token publication,
Version identity, captured gradient edges and RNG ownership retain their
frozen contracts. This batch adds no descriptor fields or object kinds.

Normal command failures remain whole-frame atomic, including earlier
successful writes in the same frame. Partial-backward command 58 retains its
existing explicit structured-error commit behavior. This is not a claim of
arbitrary Python optimizer exception-side-effect equivalence.

## Command 95 boundary

This is an explicit no-grad mutation primitive. It adds the scalar to each
logical element using native binary32 arithmetic and strided addressing, then
bumps the shared Version exactly once, including empty tensors. Definite
internal overlap from a size-greater-than-one zero stride is refused before
mutation, matching the pinned overlap check for this concrete representation.
The source custody includes the pinned `MemoryOverlap.cpp` Git blob. The
operation keeps Tensor/Storage/Version identity and does not create or rebase
an autograd node. Complete public `Tensor.add_` autograd/history semantics are
not supplied by this no-grad primitive.

## Remaining scope

This is not full Optimizer or full PyTorch closure. In addition to the stated
mode and exponent bounds, public `.grad` Tensor objects are a separate
parallel successor, as are serialization, hooks, AMP, arbitrary state-map
replacement, differentiable updates and optimizer exception partial effects.
The frozen 24-case reference and its previous 18-qualified + 1-partial +
5-open report remain unchanged. New execution evidence for cases 21–24 is
recorded under a new candidate identity, with the original tolerances,
Versions and identity requirements preserved.
