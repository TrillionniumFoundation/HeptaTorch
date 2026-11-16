# Numerical Tensor / Storage object protocol, version 3

All integers and binary32 words use little-endian encoding. This protocol is a
separate profile from `HTORCH01` and `HTORCH02`. The original arithmetic command
parameters are preserved where stated below; handles, metadata and ownership
rules have changed explicitly.

## Frame and transaction

The native owned-session wrapper receives `HTOWNR01` followed by a u64 body
length and that many bytes. It responds with `HTOWNS01`, a u64 length and the
response body before reading another frame. Input and output bodies are each
bounded by 1 MiB. The existing wrapper supplies exact partial-read/write loops,
EINTR handling, SIGPIPE handling and cleanup; EOF is not required for a reply.
The only persistent native owner starts as a live empty buffer, moves into HIA
`step`, and is replaced only by that function's returned buffer. `close` consumes
the current owner on ordinary EOF and wrapper error cleanup.

The application request starts with 32 bytes:

| Offset | Width | Value |
| ---: | ---: | --- |
| 0 | 8 bytes | Literal `HTORCH03` |
| 8 | u64 | Command count |
| 16 | u64 | Exact body byte length |
| 24 | u64 | Reserved zero |

Exactly that many 64-byte command records follow, then optional payloads. Each
record is eight u64 words: `opcode,a,b,c,d,e,f,g`. Payload offsets are relative
to the body, aligned as required by the operation and no earlier than the end
of the command records. At most 16,383 commands fit; payload and response space
can impose a lower limit. A valid zero-command request can initialize and commit
the empty state. An invalid first frame does not initialize it.

The response header is `HTORSP03`, u64 version 3, completed-count and status.
Each successful record is `opcode,0,result,payload_bytes` followed by its payload.
A failed frame returns only the 32-byte header, with the original error status
and count zero. No trial result or trial handle is published. The original
owner's pointer, length and every byte remain unchanged, including versions,
gradients, graph reset and allocations made earlier in that frame.

Execution uses a separate native trial allocation. Only an allocator-generated
capacity hint can request a larger trial and replay of the original frame.
Ordinary shape, work-budget, output-capacity or contract errors do not trigger
growth. If native allocation fails, the original owner is returned unchanged.
Unpublished guessed IDs are not granted handles: tentative IDs in an aborted
frame are not promised to be permanently reserved. Published and later revoked
handles remain invalid when a table slot is reused.

## Typed handles and native request-local references

A public typed handle is:

```text
(generation << 32) | (kind << 24) | slot
```

Generation is nonzero u32. Slot is a nonzero table index, at most 65,536 in this
profile. Kind 1 is Tensor, 2 Storage, 3 Version, 4 Weak, 5 Saved and 6 StrongHold.
Every use checks the exact kind, generation and live record. Tensor operations
also require its public flag. Internal node numbers, revoked Tensor rows that
remain graph-owned, private AD Saved records and private allocation/layout
Holds are not accepted through public commands. The private-record rejection
also applies to zero-length reads.

An ID field can instead contain `0xFE000000 | n`, where `n` is the one-based
ordinal of an earlier successful command **in the same frame**. The runtime
resolves that record's actual result, then performs the required typed check.
The marker has generation 0 / kind 254. Zero, current/future ordinals and references
carried into a later frame fail status 1. Responses always contain actual typed
IDs, never the marker. This mechanism also works inside cat/stack Tensor lists
and in optimizer parameter/state fields. Dimensions, scalar bits, indices and
other non-ID fields are not interpreted as references. In particular command 14
is entirely a scalar-bit/integer operation.

This native resolution lets a producer chain commands without predicting
internal temporary tensors, free-list choices or typed table reuse. For example,
create `x` in record 1, multiply references to record 1 in record 2, sum reference 2
in record 3, then backward reference 3. IDs from a completed frame should be read
from its response and used directly in subsequent frames.

## Arithmetic and inspection commands

Commands 1–37 retain the argument order and supported operation variants in
`docs/protocol.md`, with these explicit changes:

- Tensor ID fields use typed handles or the references above.
- Command 12 is an explicitly untracked scalar element write. It can address
  a leaf or nonleaf and advances their shared Version exactly once on success.
  It is not a full autograd graph-rewriting inplace operation.
- Command 13 returns the 128-byte metadata layout below.
- Command 18 clone preserves the stride layout of dense non-overlapping inputs,
  including the qualified empty transposed layouts, and creates new S/V IDs.
- Command 33 uses the graph/retention rule below instead of encoding generation
  directly into a descriptor ordinal.
- Frame errors roll back all earlier commands in the frame.

| Code | Operation | Fields after opcode | Result |
| ---: | --- | --- | --- |
| 38 | Construct a filled Tensor | rank, requires_grad, dim0, dim1, dim2, dim3, binary32_bits | Tensor ID |
| 39 | Read logical values | Tensor ID, start_element, element_count | Count and binary32 payload |
| 40 | Read leaf gradients | Tensor ID, start_element, element_count | Count and binary32 payload |

Full construction accepts zero-sized shapes and requires the fill field to fit
u32. It constructs a native Tensor larger than the wire frame without an input
pointer. Logical ranges use row-major logical element order, including strided
Tensors. A zero-length range can start at numel. The gradient read still checks
leaf/gradient eligibility when count is zero. Out-of-range is status 2, invalid
gradient eligibility status 5, and insufficient response space status 3. Commands
10/11 still read the whole Tensor/gradient when it fits.

Command 13 returns the actual Tensor ID and sixteen u64 words:

| Word | Meaning |
| ---: | --- |
| 0 | Rank |
| 1 | Numel |
| 2 | Requires-grad flag |
| 3–6 | Four dimensions; unused dimensions retain the ordinary descriptor values |
| 7–10 | Four element strides |
| 11 | Storage ID |
| 12 | Version ID |
| 13 | Absolute mutation counter |
| 14 | Tensor's byte offset within Storage |
| 15 | Storage payload byte size |

Storage identity and Version identity must not be inferred from each other.
New construction and clone start at counter 0. View and detach share the original
Version. Explicit writes and SGD each advance it once. The qualified Adam
correction advances parameter and first-moment once and second-moment twice;
AdamW with nonzero weight decay additionally advances the parameter once.
Zero learning rate and empty Tensors still reflect the requested logical
mutators. Optimizer state buffers are explicit user-created leaves; their
creation lifecycle is not an implicit `torch.optim` state object.

## Lifetime commands

| Code | Operation | Fields after opcode | Result / payload |
| ---: | --- | --- | --- |
| 41 | Release public Tensor | Tensor ID | 1 |
| 42 | Create weak Storage reference | Tensor ID | Weak ID |
| 43 | Lock weak Storage reference | Weak ID | Public StrongHold ID; expired payload fails6 |
| 44 | Release Weak | Weak ID | 1 |
| 45 | Release public StrongHold | Hold ID | 1 |
| 46 | Save Tensor data/version/layout | Tensor ID | Public Saved ID |
| 47 | Check public Saved version | Saved ID | 1; changed version fails6 |
| 48 | Release public Saved | Saved ID | 1 |
| 49 | Read public Saved logically | Saved ID, start_element, element_count | Count and binary32 payload |
| 50 | Read StrongHold storage order | Hold ID, start_element, element_count | Count and binary32 payload |
| 51 | Inspect Storage record | Storage ID | Same ID and 64-byte metadata |
| 52 | Inspect Version record | Version ID | Same ID and 64-byte metadata |
| 53 | Audit object obligations/extents | No fields | 1 on success |
| 54 | Inspect allocation/object statistics | No fields | 1 and 96-byte statistics |
| 55 | Create data alias | Tensor ID | Non-gradient Tensor ID; same Storage, new Version 0 |
| 56 | Unpack public Saved | Saved ID | Non-gradient Tensor ID; shared S/V and preserved layout |

Release is explicit and not idempotent. A repeated release or stale/wrong-kind
handle fails 1. A Tensor's public release immediately prevents all subsequent
public uses of that handle. Graph obligations can retain a private leaf row or
saved data afterward, but cannot restore that public handle.

Weak references retain expired Storage metadata, not its payload. Lock succeeds
only while at least one strong payload obligation remains. A StrongHold retains
the whole Storage; its range read is in physical storage order. A public Saved
retains S/V plus an independent logical layout. Its range reads use that layout
and check the recorded Version even for count 0. Saved and Hold ranges use
binary32 elements, not bytes. Neither command exposes a host address.

Storage metadata words are:
`generation,live,extent,payload_bytes,aligned_allocation_bytes,strong,weak,payload_dead`.
Extent is pool-relative, not a process pointer. A weak-retained expired record
can remain inspectable with `payload_dead=1` after the extent has been returned
to the allocator. It cannot be locked to a new allocation that reuses that hole.

Version metadata words are `generation,live,counter,refs,0,0,0,0`.
These counts describe explicit native protocol obligations, not PyTorch's
internal intrusive-pointer counts. Public IDs themselves do not add references;
the corresponding live Tensor/Saved/Hold/Weak records do.

Statistics words are:
`slots_per_kind,pool_bytes,resize_epoch,owner_bytes,free_bytes,live_T,live_S,live_V,live_Weak,live_Saved,live_Hold,node_sequence`.
Audit is intentionally bounded to at most 4,096 slots per kind; larger tables
return status 3 instead of running a quadratic audit. Ordinary numerical commands
remain available subject to their own limits.

## Graph obligations, Saved data and reset

Reverse-mode nodes retain their routing/shape metadata separately from the
public Tensor and its payload. A required gradient leaf has an AccumulateGrad-like
Tensor obligation. Other input data is retained only through the derivative's
actual Saved requirements. Nonleaf payload can therefore expire while a later
child still has a valid route to the leaf gradient. Backward validates all
reachable Saved versions before clearing or writing any gradients.

Private AD Saved records are distinct from public Saved records and cannot be
released or read through commands 47–49, even if their integer IDs are guessed.
Similarly, descriptor-page, gradient/work, operator auxiliary and Saved-layout
Holds are private and cannot be released/read by commands 45/50.

For supported operations, mutation behavior follows the qualified pinned CPU
derivative paths. Add/clone/transpose need metadata but no saved Tensor value;
mul/div retain the inputs required by their gradient masks. ReLU and nonlinear
output-based derivatives save output. Gather and pooling save input; index-add
saves source when its gradient is requested. Dense matmul preserves the
view-versus-materialized-copy distinction of the implemented rank 1–4 paths.
The saved copy has separate Storage/Version and contains the original forward
values; simply bypassing a mutation check would not implement this behavior.

Command 33 retains an initial **internal numerical-node prefix** of length `keep`.
The first successful reset fixes that count; later resets must use the same
count. Every kept node must still be a public owned leaf. Zero is permitted.
The operation revokes public Tensors outside the prefix, releases their graph
metadata and private Saved/auxiliary obligations, and reclaims descriptor pages
that no longer contain a kept node. Independent public Saved, Weak and Hold
objects remain live according to their own references. No bump-pointer rewind
can overwrite their surviving allocations.

Published handles revoked by reset remain stale after slot reuse because the
typed table generation changes. Kept parameter/optimizer identities and versions
survive. A reset followed by a later error in the same frame is entirely rolled
back. This protocol retains the backward graph until explicit release/reset;
it does not claim PyTorch-default automatic saved-data release after backward.

## Finite limits and errors

| Resource | Bound |
| --- | ---: |
| Rank | 0–4 |
| Per-dimension extent | 262,144 |
| Nonempty numel | 16,777,216 |
| One binary32 Tensor payload | 64 MiB |
| Slots per record kind | Powers of two, 64–65,536 |
| Data pool | Powers of two, 128 KiB–1 GiB |
| Numerical-node sequence | 65,535 until reset |
| Descriptor record/page | 384 / 98,304 bytes |
| Input/response body | 1 MiB each |
| Max command records without payload | 16,383 |
| Existing kernel work budget | 16,777,216 terms where applicable |

All dimensions are validated before zero-aware numel multiplication. Empty
shapes with large earlier dimensions remain empty. The per-dimension bound keeps
rank 4 empty contiguous strides within 2^54. Empty contiguous strides use
`max(dimension,1)`; dense clone can preserve a different valid empty stride.

Total owner size is `139264 + 144 + 424*slots_per_kind + pool_bytes`; it includes
the private numerical prefix, record tables and data pool. OS allocation can
fail within the stated bounds. Transactions can need original and trial state
simultaneously. Numerical nodes, Tensor rows, Saved rows and private Holds have
independent capacity costs; these maxima are not a promise that every maximum
can be reached simultaneously.

| Status | Meaning |
| ---: | --- |
| 0 | Success |
| 1 | Invalid, revoked, private or wrong-kind handle/reference |
| 2 | Shape, rank, axis, view or range mismatch |
| 3 | Allocation, capacity, output space or operation/audit work bound |
| 4 | Unsupported operation |
| 5 | Gradient or optimizer-state contract mismatch |
| 6 | Stale Saved version, expired Weak lock, or invalid reset retention |
| 7 | Invalid argument or incompatible owned-state format |
| 8 | Checked reference/version/reset-counter exhaustion |
| 9 | Internal object/extent obligation inconsistency |
| 100 | Malformed application envelope, payload or offset |

The version counter is checked u64, and object generations retire at u32
exhaustion rather than wrapping. This does not implement every upstream atomic
counter/inference rule. The service does not expose foreign pointers, generic
Storage resize, all view forms, arbitrary dtype/device dispatch, higher-order
AD or concurrent ownership. See the package README for qualification and the
remaining full-rewrite scope.
