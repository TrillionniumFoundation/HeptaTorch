# Public gradient object protocol and native layout

## Frames and identities

The outer input and output frames are `HTOWNR01` and `HTOWNS01`, each followed by a little-endian u64 body size. Bodies retain the existing `HTORCH03` / `HTORSP03` contract. The request has a 32-byte header and eight-u64 command records. Ordinary failed frames return a 32-byte error body with no tentative object IDs, and the original owner is unchanged. Command 58 retains its separately defined structured runtime-exception commit behavior.

A public Tensor ID is `(generation32 << 32) | (kind8 << 24) | slot24`, with kind 1. Public closure invalidates that token. A private reference may keep the same real Tensor alive; a later getter or Optimizer publication then creates a fresh public generation. The Tensor's incarnation at descriptor offset 320, Storage ID, Version ID and layout remain stable. An internal descriptor ordinal is never accepted as a public Tensor ID.

Request-local IDs `0xFE000000 | prior_record_ordinal` can refer to a real earlier Tensor result. Current, future, zero, stale and wrong-kind references are rejected. A setter's scalar success value or a getter's None result cannot be used as a Tensor capability. Runtime payloads contain no owner pointers.

## Commands

All words are u64. `a` is record word 1 and `b` is word 2. Fields not used below must be zero.

| Opcode | Arguments | Successful record value | Payload |
| --- | --- | --- | --- |
| 91 | `a = public Tensor` | Existing gradient Tensor's public token, or 0 for None | None |
| 92 | `a = public Tensor`, `b = public gradient Tensor or literal 0` | 1 | None |
| 93, 94 | Reserved | Rejected with status 4 when otherwise well formed | None |

The getter publishes the existing private gradient Tensor and does not copy its values or create a new Version. A defined empty gradient is a real Tensor and returns a nonzero token. Repeated None reads do not allocate a gradient object.

The setter retains the supplied Tensor as the exact gradient object. Shape must match exactly, including scalar versus size-one and empty shapes. Setting the same existing gradient is a no-op. Assigning the Tensor to its own gradient is rejected. Literal zero clears the binding. An independently retained gradient or alias survives clearing and can be read or mutated through its own public token.

Commands 59, 11 and 40 remain available. Command 59 reports whether the binding exists. Legacy gradient reads 11/40 return zeros for None after their normal bounds checks; they do not change None into a defined gradient. Getter 91 is the public object interface.

Command 70 and Optimizer 73 with a true flag remove a leaf's binding. Command 8 and Optimizer 73 with a false flag zero a defined gradient through its logical layout, preserve the same Tensor and Storage, and advance the real gradient Version once, including an empty gradient. A None gradient stays None. Qualified in-place detach behavior is performed before zeroing. Nonleaf views and unsupported downstream autograd identity splitting are refused rather than represented as copied temporary gradients.

The existing SGD object reads `tgrad`, which now follows the real binding and layout. Its first momentum layout is chosen from the actual gradient. It retains the original private parameter roots, lazy state semantics and captured AccumulateGrad edge behavior. Adam's separate owned-state allocation must preserve parameter layout; that branch is not merged by this package.

## Refusals

| Status | Meaning relevant to this component |
| --- | --- |
| 1 | Wrong/stale typed Tensor, invalid request-local reference, or a closed public token |
| 2 | Gradient shape mismatch or direct self-assignment |
| 3 | Admitted capacity/allocation/output bound cannot be satisfied |
| 4 | Opcode 93/94 remains reserved/unimplemented |
| 5 | Unsupported leaf/nonleaf or in-place detach condition |
| 6 | Saved Version mismatch, as in the inherited Saved interface |
| 7 | Nonzero reserved words, refused ownership cycle, or incompatible internal owner format |
| 8 | Generation, incarnation, Version or reference-count overflow |
| 9 | Internal consistency invariant failure |
| 10 | Backward attempts to read released Saved data |
| 100 | Malformed inherited request framing/payload |

Ordinary errors preserve the complete committed owner. A command-58 runtime error remains a normal frame containing the documented error result, stops subsequent commands, commits previously executed backward work, and drains pending object release. Do not reinterpret it as an atomic rollback of the whole backward computation.

## Persistent descriptor layout

Descriptors are 576 bytes, in the existing 256-entry offset-addressed pages. Page size is 147,456 bytes. The one real owned buffer holds the page directory, object tables, extents and data. Growth copies relative offsets; it does not manufacture borrowed capabilities from stored raw pointer words.

The following preexisting fields remain unchanged:

| Offset | Role |
| --- | --- |
| 88 | Captured AccumulateGrad strong-edge count |
| 208 | Total node metadata/private references, audited against each role |
| 312 | Private Optimizer Tensor-root count |
| 320 | Stable Tensor incarnation |
| 360 | Gradient binding presence |

The public-gradient persistent fields are:

| Offset | Role |
| --- | --- |
| 384 | Internal gradient Tensor node, or 0 |
| 392 | Expected gradient Tensor incarnation |
| 400 | Incoming strong gradient-binding count on this Tensor |
| 408 | Differentiable-view marker used by in-place detach admission |
| 416 | Temporary cycle/audit mark; zero at commit |
| 424, 432 | Temporary iterative GC link/queued flag; zero at commit |
| 440 | Temporary cycle traversal link; zero at commit |

Old private gradient-cache fields 24 and 248 are zero at commit. A binding of None has offsets 360/384/392 all zero. A defined binding names a live real Tensor of the recorded incarnation, with matching shape and a positive audited incoming binding count. Public getter publication does not change the incarnation.

A constructor may retain the inherited private work cache at 160/256 before its first backward. Those fields must be a valid owned pointer/Hold pair. This is separate from the gradient binding. Nodes that execute backward release their temporary work caches, including retained-graph calls; the qualified tests distinguish those two cases.

## Temporary incoming values

These fields exist only during native backward and are all zero at commit:

| Offset | Role |
| --- | --- |
| 448, 456 | Incoming Storage and Version IDs |
| 464 | Storage byte offset |
| 472 | Copy-required or independent eligible incoming identity role |
| 480 | Preallocated cache Version owner |
| 488 | Incoming Tensor-value identity origin node |
| 496–520 | Four incoming element strides |
| 528 | Transient references to that incoming identity |
| 536 | Physical logical-cache Storage ID |
| 544–568 | Reserved zero words |

These counters model this implementation's ownership obligations. They do not claim to equal PyTorch's numerical intrusive-pointer use counts. A view can have a distinct incoming Tensor identity while sharing Storage and Version. A held seed, shared input buffer or external alias constrains which mutations and ownership transfers are allowed.

Global words 3992/4000/4008 temporarily identify the current edge staging table, count and selected concat edge; they are zero at commit. The existing GC queue words 3968/3976 and cycle traversal word 3984 are also zero at commit. The short description area 3808–3895 is cleared after each apply.

Every original parent edge receives a separate value buffer and first-write bitmap. Direct derivative writes preserve their first binary32 word exactly, including signed zero and the qualified payload-preserving controls. Real scatter/reduction paths start with positive zero and perform their actual additions. Completed edge values enter the parent's InputBuffer in edge order. Values are copied on first receipt, added on subsequent receipt, and take real in-place or out-of-place Storage/Version paths according to the modeled ownership and layout conditions.

Before partial execution, each reachable node has its required real cache Storage and reserved Version. The first gradient Tensor placeholder is also admitted before side effects. A qualifying first accumulation adopts the incoming Storage/Version by retaining them before releasing the placeholder's Storage/Version. A copy writes the actual new Storage; existing gradient accumulation mutates the actual assigned object. There is no counter reset to force reference equality.

On success or a structured runtime exception, temporary incoming references, cache Holds, reserved Versions and unused gradient placeholders are released. Only actual committed gradient bindings and independently retained aliases/Saved/Optimizer objects survive. If admission fails, the owner transaction discards the trial state.

## Format isolation and scope

Internal owner tags are `HTGRAD02` and `HTOGRAD2`. They are incompatible with `HTOPT002` / `HTOOPT02` and the intermediate `HTGRAD01` / `HTOGRAD1`. The owner bridge is still the checked `step(readbuf, mutbuf, buffer) -> (buffer, u32)` / `close(buffer)` interface. The wrong step returns the original real owner unchanged. Close consumes that owner normally.

This profile inherits rank 0–4, CPU binary32, maximum dimension 262,144, maximum elements 16,777,216, descriptor ordinals 1–65,535, packet/response size at most 1 MiB, and the existing allocator and work-budget caps. It does not implement Python hooks, create_graph, arbitrary retained nonleaf gradients, general ownership cycles, sparse/complex/device/dtype setter behavior or every possible overlap/layout path. Commands 93/94 remain reserved. The same-owner Adam/AdamW plus distribution composition remains separately unqualified here.
