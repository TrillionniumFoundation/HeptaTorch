# Dynamic owned tensor protocol, version 2

This is a separate native HIA execution profile. The original `HTORCH01`
single-request runtime, borrowed persistent sessions, and their fixtures keep
their previous identities and semantics. The profile here reuses the canonical
143-function numerical source through a checked overlay; it is not a second
manually maintained tensor implementation.

## Interactive envelope and transaction

The already qualified owned-session wrapper accepts a 16-byte outer header:
`HTOWNR01` and a little-endian u64 body length. It replies with `HTOWNS01`, a
u64 length, and the response body before reading the next frame. Each input or
output body is at most 1 MiB. The native process retains the one HIA-owned state
buffer between frames. Owners originate from trusted live-empty startup or
checked HIA returns; user bytes never provide a native owner descriptor.

The v2 request body has four u64 words, followed by 64-byte command records and
payload:

| Offset | Value |
| ---: | --- |
| 0 | Eight literal bytes `HTORCH02` |
| 8 | Number of commands |
| 16 | Exact body length, including records and payload |
| 24 | Reserved, zero |

Command fields remain `opcode,a,b,c,d,e,f,g`. Payload offsets are absolute body
offsets, with the alignment and command-tail checks of the original protocol.

The response starts with `HTORSP02`, version 2, completed response count, and
final status. On success each record is `opcode,0,result,payload_bytes`, followed
by its payload. A failed frame returns exactly the 32-byte header with its
original status and count zero. It publishes no successful trial results or
IDs. Every failed frame returns the exact original owner with all of its bytes
unchanged, including descriptor allocation, gradients, optimizer states,
versions, and reset generation.

The HIA `step` keeps the original owner while it executes against a separately
allocated trial. Only an allocator-originated internal capacity hint permits a
larger trial and replay of the entire original frame. Ordinary shape, work
budget, output capacity, and handle refusals do not cause blind growth. Native
allocation failure aborts the frame and returns the original owner. A successful
frame drops the original owner and returns the trial. The syscall wrapper does
not copy owner descriptors to implement rollback.

Unpublished guessed tensor numbers are not granted handles. This protocol does
not claim that a number tentatively allocated inside an aborted frame can never
be used later. Successfully published handles discarded by reset remain stale
even after their slots are reused.

## Commands

Commands 1 through 37 retain the parameter order, binary32 arithmetic, strided
tensor behavior, reverse-mode differentiation, and supported operation variants
specified in the repository's main `docs/protocol.md`, with the identity,
construction-version, capacity, and transaction changes stated here.

| Code | Operation | Fields after opcode | Result |
| ---: | --- | --- | --- |
| 38 | Construct a filled tensor | rank, requires_grad, dim0, dim1, dim2, dim3, binary32_bits | Tensor ID |
| 39 | Read a logical value range | Tensor ID, start_element, element_count | Count; binary32 payload |
| 40 | Read a leaf-gradient range | Tensor ID, start_element, element_count | Count; binary32 payload |

Code 38 creates an ordinary owned leaf, including zero-sized shapes. Its fill
field must fit in 32 bits. It permits tensors larger than the wire frame without
importing a pointer or delegating their construction to a host numerical runtime.

Range reads use logical row-major order, including transposed/channels-last
inputs. The range must fit the tensor; a zero-length range may begin at its end.
Code 40 requires a gradient-tracked leaf even when the range is empty. Out of
range returns status 2; a noneligible gradient read returns status 5. A payload
that does not fit the remaining output frame returns status 3 and rolls back
the entire frame. The original whole-tensor reads remain available.

## Capacity and shape bounds

| Resource | Qualified bound |
| --- | ---: |
| Rank | 0 through 4 |
| Each dimension | 0 through 262,144 |
| Nonempty tensor elements | 16,777,216 |
| One tensor's binary32 data | At most 64 MiB |
| Owned state capacity | Powers of two from 64 KiB through 1 GiB |
| Live tensor descriptors | 65,535 |
| Descriptor pages | 256 pages of 256 slots; slot zero is reserved |
| Slot bits / reset generation bits | 16 / 48 |
| Existing operation work budget | 16,777,216 where the operation applies it |
| Request / response body | 1 MiB each |

These are finite profile bounds, not claims of unrestricted memory or complete
PyTorch coverage. OS allocation availability can refuse a request within them.
Transactions can temporarily require both the original and trial allocations.

Every dimension is validated before a zero-aware product check. A zero dimension
makes numel zero even when preceding dimensions have a large product. The
ordinary contiguous stride is constructed with `max(dimension,1)`; retaining
the original dimension bound keeps rank-four empty strides within `2^54`.
This overlay applies that two-pass product check to matmul and gather as well
as the already qualified descriptor/concatenation/indexed-addition checks.
Matmul/convolution/indexed-addition work limits remain separate from storage
capacity. Increased owned memory does not authorize excess operation work.

## Handles, aliases and graph reset

Before the first reset, handles are sequential slot numbers starting at 1.
Afterward a transient tensor uses `(generation << 16) | slot`. Retained prefix
handles remain their original raw slot numbers. The old `generation << 8`
encoding is not accepted as an interchangeable v2 encoding.

The first successful reset fixes `keep`; all later resets require the same
value. Each kept tensor must be an owned leaf. Reset retains values, gradients,
work buffers, optimizer tensors, and versions. It reclaims all later graph
descriptors and allocations, advances generation, and rejects old-generation
handles on subsequent frames. Generation is bounded by `2^48-1`; exhaustion
refuses before reclamation. `keep=65535` is legal after allocating all slots,
but leaves no slot for a new tensor.

Descriptor pages are appended dynamically within the same owned byte buffer.
The private page directory and all data/gradient/work/auxiliary references use
offsets, not physical addresses. Growing the actual native allocation therefore
preserves logical handles and stored offsets without rewriting every tensor.
Reset retains the full extent of each page covering a kept descriptor, including
pages whose tensors have zero elements. It clears discarded slots and higher
directory entries before reclaiming the allocation tail.

Views and detach still share the source tensor's storage owner and version.
This stage retains that original owner-tensor model. Paging alone does not
connect the numerical runtime to the separate Storage/Version/weak/saved-object
service and does not implement independent object reclamation or all Storage
APIs. That connection remains separate work.

## Construction versions and outstanding parity work

Payload import, filled-tensor construction, and internal constant construction
now write initialization bytes without incrementing the mutation counter.
Their initial version is zero. Clone and newly owned numerical results also
start at zero. A successful explicit leaf/alias write and SGD update advance the
shared owner version once. Backward's saved-version checks remain active across
physical owner growth, and a failure rolls back the entire frame.

The original Adam/AdamW/momentum numerical functions and their per-command
version increments are retained in this frozen candidate. Complete absolute
optimizer parameter/state counter equivalence is not claimed. A separate
pinned-PyTorch oracle must record and strictly compare those counts; any mismatch
is a failure requiring an explicit native metadata correction, not a relaxed
expectation. Existing numerical tolerances are not changed to hide metadata
differences. The original candidate and its receipts remain identifiable when
such a correction produces a later profile.

## Typed state admission

The private state begins with a format tag and checked counters, capacity, and
page directory. A nonempty typed buffer with a wrong format or invalid header is
refused before clone/execute; this matters because a generic HIA `buffer` type
does not itself encode this service's state schema. The public wire never
accepts a serialized owner buffer. Header validation is not advertised as a
full verifier for arbitrary attacker-authored native modules.
