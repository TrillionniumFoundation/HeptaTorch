# Native owned storage object service — candidate contract

This independent candidate is a byte-storage object service. It is not the
HeptaTorch tensor wire protocol and does not claim complete Tensor/PyTorch
storage support. No user payload contains or determines a native pointer.

## Checked owned exports

- `step(readbuf, mutbuf, buffer) -> (buffer, u32)`, in that exact order.
- `close(buffer) -> ()`.
- `step` declares `effects(alloc,read,write,trap)` (mask 15); `close` declares
  exactly `effects(alloc,trap)` (mask 9) under the frozen ABI2 profile.

The caller starts with one live empty owner `(0,0)`. Each call consumes that
owner; the first returned result is the sole next committed owner. The `u32`
result is the exact response byte count, not the command status. Normal and
command-error responses are at least 32 bytes. If the supplied output capacity
is below 32, `step` returns the unchanged owner and zero bytes. A wrapper must
treat zero or a result exceeding output capacity as a terminal transport error.
EOF calls the checked `close` export once on the current owner. A native trap
terminates the process; no recoverable crash/durability guarantee is made.

`step` executes an entire request transaction on independently owned temporary
state. Any command/validation/allocation error returns the original owner and a
32-byte error response with zero completed command count. It does not publish
provisional IDs or payloads. Successful read-only requests may replace the
physical owner with byte-identical state. Logical IDs contain no native address.

## Request frame body

All integers are little-endian u64. Maximum body length is 1,048,576 bytes.

| Byte offset | Field |
| --- | --- |
| 0 | Exact eight bytes `HTOSOBJ1` |
| 8 | Command count, 0 through 1024 |
| 16 | Initial slot capacity, only when the incoming owner is empty; zero chooses 8 |
| 24 | Initial payload-pool capacity in bytes, only when empty; zero chooses 256 |
| 32 | Command records, exactly 64 bytes each |

The body length must equal `32 + 64 * command_count`. Each record is eight
words: opcode followed by seven arguments. Unused arguments must be zero.
After initialization both initial-capacity fields must be zero.

Slot capacity rounds upward to a power of two, minimum 4 and maximum 65,536.
Pool capacity rounds upward to 16 bytes, minimum 16 and maximum 1,073,741,824.
These are explicit candidate limits, not claims about PyTorch limits.

## Response frame body

| Byte offset | Field |
| --- | --- |
| 0 | Exact eight bytes `HTOSRES1` |
| 8 | Version 1 |
| 16 | Completed command count; zero on any failed transaction |
| 24 | Status |
| 32 | Successful command records and optional payloads |

A successful command record is 64 bytes: opcode, status zero, five result
words, payload byte length. Exactly that many payload bytes follow the record.
The next record starts immediately after the payload. Dump is the only current
command that emits a payload. No failed transaction emits command records.

Statuses are 0 success, 1 invalid/stale/wrong-kind handle, 2 invalid shape/range
or capacity request, 3 allocation/capacity exhaustion, 5 saved-version mismatch,
6 expired weak reference, 7 unknown opcode/reserved field, 8 counter/generation
exhaustion, 9 internal state-invariant refusal, and 100 malformed frame.

## Object identities and commands

Handles encode `(generation << 32) | (kind << 24) | slot`. Slot zero is invalid;
live generations start at one. Kinds are Tensor 1, Storage 2, Version 3, Weak 4,
Saved 5 and StrongHold 6. Reuse advances the slot generation. A generation at
`0xffffffff` is never wrapped; an exhausted free slot is skipped.

| Code | Command | Arguments | Main result |
| --- | --- | --- | --- |
| 1 | Create byte tensor | byte length, fill byte | Tensor, Storage, Version, extent offset, length |
| 2 | Alias/view | Tensor, relative offset, byte length | New Tensor with shared Storage and Version |
| 3 | Drop tensor | Tensor | No new handle |
| 4 | Create weak storage reference | Tensor | Weak handle, Storage |
| 5 | Lock weak reference | Weak | StrongHold, Storage; status 6 after last strong reference |
| 6 | Drop weak reference | Weak | No new handle |
| 7 | Drop strong storage hold | StrongHold | No new handle |
| 8 | Save dependency | Tensor | Saved, Storage, Version, snapshot version |
| 9 | Check saved version | Saved | Snapshot/current version match or status 5 |
| 10 | Drop saved dependency | Saved | No new handle |
| 11 | Write byte | Tensor, relative byte index, byte value | Updated version |
| 12 | Read tensor byte | Tensor, relative byte index | Byte value |
| 13 | Read held storage byte | StrongHold, byte index | Byte value |
| 14 | Read saved dependency byte | Saved, relative byte index | Byte value, after version check |
| 15 | Inspect object metadata | Any live typed handle | Five scalar fields; never a native pointer |
| 16 | Pool statistics | None | Slots, pool capacity, free extent count, free bytes, resize epoch |
| 17 | Reserve capacities | Minimum slots, minimum pool bytes | Actual slots and pool capacity |
| 18 | Dump logical state | None | Entire owned-state byte payload, subject to output capacity |
| 19 | Alias with fresh version identity | Tensor, relative offset, byte length | New Tensor sharing Storage but with a distinct Version |
| 20 | Audit logical state | None | Checked metadata/refcount/extent invariants; explicit bounded work |

Command 15 returns the following five words, in order:

| Kind | Fields |
| --- | --- |
| Tensor | Storage ID, Version ID, logical storage offset, byte length, current version counter |
| Storage | Pool-relative extent offset, logical byte length, strong count, weak count, dead-payload flag |
| Version | Mutation counter, reference count, zero, zero, zero |
| Weak | Storage ID, zero, zero, zero, zero |
| Saved | Storage ID, Version ID, saved counter, logical storage offset, byte length |
| StrongHold | Storage ID, zero, zero, zero, zero |

New Version objects start at counter zero. A successful command 11 increments
the referenced Version exactly once. Creating/filling initial bytes does not
increment it. The maximum counter and resize epoch are `2^64 - 1`; an operation
requiring another increment refuses with status 8 before commit. Strong, weak
and Version reference counters also have checked bounds. There are no public
commands for setting counters, generations or raw state bytes.

Payload extents are aligned to 16 bytes. A zero-length object has a distinct
Storage and Version but no allocated payload extent. Free extents are sorted
and merged when adjacent. A new allocation uses the first sufficiently large
free extent; if none fits, the pool grows subject to its declared maximum.
An exhausted table grows until its declared maximum. No live object ID changes
during table or pool migration. A free slot whose generation is `0xffffffff`
is skipped permanently; new table slots remain available after growth.

`resize_epoch` counts actual logical table/pool capacity increases. It is not
a native address, a persistent allocation ID, a frame sequence number or a
Tensor version. Physical allocation identity is the genuine HIA buffer owner:
even a successful read-only frame can replace its physical mapping with an
independent byte-identical buffer. No raw pointer or borrowed view is exported
to the wire client.

Command 20 performs reference accounting and verifies that live extents plus
free extents cover the pool exactly without overlap. Its pairwise interval
check is limited to at most 4096 slots (`4096^2 = 16,777,216` slot pairs).
It returns status 3 for a larger table. This audit-work limit is separate from
the 65,536-slot allocation limit. Each frame is limited to 1024 commands.

Command 19 demonstrates separation of storage and version identities. It is not
a claim that the complete Python `.data`/`set_data` API has been implemented.
StrongHold is a storage lifetime token; it offers read access and does not carry
a tensor/version identity. Weak reference creation increments an explicit weak
reference; storage also retains one implicit weak count while strong count is
positive. Saved dependencies retain both Storage and Version.

## Phase-one trace driver

The executable accepted by the frozen ABI2 linker uses a checked HIA driver with
the existing `(readbuf,mutbuf,mutbuf)->u32` entry. Input begins with `HTOSTRC1`
and u64 frame count, then repeated u64 body lengths and exact request bodies.
The driver retains the genuine state owner between calls to `step`, and calls
`close` after the last frame. Frame input/output buffers are genuine owners;
only live loans are passed to `step`.

Trace output begins with `HTOSTRR1`, version 1, completed frame count and status,
then repeated u64 response lengths and bodies. The trace driver is a finite
qualification transport. It does not reply before stdin EOF and is not called
an interactive service. The following separate wrapper phase will use the same
checked `step`/`close` exports with true live read/reply/read-next framing.
Its distinct outer magic is `HTOWNR01` for requests and `HTOWNS01` for responses,
each followed by a u64 body length, at most 1,048,576 bytes. The wrapper treats
the service body as opaque and adopts the returned owner before response I/O.
