# Native persistent session v1

This profile keeps the existing bounded HeptaTorch tensor arena alive across
interactive requests in one Linux x86-64 process. A response is written before
the next request is read. The process does not wait for stdin EOF to execute a
request. The native runtime, framing, state copying, error handling, linking
and packing contain no Python or C numerical runtime.

It establishes a bounded persistent execution foundation. It does not provide
general resizable storage, cross-process durability, an owned-resource module
ABI, or complete PyTorch coverage. The frozen numerical source remains the
single authority in `src/heptatorch.hepta`; this profile substitutes only its
entry function in build storage.

## Frame layout

All integers are unsigned little-endian u64 unless stated otherwise.

| Offset | Request frame | Response frame |
| --- | --- | --- |
| 0 | Eight bytes `HTOSES01` | Eight bytes `HTOSRP01` |
| 8 | Body length, 0 through 1,048,576 | Body length, 32 through 1,048,576 |
| 16 | One complete `HTORCH01` packet | One complete `HTORSP01` packet |

The body is the existing packet format. Command/configuration/payload offsets
remain relative to the start of the body, with the same checked bounds. No
additional numerical command is introduced. Tensor IDs returned by a
successful frame remain usable in later frames of that process, subject to
the existing mutation/version and graph-reset rules. Saved index/configuration
data are arena offsets and copies, not pointers into the reusable input frame.

A zero-length, short, or otherwise invalid packet body produces packet
status 100 and leaves the process ready for another frame. A complete valid
zero-command packet is a successful transaction: on a fresh process it
initializes the tensor arena. Invalid first packets do not initialize durable
state. Per-frame input and output remain bounded; the cumulative input of a
live session may exceed those per-frame limits.

The wrapper reads exactly the header and then exactly the declared body. It
does not consume bytes belonging to a later frame. Pipelined frames are
processed and answered in stream order. An incomplete frame has no response.

## Complete-frame transactions

The process owns one private anonymous mapping of 4,198,400 bytes:

| Relative offset | Bytes | Purpose |
| --- | ---: | --- |
| 0 | 4,096 | Frame/control records and SIGPIPE action |
| 4,096 | 1,048,576 | Current request body |
| 1,052,672 | 1,048,576 | Current response body |
| 2,101,248 | 1,048,576 | Tensor workspace A |
| 3,149,824 | 1,048,576 | Tensor workspace B |

Control offsets 32 and 40 hold the durable and shadow workspace pointers. For
each complete bounded frame, the wrapper copies all 1,048,576 durable bytes to
the shadow, then calls the real HIA entry with the shadow as its workspace.
The six-word native call carries input pointer/length, output pointer/length,
and shadow pointer/length. The workspace length supplied to numerical code is
exactly one MiB, not the size of the full mapping.

On status zero, the pointers are swapped. On any nonzero packet status, the
durable pointer and its bytes are unchanged. Thus a command that allocated or
mutated successfully before a later command failed does not leak a partial
commit. This includes tensor contents, gradients, optimizer states, versions,
descriptor slots, allocation cursor, graph traversal metadata, and a graph
reset that was followed by a failing command. Output-capacity exhaustion also
aborts the entire frame.

Every failed frame publishes exactly this 32-byte body:

| Offset | Value |
| --- | --- |
| 0 | `HTORSP01` |
| 8 | 1 |
| 16 | 0 response records |
| 24 | The original nonzero packet status |

No successful trial record, tensor ID, value payload, or failing command
record from that frame is published. Successful response bodies remain byte
compatible with the original single-request protocol. The failure-body rule
is a deliberate property of the separately named session framing protocol.

A speculative ID that was never returned is not a granted handle. Its numeric
slot may be reused after rollback. This profile does not claim that every
guessed per-allocation number is permanently unique. Previously published
handles still receive the complete existing generation check: a committed
graph reset invalidates discarded IDs even after slot reuse. A rolled-back
reset does not advance the durable generation.

The commit decision happens before response transmission. An I/O failure
terminates the process, so there is no later frame in that process after a
partially transmitted response. No reconnect, replay, or exactly-once
cross-process delivery guarantee is implied.

## Transport termination

| Condition | Result |
| --- | --- |
| EOF before any byte of the next header | Exit 0 |
| EOF during a header or body | Exit 2; no response for that incomplete frame |
| Wrong frame magic | Exit 2, before body processing |
| Declared body greater than one MiB, including u64 maximum | Exit 3, before reading that body or copying tensor state |
| Read/write failure other than EINTR, or zero-progress write | Exit 4 |
| Extra argv arguments | Exit 64 |

Partial reads and writes advance by the actual positive byte count. EINTR
retries the same remaining pointer and count. The production wrapper installs
the real Linux `rt_sigaction(SIGPIPE, SIG_IGN, NULL, 8)` action, so a closed
output pipe follows the I/O error path and exits 4. It installs no SIGUSR1
handler. The separate qualification adapter installs a signal observer only
for tests; its identity is not the production executable identity.

## Storage boundary

Whole-arena copying is valid for this runtime because tensor data, gradients,
work arrays, shapes, saved indices and lists are offsets into that same arena.
Its 255-descriptor and one-MiB arena limits remain. `graph_reset` permits long
training loops by reclaiming transient graph allocations while keeping the
initial validated leaf/state prefix.

The workspace contains no external heap owners or borrowed host pointers.
Copying future owned descriptors would neither isolate their payload nor
preserve unique ownership. A future owned-storage session needs a separate
typed move/return and failure protocol, or a suitable deep-copy/journal/COW
design. It must not reinterpret this borrowed six-word entry as an owner ABI.

## Native modular and packing profiles

The modular path independently compiles the complete source partition and
links actual machine code. The flat linker uses `HIAFLK01`, distinct from the
frozen executable linker's `HIABLK01`. Its 25 admission/helper HIA functions
are byte-identical to that linker. Only the wire-magic constant in `link` and
the output `headers` function change.

The new output begins with the upstream 78-byte six-word native bridge.
Object text still begins at offset 1,024. The selected public entry is checked
by the unchanged native `module_entry` gate and must be
`(readbuf, mutbuf, mutbuf) -> u32`. Its call displacement at byte 65 is the
selected entry offset minus 69. The unchanged canonical-object gate admits
only the checked call relocation family with addend -4; relative source and
target addresses cancel the old ELF base, so the raw image runs at different
mapping addresses. Native signature, effect, dependency, symbol, relocation
overlap and capacity checks are retained.

The module compiler and object producer custody remain part of the contract.
Interface records do not authenticate arbitrary independently fabricated
machine code. No owned resources or foreign allocator calls have been added
to the frozen borrowed profile.

| Capacity | Exact bound |
| --- | ---: |
| Flat-linker objects | 64 |
| Flat-linker input/output bytes | 8,388,608 each |
| Flat-linker workspace bytes | 16,777,216 |
| Raw image admitted by session packer | 8,388,608 |
| `HIAFLT01` packing request, including its 16-byte header | 8,388,624 |
| Packed session ELF, including 774-byte startup | 8,389,382 |
| Wide packer's actual native output allocation | 8,392,704 |
| Wide packer's actual native workspace allocation | 4,096 |
| Wide packer's actual native total mapping | 16,789,504 |

The wider packer is itself compiled and packed by native HIA. Its native
wrapper really allocates the listed input/output/workspace regions; these
bounds are not solely source-level checks. Current monolithic and modular
session images produce byte-identical executables through the old small
packer and this wider packer. Tensor workspace capacity is unchanged.

Compiler limits remain per unit. The linker and packer provide an actual
multi-unit path above one unit's 8,192-instruction limit and the former
one-MiB packing limit. Aggregate limits, object count, stack requirements,
the current source partitioner's policy and the tensor arena remain finite.
