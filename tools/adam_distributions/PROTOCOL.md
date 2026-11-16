# Combined Adam and distribution profile contract

The production entry stays `step(readbuf, mutbuf, buffer) -> (buffer, u32)` with
effects15, and `close(buffer) -> ()` with effects9. The frozen native wrapper
owns the one returned buffer across frames and closes it at termination. It
does not clone owner descriptors or accept user-supplied native pointers.

Transport framing stays `HTOWNR01` / `HTOWNS01`. Tensor request/response bodies
stay `HTORCH03` / `HTORSP03`. A frame is at most 1 MiB. The internal runtime tag
is now `HTADIST1`; the object-pool tag is `HTOADIS1`. Those tags intentionally
reject both predecessor owners. A rejected handoff returns the exact original
owner, which remains usable by its producing service.

## Object state

The actual Adam02 eight-kind layout is preserved. The object subsystem begins
at owner byte139264. Each kind has `slots` records of64 bytes; the pool data
offset within that subsystem is `144 + 552 * slots`. Tensor graph descriptors
remain384 bytes. There is no seventh-table-only overwrite or memcpy of an
external owner into this layout.

| Kind | Object |
| --- | --- |
| 1 | Tensor binding |
| 2 | Storage |
| 3 | Version |
| 4 | Weak Storage reference |
| 5 | Saved |
| 6 | Private Hold |
| 7 | CPU Generator |
| 8 | Optimizer |

Generator5056-byte state continues to live in an owned pool allocation retained
by a private Hold. Optimizer parameter and state roots remain actual Tensor
references. Node312 is private strong Tensor references, node320 is Tensor
incarnation, and header120 is the next incarnation. Header112 is the active
backward node, cleared before a successful frame is returned. Existing Saved,
captured-edge and temporary gradient-field invariants are preserved.

## Command composition

All existing commands keep their predecessor wire definitions. The combined
entry accepts the union, with no new numeric operation introduced here:

| Commands | Contract |
| --- | --- |
| 57 / 58 / 59 / 70 | Atomic backward / explicit partial backward / gradient presence / clear to None |
| 60–69 | Explicit CPU Generator and binary32 uniform/view operations |
| 71–79 | Actual owned SGD, Adam and AdamW objects, state publication and release |
| 80–85 | Atomic Bernoulli, ordinary dropout and Saved noise alias operations |
| 86 | Explicit partial Tensor-probability Bernoulli |
| 87 | Explicit partial ordinary in-place dropout |
| 88 / 89 | Unknown commands, error4 |
| 90 | Existing captured-edge-aware requires_grad_ contract |
| 95 | Existing explicit no-grad whole-Tensor scalar addition |

Functional Bernoulli uses graph opcode41 with two generic parent slots:
`record(result,41,probability,0)`. The second slot and captured flag are zero.
The old graph audit, parent retention and release machinery already handles
this shape. Its derivative writes genuine zero contributions through the
existing staged-gradient machinery. It does not draw random values backward.

## Commit and rollback

The unchanged actual owner step creates a private trial state. Ordinary
admission, allocation, capacity or operation failure drops the trial and
returns the original owner. This covers Tensor writes, Versions, Generator
advancement, optimizer state and all preceding commands in that frame.

Commands58,86,87 retain their explicit runtime-error exception: the response
record has outer status0 and a nonzero operation error in `value`; the entry
stops the remaining suffix and returns the trial owner. This commits exactly
the prefix defined by that operation, including preceding optimizer steps.
It does not convert a later ordinary error after a successful explicit
operation into partial commit.

For86/87 the32-byte record payload is four little-endian u64 words:
`[operation_error, Tensor_handle, completed_writes, consumed_raw32]`.

For86, an invalid per-element probability consumes no RNG for that element.
Earlier valid writes and draws commit; the Version increment happens only on
complete success. For87, nontrivial dropout can draw its temporary noise
before a tracked-leaf or stride-zero overlap refusal. That operation error
commits Generator advancement, leaves the target values and Version unchanged,
and retires the temporary noise. The frozen partial01 implementation supplies
these behaviors unchanged.

Empty86/87 requests contain invalid typed handles and return ordinary error1
with whole-owner rollback, as in frozen partial01. They are no longer unknown
commands. This is the sole error-class adjustment in the explicitly derived
old atomic admission control;80–85 expectations and88/89 unknown behavior do
not change.

## Retained boundaries

Tensor rank is at most4, descriptor ordinals at most65535, and Tensor numel at
most16,777,216. Distribution active work remains bounded by1,000,000 elements.
Optimizer groups are bounded by1024 and parameter occurrences by4096. Adam's
actual binary32 incremented step must be a positive integer below2^32; this
still excludes general real-exponent bias corrections. The existing graph,
pool, input and response capacity checks remain in force.

There is no claim of arbitrary externally fabricated owner-state
authentication. The persistent owner has checked native producer custody;
wire requests cannot install arbitrary owner bytes or a foreign pointer.
The qualification-only owned entry harness explicitly marks the controlled
byte faults it uses to test refusal. Public `.grad` object semantics remain
outside this profile until their separate real object implementation is
composed.
