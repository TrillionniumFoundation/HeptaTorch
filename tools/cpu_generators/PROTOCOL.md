# CPU Generator wire protocol v1

All integers are little-endian. Signed int64 bounds are represented by their
64-bit two's-complement bit patterns. There are no pointer-valued wire fields.

## Framing and request

An input frame is ASCII `HTOWNR01`, a u64 body length, then exactly that body.
An output frame uses `HTOWNS01` and the same length framing. Each body is limited
to 1048576 bytes. The process responds before reading the next frame and retains
its one owned state between frames. EOF closes that state. A malformed transport
header or truncated frame terminates through the existing close path; it is not
a normal application response.

The request body begins with exactly 64 bytes:

| Offset | Field |
|---:|---|
| 0 | Eight ASCII bytes `HTCGEN01` |
| 8 | u64 operation |
| 16 | u64 Generator handle |
| 24 | u64 count |
| 32 | u64 argument a |
| 40 | u64 argument b |
| 48 | u64 payload length |
| 56 | u64 reserved, must be zero |
| 64 | Payload bytes |

The body length must equal 64 plus payload length. Every unused argument must
be zero. Only `set_state` accepts a payload, exactly 5056 bytes.

| Op | Operation | Nonzero/required arguments | Response payload |
|---:|---|---|---|
| 1 | Create | count 1..remaining slots; a is seed | count u64 new handles |
| 2 | Manual seed | handle; a is seed | Empty |
| 3 | Nondeterministic seed | handle | One u64 seed |
| 4 | Current seed | handle | One u64 seed |
| 5 | Get state | handle | Exact canonical 5056-byte state |
| 6 | Set state | handle; 5056-byte payload | Empty |
| 7 | Clone state | handle; count 1..remaining slots | count u64 independent new handles |
| 8 | Advance | handle; count raw32 words | Empty |
| 9 | Raw32 | handle; count | count u64 values, upper 32 bits zero |
| 10 | Raw64 | handle; count | count u64 values |
| 11 | Uniform float32 | handle; count | count u64 IEEE bits, upper 32 bits zero |
| 12 | Uniform float64 | handle; count | count u64 IEEE bits |
| 13 | Uniform float16 | handle; count | count u64 IEEE bits, upper 48 bits zero |
| 14 | Uniform bfloat16 | handle; count | count u64 bits, upper 48 bits zero |
| 15 | Int64 range | handle; count; a=low inclusive, b=high exclusive | count u64 int64 bit patterns |
| 16 | Release | handle | Empty |
| 17 | Statistics | None | Empty |
| 18 | Dump internal bytes | count bytes; a=byte offset | Selected owned-state bytes |
| 19 | Default construction | count 1..remaining slots | count handles; seed 67280421310721 |

Zero is a valid seed. Draws and advance allow count zero. Draws return at most
100000 values, advance consumes at most 1000000 words, and the general engine
word budget is 1000000. The importer, numeric transforms and state consumption
are the frozen RNG03 semantics. Uniform values are in `[0,1)`; float16 and
bfloat16 results that round to exactly one are clamped to zero, matching the
pinned CPU kernel. Raw64 concatenates the first raw32 word as its high half.

`dump` is a read-only qualification/inspection operation. It accepts a byte
range entirely within the existing owner and returns at most 1048512 bytes.
The stored bytes contain identifiers and RNG state, not pointers. Dumping
zero bytes at the end is valid, including the initially empty state.

## Response

Every application response is at least 64 bytes:

| Offset | Field |
|---:|---|
| 0 | Eight ASCII bytes `HTCGRS01` |
| 8 | u64 status |
| 16 | u64 payload length |
| 24 | u64 live Generator count |
| 32 | u64 owned-state bytes |
| 40 | u64 last allocated handle |
| 48 | u64 mutation epoch |
| 56 | u64 reserved zero |
| 64 | Payload bytes |

Status 0 means success. All nonzero statuses have no payload and report the
original retained state. Status 1 is malformed input/arguments; 2 is an invalid
RNG state; 3 is capacity, output/work budget or allocation failure; 4 is an
unknown or released handle; 5 is an unknown operation; 8 is identifier or epoch
exhaustion; 9 is missing/failed trusted entropy; 10 is invalid internal state.
When several errors are present, the implementation's ordered validation
determines the first returned status. There is no promise of matching PyTorch's
exception classes or error text.

The epoch increases once per successful mutating frame, including batch create
or clone. Read-only operations and zero-count generation leave it unchanged.
At `UINT64_MAX` the epoch refuses further mutation. Handles never wrap; creating
the final `UINT64_MAX` handle is valid, and a further allocation fails. Release
preserves the last-handle counter. Releasing an unknown handle is an error.

## Stored layout

Initial state is a genuine zero-length owned buffer. Once initialized, its
length is exactly `64 + 5088 * live_count`:

| Header offset | Value |
|---:|---|
| 0 | ASCII `HTCGST01` |
| 8 | Last allocated handle |
| 16 | Live count, at most 1024 |
| 24 | Mutation epoch |
| 32 | Version 1 |
| 40 | Record width 5088 |
| 48, 56 | Zero |

Each record contains its u64 handle, three reserved zero u64s, and the complete
5056-byte canonical state at record offset 32. Records may move within the
compact allocation; handles remain stable. Record order is not an object API.

The 5056-byte serialized state is seed u64 at 0; left u32 at 8; seeded u32 at
12; next u64 at 16; 624 u64 words at 24; dead normal_x at 5016; normal_y cache
bits at 5024; dead normal_rho at 5032; double-cache flag u32 at 5040; padding
at 5044; float-cache bits u32 at 5048; float-cache flag byte at 5052; padding
at 5053. Dead fields and padding are zero in canonical output. The engine uses
the low 32 bits of serialized next and each word; nonzero seeded normalizes to
one. Cache flags must be zero or one, enabled cache bits are preserved exactly,
and disabled caches become zero. Seed/manual_seed clear both caches. Ordinary
draws and advance preserve them; normal sampling itself remains unimplemented.

Accepted positions are `(left,next)=(1,0)` or `1<=next<=624` with
`left+next=625`. This is an intentional canonical/reachable-state subset; it
does not duplicate acceptance of inconsistent upstream states that could cause
an out-of-bounds engine access.
