# Native owned storage object service

This isolated implementation supplies a native HIA byte-storage service with
checked object identities, shared lifetimes, independent mutation versions,
reusable pool extents and real interactive owner transfer. It is a foundation
for Tensor storage integration. It is not a complete Tensor implementation or
a claim that the PyTorch rewrite has closed.

The PyTorch audit denominator remains **2590** and
**`full_rewrite_complete=false`**. The existing HeptaTorch tensor runtime and
the pinned HIA toolchain were not edited by this implementation.

## Implemented and executed

The production candidate is the five business modules in `candidate-04/`.
Their genuine HIA `buffer` contains the object metadata and payload pool.
The metadata tables contain checked scalar handles and pool-relative byte
offsets. No pair of scalar words is cast into a capability or accepted as a
native pointer. The compiler checks every borrow, consume, return, branch and
drop in the owner functions.

The implementation includes:

- Six typed handle spaces: Tensor, Storage, Version, Weak, Saved and StrongHold.
  Every lookup checks kind, slot, live flag and exact generation. Reuse advances
  the generation; `0xffffffff` is never wrapped.
- Logical strong/weak reference counts. Tensor aliases, saved dependencies and
  storage holds retain the storage. Saved dependencies also retain the Version.
  The last strong release returns the extent to the free list. A weak reference
  can retain dead metadata, but cannot revive or read its released payload.
- Independent Storage and Version identities. New Version objects start at
  zero; one explicit byte write increments the selected Version once. Ordinary
  aliases share it. A separately named demonstration operation can share
  Storage with a fresh Version, without claiming the full `.data` API.
- Sorted free extents, interior-hole reuse, splitting and adjacent coalescing.
  A full free table can release an extent by coalescing in place without an
  extra temporary entry.
- Actual owned-buffer growth. Slot-table and pool capacities can increase while
  IDs, values, reference relationships and relative object offsets remain valid.
  Growth allocates a new HIA owner, copies valid bytes through live loans,
  then drops the former owner. It does not merely extend the old static arena.
- Whole-frame transactions. The committed owner remains alive while commands
  run on an independently allocated trial. A recoverable error drops the trial
  and returns the original owner, with a 32-byte response and no provisional
  IDs. Success drops the old owner and returns the trial.
- True interactive execution. `step(readbuf,mutbuf,buffer)->(buffer,u32)`
  consumes the old state and returns the sole next owner. A separately checked
  native startup starts with a live empty owner, reads one frame, invokes step,
  replies before stdin EOF, and passes only its returned owner to the next call.
  `close(buffer)->()` performs the checked drop on normal cleanup paths.

## Identity and layout

These identities are deliberately separate:

| Identity | Representation and lifetime |
| --- | --- |
| Tensor | Kind 1 slot/generation; records logical offset, length, Storage and Version IDs |
| Storage | Kind 2 slot/generation; records one pool extent and strong/weak counts |
| Version | Kind 3 slot/generation; records its mutation counter and references independently of Storage |
| Physical allocation | The actual linear HIA buffer owner; contains tables and all pool extents |

The physical allocation is not an address encoded in the Storage ID. A
transaction can replace the physical mapping without changing logical IDs.
The `resize_epoch` describes logical capacity increases only. It is neither an
allocation pointer nor a Tensor version. There is no public pointer export or
operation that reconstructs an owner from a state dump.

For slot capacity `N` and pool capacity `P`, all intervals below are disjoint:

| Region | Start | Length |
| --- | --- | --- |
| Header | 0 | 128 |
| Six typed tables | 128 | `6 * N * 64` |
| Sorted free extents | `128 + 384 * N` | `(N + 1) * 16` |
| Audit counters | `144 + 400 * N` | `24 * N` |
| Payload pool | `144 + 424 * N` | `P` |

Total allocation size is `144 + 424 * N + P`. Both the payload base and extents
are aligned to 16 bytes. Audit scratch and temporary growth-plan words are zero
at every committed frame boundary. Inactive records retain only their last
generation. All other fields are cleared.

The scalar layout is defined in `records.authoring`; the wire contract and the
five result words for every metadata kind are in [PROTOCOL.md](PROTOCOL.md).
`dump` is a diagnostic operation over the caller's whole logical pool. Free
payload bytes need not be erased; this service does not claim secure erasure.

## Qualification and exact artifacts

| Evidence | Artifact identity | What actually ran |
| --- | --- | --- |
| Finite trace driver | `67099048bccc809a835ea716e7c83c524f17f61b53e402f4632988be6356700f` | 12 scenario groups, 63 frames, 171 explicit scalar-field comparisons and 22 independently checked state dumps |
| Interactive service | `e4f7632e9cfc4c4d46d2c265b75bd3be2248085e1ef085c6ac04148cde06494d` | The same 12 scenario groups and 63 frames; every response observed with the same process still alive and stdin still open within its scenario |
| Trace/session parity | `trace-session-parity.json` | All 63 response bodies are byte-identical across those two different native entry paths |
| Extreme internal-state guards | `extreme-state-qualification-01/report.json` | Five current native guard witnesses plus one separately identified, preserved legacy refusal |

The two executions of the same scenario corpus are not 126 distinct upstream
PyTorch tests. No PyTorch API compatibility claim is derived from these service
tests. Per-request bytes, exact native responses, process exits and report
bindings are retained beside each report.

The central lifecycle witness allocates A/B/C, retains B through alias, saved
dependency and storage hold, drops B's original Tensor, then releases each
remaining strong reference. D reuses B's interior extent while A and C preserve
their bytes. Old Tensor, Version, Weak and Storage handles cannot acquire D.
The weak tombstone and its subsequently reused slot are checked separately.

Growth tests move `N=4` to 8 and then 32, and preserve all live values/IDs.
An additional payload contains 1,048,593 actual bytes and migrates to a
3,145,744-byte pool, beyond the former one-MiB tensor arena. Oversized reply
requests fail before publishing a partial response.

Allocation-failure tests impose `RLIMIT_AS=64 MiB` on the native child, then
request a valid 128-MiB pool through the normal reserve operation. This passes
the service's declared capacity checks and forces the allocator's actual
`buffer.try_new` failure path. The complete prior state dump remains identical,
and subsequent allocation/read operations succeed with the expected IDs and
extent. A first-allocation failure also returns the live empty owner and allows
a later first allocation. The evidence is native execution under the address
space limit, not a substituted host allocation result or a fake failure opcode.

The extreme-state fixtures obtain a genuine owner from step and seed only
scalar metadata through real mutable loans in a separate checked HIA driver.
They prove generation exhaustion is skipped, a wholly exhausted old table
grows, and `u64` Version/resize counters refuse another increment. Snapshot
bytes are observed, never reinterpreted as owners. Native audit and an
independent interval/refcount observer accept the injected states before and
after the tested operation. These fixtures are not public protocol features.

Cleanup-before-process-exit and transport-failure qualification belong to the
separate owned-session wrapper review. This service's successful EOF exit alone
is not presented as proof that memory was unmapped before kernel process cleanup.

## Reproducible native build

The frozen compiler is
`0ccdb524ea327ab290b2e70feaac49a4789abb2fee6da8746a095f435c17af73`.
The ordinary ABI2 linker used for the finite trace is
`23eed229e329fb5b7e2116012f74f210d5de4acd2b4f8475e364890748da86ee`.
The separately identified interactive owner-entry linker is
`55a2b0a762c8db574a65d0d0e6a18e3cedc668c01d20080a11c26cf1ccb601b3`.
These are local, qualified HIA profiles; they are not described as unchanged
upstream language support.

`build.py` verifies the pinned upstream support contract and all 32 authoring
source files before importing them, and verifies the exact loaded file set and
hashes again after generation. Python authors ordinary HIA only. The native
compiler performs compilation and ownership/effect admission; the native linker
performs symbol/interface checks and relocation. The executable contains no
Python storage runtime.

Run from this directory with absolute tool/output paths:

```bash
python -I -S -B build.py \
  --upstream /absolute/path/to/pinned-hia-source \
  --compiler /absolute/path/to/owned-module-compiler \
  --linker /absolute/path/to/linker-abi2-w1m \
  --output /absolute/path/to/new-native-build

python -I -S -B link_session.py \
  --modules /absolute/path/to/new-native-build \
  --linker /absolute/path/to/owned-session-linker \
  --output /absolute/path/to/new-session-build

python -I -S -B qualify.py \
  --binary /absolute/path/to/new-native-build/storage-trace \
  --transport trace --output /absolute/path/to/new-trace-review

python -I -S -B qualify.py \
  --binary /absolute/path/to/new-session-build/storage-session \
  --transport session --output /absolute/path/to/new-session-review

python -I -S -B qualify_guards.py \
  --compiler /absolute/path/to/owned-module-compiler \
  --linker /absolute/path/to/linker-abi2-w1m \
  --module-dir /absolute/path/to/new-native-build \
  --output /absolute/path/to/new-guard-review
```

The guard command runs the five current witnesses. Passing a distinct
`--legacy-module-dir` additionally replays the preserved old coalescing refusal;
that control is not a production requirement. `link_session.py` independently
reproduced the same `e4f763...` executable from the frozen five objects.

## Preserved failures and corrections

1. The first authoring-only attempt replaced `REQUEST_MAGIC` inside the longer
   `TRACE_REQUEST_MAGIC` token. It failed in the Python source emitter before
   any native compilation. The raw exception is preserved under `failures/`.
   Constant tokens are now replaced longest first.
2. The first native owner source contained blank lines between functions. The
   pinned native parser refused it with diagnostic code 45. The failed source,
   empty compiler output and native diagnostic are in `candidate-01/`. The
   author emits the same HIA statements without empty source lines. No checker
   or toolchain was modified.
3. Review identified that an extent-release insertion could reject a full free
   table even when immediate neighboring coalescing required no extra slot.
   Candidate 04 performs such coalescing in place first. A later native witness
   reproduced status 9 against the preserved candidate-03 function and success
   against candidate 04, with independent state invariants checked in both.

## Remaining integration boundaries

The service currently operates on byte lengths and byte values, with no dtype,
shape/stride descriptors or tensor arithmetic. Existing Tensor kernels and
autograd graph nodes still use their prior arena and descriptors. Connecting
those kernels to this Storage/Version foundation remains real implementation
work; the service is not silently substituted for the tensor runtime.

There is one native owned pool per service state. Its reference counts are
logical single-threaded counts, not PyTorch's concurrent intrusive-pointer API.
Native pointers/borrows do not escape a call; foreign buffers, stable external
pointers, custom deleters, device allocators, pinned memory and cross-process
serialization are not provided. A dump cannot restore an owner. Storage resize
for an individual existing Tensor, arbitrary strides, device synchronization,
atomic sharing and the full dispatcher/storage API remain open.

The current transaction copies the complete logical state before each valid
frame. This supplies clear rollback behavior but has linear copying cost and
temporary extra memory. Pool capacity is at most 1 GiB; each of the six tables
has at most 65,536 slots; the diagnostic audit refuses tables above 4096 slots
to bound its interval work. These are explicit candidate limits, not a reduced
definition of the user's requested full PyTorch rewrite.
