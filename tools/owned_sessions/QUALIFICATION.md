# Native owned-session wrapper qualification

This isolated implementation supplies the syscall prefix for the separately
checked native HIA owned-session linker. It never changes the previously
qualified borrowed tensor/session profiles. The production prefix is 1,059
bytes, SHA-256 `f68e28de33e2d4120b1a8a4b4463dc9b1d63f095eb8920955d43647ab3fa7f8b`.

## Owner and framing contract

The native linker checks both exported signatures and effects:

```
step(readbuf, mutbuf, buffer) -> (buffer, u32)  effects(alloc, read, write, trap)
close(buffer) -> ()                          effects(alloc, trap)
```

Startup creates a trusted live-empty owner `(0,0)`. Each call moves that owner
into the checked HIA `step` parameter and adopts exactly the returned owner.
The previous home and dead argument/result slots are cleared. User input never
supplies a native owner pointer. The wrapper does not clone an owned descriptor,
copy a heap allocation for rollback, or interpret application status. The HIA
service implements its own transaction and returns its original owner on an
application failure.

Requests are `HTOWNR01` followed by a little-endian u64 body size and that many
body bytes. Responses use `HTOWNS01` and the same 16-byte framing. A body is at
most 1 MiB; a returned response must have 32 through 1,048,576 bytes. Each reply
is written before the next frame is read, while standard input remains open.
The wrapper allocates a 4 KiB control page and independent 1 MiB input/output
buffers. The service's owned allocation is separate and may exceed those
frame buffers.

After the owner becomes live, all normal termination paths share one checked
`close` call. The pending exit status is 0 for clean between-frame EOF, 2 for
malformed/truncated framing or an invalid returned length, 3 for an oversized
request, and 4 for a read/write error. `EINTR` retries the unchanged remaining
span; positive short I/O advances its pointer and remaining length. SIGPIPE is
ignored through native `rt_sigaction`, so a broken pipe returns EPIPE and
reaches close. The cleanup restores the original stack alignment even when
entered from a helper. An argument-count error exits 64 before introducing an
owner. A startup mapping failure exits 4 before introducing an owner.

External fatal signals and traps inside a HIA callee are not represented as
successful calls to `close`; operating-system reclamation is a separate fact.
The qualified target is Linux x86-64. This sequential owned byte-storage
foundation does not establish full PyTorch storage or tensor compatibility.

## Three explicitly different native profiles

| Profile | Linker SHA-256 | Purpose |
| --- | --- | --- |
| Production | `55a2b0a762c8db574a65d0d0e6a18e3cedc668c01d20080a11c26cf1ccb601b3` | `HIAOWNS1`, exact production prefix |
| Post-close qualification | `4381b6d3425b501c8a1b5e2a5496ab0f27037e18f0e74727b321006327e6a9ee` | `HIAOWNT1`, pauses after actual HIA close |
| Signal qualification | `07af93a2b1986147521ec55f23ff2b5704e001ae903e063263007afddb274df6` | `HIAOWNT1`, installs a kernel-register observer |

The post-close profile writes a private fd3 marker and awaits an fd4
acknowledgment only after HIA close has returned. Tests inspect their own live
child through `/proc` to prove the owner's mapping is already absent. This
distinguishes an actual drop from process-exit reclamation. Dead argument bits
are inspected without reusing them as a second owner. No pointer is sent in
the public response stream.

The signal profile installs a test-only SIGUSR1 handler without SA_RESTART.
The handler only copies kernel-supplied saved registers; it never changes
ucontext. All 124 bytes of the production read/write loops are identical.
Signals are sent only after `/proc` confirms a blocked read or write syscall,
where the wrapper owns r12. This observer is not exercised during a HIA callee.
The production prefix contains neither qualification hook.

## Recorded execution

The five actual candidate-04 storage modules, ordered bytes/records/audit/ops/
owners, were compiled by the checked native compiler and linked into:

| Image | SHA-256 |
| --- | --- |
| Production | `e4f7632e9cfc4c4d46d2c265b75bd3be2248085e1ef085c6ac04148cde06494d` |
| Post-close qualification | `4f6fb5d462e5d357440393c8ee79165ae7d4d86a6b593cd7e1f69beffbdc792c` |
| Signal qualification | `37f77b6178090a8d361b75e01099d5d11627f48420c685b7b18599c5c2099f04` |

All three use identical linked HIA object text from offset 4096 onward.
Different object orders are legitimate distinct images and retain separate
receipts.

| Suite | Actual result | Receipt |
| --- | --- | --- |
| Transport and cleanup | 20 cases, 29 replies before input EOF; 15 live post-close unmap observations | `transport-01/report.json` |
| Returned u32 length | 6 cases, 12 native processes, 8 replies; 6 live post-close unmap observations | `result-bounds-01/report.json` |
| Real signals and short writes | 2 cases, 6 replies; three real EINTR returns and one positive short write | `real-signals-02/report.json` |

Transport tests include admission of an exact 1 MiB request, fragmented and
pipelined frames, application rollback, short headers/bodies, oversized u64
lengths, actual EPIPE/EAGAIN/EISDIR, and cleanup of an owned allocation larger
than 2 MiB. Returned-length tests vary only the u32 scalar in real HIA owner
fixtures: 0/31/1,048,577/u32-max reject with exit 2 and no response; 32 and
1,048,576 accept and preserve the real owner's incrementing byte across frames.

The signal execution observed a write of 264,080 bytes returning a positive
short count of 63,376, followed by a retry at pointer+63,376 and length 200,704.
The next interrupted write returned actual -EINTR and retried the same span.
All 263,984 payload bytes matched the previously committed state, and a later
frame still read the original tensor byte. These counts are observations from
this run, not portable assumptions about pipe capacity.

The storage author and root separately ran business/reference-model suites;
their original receipts retain their own source and executable identities.
The counts above do not include or relabel those independent results.

## Portable test interfaces

All Python files here are source authoring, wire transport, or independent
observation tools. The child executables run with an empty PATH and execute
native HIA allocation, mutation, transactions, and drop.

Keep these test files and `client.py` in the same directory. Keep the two
small native HIA fixtures in the adjacent `fixtures/` directory.

```
python -I -S -B test_transport.py \
  --service SERVICE --close-probe CLOSE_PROBE \
  --build-report NATIVE_BUILD_JSON --output NEW_TRANSPORT_DIRECTORY

python -I -S -B test_result_bounds.py \
  --compiler OWNED_COMPILER --linker PRODUCTION_LINKER \
  --qualification-linker POST_CLOSE_LINKER --output NEW_BOUND_DIRECTORY

python -I -S -B test_real_signals.py \
  --service SERVICE --signal-probe SIGNAL_PROBE \
  --signal-metadata SIGNAL_PREFIX_JSON --output NEW_SIGNAL_DIRECTORY
```

`link_service.py` accepts the three checked native producer paths and an ordered
list of `.hepta` sources. It snapshots sources, actually compiles each module,
actually native-links production and post-close images, and verifies source,
object, and producer identities before and after. The signal image must be
native-linked from those same objects with the signal qualification linker.
Output directories must be new, so earlier observations and failures are not
silently overwritten.
