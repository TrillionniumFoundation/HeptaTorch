# Opt-in bounded one-way Bool-mask broadcasting for F32 masked_select
               
This continuation targets exact main-based candidate #67 commit
`ad964ebaf3f7b3913f039d3ee44802aaff447198` (tree prefix `21bdbf9`),
not default runtime installation. It preserves the opcode-130 frame and
existing equal-shape cases. It admits right-aligned ordinary Bool masks
of rank 0–4 only when the result broadcast shape is exactly the ordinary
F32 value shape (rank 1–4). Each dimension and actual input count is bounded
by 262,144. A true scalar mask selects all values and false selects none.
Value expansion, all rank-zero values, higher-rank masks, non-owned
layouts/offsets, requires-grad values, other dtypes and `out` remain outside
this scope. No whole-runtime or full PyTorch acceptance claim.

The value stream is traversed in logical C-order. For unequal shapes,
each mask index uses the actual aligned mask dimensions, assigning zero
coordinate to a singleton axis. Both compaction passes use the same
bounded coordinate helper; before reading mask data the final result must
fit the actual mask count. The count loop tests `i < N` before the first
mask data load. For an empty value count it bypasses that load, produces
K=0, and bypasses the second pass; this is a source control-flow argument
backed by actual empty cases, not a native memory-access instrument. The
existing two-pass allocation/restore and commit/rollback path is unchanged,
as are element stride 1, typed store byte width 4 and the live reply cursor.
No physical allocator OOM was induced; status3 here is an input-policy cap.

`prereg.json` was frozen before the locked real Torch run. Its Cartesian
expectations include exact zero/dense polarity, qNaN/sNaN payloads and
ordering-sensitive repeat patterns. `torch-receipt.json` verifies pinned
PyTorch 2.15.0+cpu at threads1: 23 rows, 527,549 words, no mismatches. PyTorch's
prefix-sum route can apply even on one thread because a broadcast operand
is not contiguous. The three overcap rows are intentionally refused by
HIA even when PyTorch returns zero or two words.

`pure.json` independently reads raw wire and owner snapshots: 280 exact
predecessor frames with 23 actual policy refusals, and 630 candidate
frames, 20 successes (264,891 words) and 35 exact rollback comparisons.
Positive cases include all five zero-value targets and both lifetime
destruction orders. Rollback includes actual Bool offset view, requires
grad F32, unsupported geometry and same-frame successful compaction
followed by an unknown operation. Seeded scratch remains unchanged.

Tracked source reproduction starts from the exact prior split projection,
not from a changed default baseline. `project.py` adds only two functions
to the same split unit and revises `ms_tensor`, validates the exact
inverse and existing split checker, then `names.py` checks the pinned
native keyword/loader reservations. The first and second build receipts
bind each source and unit; SHA-256 of each ELF is
`458e9cffa53dbb77a4474a992775aa62b64dd6c25b314c0cfd0e1ab8fbb7494f`.
Decode complete locked Torch raw words in receipt-offset order with:
`lzma.decompress(base64.b85decode(b''.join(Path('torch-words.bin.xz.b85').read_bytes().split())))`.

The replay comparison preserves 1,043 prior exact request/response pairs,
including 262 comparable earlier main setup frames (not the entire
hosted main gate). Only two physical records were retired from the
comparable set: the same locked `[2,3] + [1,3]` operand fixture recorded
twice with independently generated IDs, at archived guards frame 20 and
candidate frame 692. Both are physically replayed from the exact
preceding state: the previous image refuses without mutation, and the new
image selects the same six raw words in order. `replay-receipt.json` pins
these actual frames, and `replay_readback.py` independently checks raw
retained bytes. The original first replay that noticed the second
occurrence and its emitted status0 record are separately retained.
