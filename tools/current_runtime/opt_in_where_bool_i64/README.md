# Native CPU where.self: one explicit Bool/I64 Tensor value pair

Adds precisely Bool plus public I64 value tensors, in either order, when
the condition is Bool. The original same-dtype fast paths remain, as do
all unrelated promotion rules and wire ABI. Inputs independently obey
width-1 or width-8 ordinary CPU root, shape, live owner and raw bounds.
The existing right-aligned shape allocator copies into a fresh result,
always public I64 for this pair. When selected, I64 words copy unchanged;
a selected canonical Bool explicitly produces I64 0/1. Scalar means an
actual typed rank-zero Tensor, not a wrapped Python scalar. This is a
deploy-disabled opt-in native slice, not default or hosted activation,
general mixed promotion, views, autograd, or out=.

Two cold builds produced byte-identical ELFs. Independent locked CPU
PyTorch, a no-Torch capture, and a third raw-coordinate reader agreed on
12 cases and 108 I64 values, including both orders, four rank-zero
outputs, all four orders/roles with only one rank-zero value, rank-four
expansion, and two empty outputs. The raw reader observed 47 selected
Bool values (23 false, 24 true) and 61 selected I64 words, including the
signed extrema and signed values beyond 2^53. Fresh Tensor/Storage/Version,
zero initial Version, both release orders, weak expiry, unchanged input
bits/metadata, and request-local output consumption were verified.

The previous exact ELF refused the 12 valid setups with status 7 and
unchanged complete owners, separately (120 frames). The successor's new
suite verified 31 complete owner rollbacks in 289 frames. These include
real Bool and I64 positive-offset input views in either order (four),
a compatible 262,656-element capacity refusal (status 3), incompatible
shape, malformed/stale input, an accepted operation followed by invalid
suffix, and both orders of four unadmitted mixed pairs (including F32/I64).
All 4,261 previous raw request/response frames and 452 previous complete
rollbacks independently replayed byte-for-byte, with no additional retired
control. Total successor: 4,550 frames and 483 rollback controls.

An initial offline diagnostic noticed the old reader summary had neither
a rank-zero bucket nor a guard for empty shape-vector indexing. These
reader-only corrections were made before independently reading the same
unchanged native captures. No compiler failure or candidate change occurred.

Use isolated Python for `prepare_bool_i64.py EXACT_PARENT EXACT_BUILD_PY`.
Optional final directory materializes the checked split projection. Cold
build with the original pinned tools and use the included exact pin manifest
for the second build. Run `oracle_bool_i64.py PINNED_HELPERS ORACLE_LOCK
INSTALL_REPORT FIXTURE` with the locked official wheel only. Run
`native_bool_i64.py PINNED_HELPERS PROTOCOL_PY ELF FIXTURE OUT
baseline|candidate` in a separate interpreter without Torch for both
ELFs, then independently use
`pure_bool_i64.py FIXTURE BASELINE_CAPTURE CANDIDATE_CAPTURE`.
Retained suites change only the candidate executable pin.
