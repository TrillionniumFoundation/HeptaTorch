# Opt-in I64 values for bounded CPU masked_select

This exact continuation of main-based #67 commit
`695c770273ef7a1260272d54c14610dbb0e282cf` keeps public wire opcode 130
and its existing F32 lane. It adds I64 values only with an ordinary Bool
mask whose shape is equal or right-aligned broadcastable to the fixed value
shape, including a true scalar mask. Value rank is 1–4, input count at most
262,144, strides/offset/ownership constraints unchanged. Result is fresh
rank1 I64 [K], element stride1, zero offset and Version0, with 8*K raw
storage bytes. No value promotion, symmetric expansion, views, grad,
dtype expansion, out variants or default runtime change.

Projection changes only ms_tensor and ms_output; geometry/mapping, count
limits, scratch restore, owner and wire cursor remain as in the predecessor.
The existing logical load returns raw u64, and the same byte store uses
the guarded width (4 for F32, 8 for I64). There is no float conversion.
Two distinct clean builds on exact combined source
`c5198ea3ddbbfbc3aeff244997dfa05660dba1c98d05edcf42d18c899b40768f`
are byte-identical, ELF
`5130bc446381049134363953a687656e7bbfe9bfe6fb87c331383e3acaf27cd8`.

Independent preregistered Cartesian fixtures and locked single-thread
CPU PyTorch 2.15.0 agree on 24 rows and 365,699 raw signed <q words.
The real no-Torch large inputs were synthesized with pre-existing
functional integer op103 on ordinary row and column inputs, followed by
full raw and owner validation; no use of cat. 22 rows/365,697 words are
in candidate scope. Both policy overcaps are real N=262656 with K=0 and
K=2; no physical allocator OOM is claimed. Wide signed patterns, exact
source bytes and 32767/32768/32769 read boundaries are covered.

The first exact projection builds matched but the first valid operation
returned status2: the allocator call had put tag2 into its requires-grad
slot. The exact one-line reversible correction moves the value dtype to
the final allocator argument and leaves requires-grad zero. For F32 the
effective tuple remains literally identical. The first exact request,
input owner, response digest and held build receipts are preserved.
The preliminary Python harness with an indentation fault stopped before
a native child; corrected and held runs use identical frozen fixtures.

Independent readback checks I64 and Bool input storage and output storage
directly from wire and owner snapshots, with a separate Cartesian index
calculation. It covers 325 exact predecessor frames and 24 actual status7
I64-value refusals. The corrected candidate covers 706 frames, 22
successful fixtures, and 37 exact pre/post rollback comparisons (22 weak
expiry records plus strict guards, including actual I64 byte-offset 8 and
Bool byte-offset 1 views, and whole-frame suffix rollback).
No scratch changes persist; operand handles and metadata stay unchanged.

The original 630-frame F32 one-way campaign, 1,043 previous comparable
frames, and 712 separately held previous witness-prefix frames compare
byte-for-byte (2,385 exchanges, including overlapping prefixes,
no retirements). This is not the entire hosted repository gate and not
a certification of unrelated tensor dtype or nonordinary layouts.
