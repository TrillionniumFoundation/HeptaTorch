# Opt-in rank-2 ordinary Bool nonzero

This #69 update is based on exact dceb788712c7af08f00ea61c7c507dbbe3493dc7,
tree 84a833773eee3dbed33c5f599c689a702a535f7e. It adds only canonical
ordinary zero-offset rank-2 CPU Bool to the existing opcode135 nonzero.
The prior six-dtype rank-1 bodies and their outputs are unchanged. No
change to default runtime, wire geometry, tuple/out API or higher ranks.

Each selected coordinate is in lexicographic (row, column) order. The
fresh I64 output has shape [K,2] and strides [1,K], physical columns
stored consecutively, exact 16*K storage bytes, Version0, no view or
nonzero offset. K=0 and K=1 are contiguous; K>1 is not row-major
contiguous. The implementation validates row-major source geometry
before work, operates under N≤262144 and K≤131072, restores all scratch,
and publishes only after raw column materialization and exact private
stride commits. It does not rely on a global allocator layout flag.

Locked CPU PyTorch measured all 26 frozen rows. The 23 scoped rows have
786,800 I64 words each in two independently measured streams. All 26
including three over-profile references have 1,048,950 words per
stream. The three above-cap rows were not issued to native and do not
prove a native refusal, allocator limit or out-of-memory event.

Two clean cold builds produced identical ELF
ac17c641a0e90a08a79dc913e0a3afd2476797ab40d5a48d06190f3b8fd12d33
from combined source 8420d6599d484e2b6844576a0f60207d6b318cb21bda269f0d60bf135bac10db.
The exact predecessor responded status7 on all 23 real ordinary rank-2
setups without owner changes. The candidate admitted 23/23, including
eight K=0, two K=1, four exact N cap and two exact K cap rows. The raw
readback verifies each logical wire stream and each physical Storage
stream, fresh owner identities and Version0, byte-exact preservation
of the source, scratch and both destruction orders. Physical/logical
orders differ on 13 scoped rows. The preserved old rank-2 witness
includes its exact factory and live authority context, not a bare q hash.

All 1,197 ordinary positive request/response pairs of the exact
six-dtype rank-1 campaign, including 50 native rank-1 nonzero results,
replayed byte for byte. This is a scoped positive corpus, not a full
default-runtime or unsupported/malformed-path qualification.
