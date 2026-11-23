# Rank-1 ordinary CPU nonzero: six value dtypes

Exact #69 predecessor cb83670b272a6a9816b12c4ac8b2ef6f5473a087, tree
2a76d48da8ddd23a45df734c6020491e28fed88a. Candidate remains opt-in wire
opcode 135 and adds I64, F32, F64, F16 and BF16 to existing Bool. It
accepts ordinary owned contiguous rank-1 CPU roots with offset zero and
no grad, up to 262,144 elements. Bool remains canonical 0/1, I64 uses
all 64 bits, and floating values test the raw magnitude bits: both
signed zeros are false; subnormals, normals, infinities and signed NaNs
are true. No floating conversion or dtype promotion is involved.

Two distinct cold builds reproduced ELF
f952bf025e63f6545df424dff55e0d5a64cea196fc06a14d53ea76b5a37a9487
for combined HIA source
4b3274114665152bb66544a88b49b11047b1aac6d0b5438c3dc07727b01d0055.
The default, wire, allocation, shape and stride code are unchanged.

Locked CPU PyTorch 2.15.0+cpu (one thread) matched the frozen oracle in
all 62 rows including 12 over-cap reference-only fixtures. Both the
candidate and exact predecessor accepted all 50 in-scope input setups.
Nine exact-cap wide inputs were constructed using ordinary public F16
factory and exact 0/1 dtype-copy operations to F32, I64 and F64, with
complete raw root and ownership readback. The predecessor refused all
42 newly supported ordinary inputs with status 7 and unchanged owner;
the candidate passed all 50 rows and 1,715,103 output words. This
includes 18 exact-cap and 18 zero-selection rows. Both archived F32/I64
witness inputs are covered. Input bytes and identities were unchanged;
the fresh owning I64 result had shape [K,1], strides [1,K], Version 0,
offset zero and storage size 8*K, including K=0, independently verified
from response captures and live owner snapshots. Both destruction
orders preserved the remaining Tensor.

All 315 successful ordinary request/response pairs from the previous
#69 Bool success region, including its 17 successful nonzero rows,
replayed byte for byte. The 17 old expired-weak checks were not rerun.
No new malformed ownership, fault, pressure, above-cap native opcode
135, higher-rank, scalar, view, tuple/out or full default-gate claim.

The first old-run harness had a Python cleanup typo after its 653
responses; the rerun used the correct existing session finish method
and reproduced its request and response streams byte-for-byte. The
first prior-corpus replay attempt hit the 6.5 GB pre-launch guard and
did not launch a runtime; a later guarded replay completed above the
same threshold. Logs are included.
