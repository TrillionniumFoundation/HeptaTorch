# Ordinary rank-3 CPU nonzero: all six public input dtypes

Exact parent #69 606e5c182d8132b0646a62fb9c51aad3fbc6b371, tree
42e0bb1a78d9e4439de71832aefafefb164477a5. This adds I64, F32, F64,
F16 and BF16 to existing Bool for opt-in rank-3 ordinary contiguous
owned CPU tensors with zero offset and no grad. Rank-3 passes checked
dtype width and tag into the existing shared typed scan. Private tag3
still excluded. Signed floating zeros are false; signed subnormal,
normal, infinity and q/sNaN bit patterns are true. I64 truth uses
the full word. No type conversion or global arithmetic change.

The single changed body is limited to the rank-3 entry shape blocks.
Rank-1/2, wire135, allocation, caps, default and output semantics
are unchanged. Fresh owning I64 output has [K,3], element strides
[1,K], storage 24*K, zero offset and Version0. Logical coordinates
are plane/row/column in lexicographic order; backing storage contains
the columns. K=0 and K=1 are row-major contiguous.

Two independent cold builds reproduced the same ELF
db28aa0c8f2ca4f59ccf7a8d52eb30a324d8d1464fdd7ac7a3cf920df5353772
from exact source b6d42642301c7b35da2632c9ef64db61715f2f21dcb24547b9b34aaba98dba52.

All 63 frozen ordinary fixtures matched locked CPU PyTorch and the
native candidate, i.e. 3,222 I64 words per logical and physical order
(31 order discriminants), including individual column digests, 26 K0
and six K1, and exact metadata, owner/scratch and both drop orders.
The old image refused all 60 valid new dtype roots with status7 and
unchanged owner, and still passed the three retained Bool controls.
Fresh authority is verified for empty results too, without using data
pointer identity.

Maximum input N=270 and output K=68 in this qualification. No claim
of new cap test, OOM, fault, malformed ownership, higher rank, views,
tuple/out, default or complete PyTorch parity. The first native
preflight below the memory floor did not execute; the guarded rerun
used the original threshold.

All 2,555 historical successful ordinary frames across the four
previous rank-1/2/3 campaigns replayed byte for byte, including
148 successful nonzero cases. No historical error path or
unsupported case was rerun.
