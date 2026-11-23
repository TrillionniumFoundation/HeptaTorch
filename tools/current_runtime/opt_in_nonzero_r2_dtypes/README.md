# Ordinary rank-2 nonzero: all six public value dtypes

Exact parent #69 2b5b2fcd8b7937d5441317da1becf9cbef360a1b, tree
e4ca3d95e47728428380adf5c3f4710d54ae62f1. The patch extends the existing
opt-in rank-2 Bool path to I64, F32, F64, F16 and BF16, for fresh ordinary
owned contiguous zero-offset CPU inputs with no grad. Opcode135, input
and output caps, scratch, rank-1 path, default and the Bool semantics
stay unchanged. The one changed body uses the existing exact dtype
widths and shared truth on raw IEEE words and all I64 bits. Both signed
floating zeros are false; signed subnormals, infinities and q/sNaNs are
true. No cast, promotion, numeric conversion or change to global helpers.

Two cold builds produced identical ELF
c971732c76049f2b4d760177085f81f73ba602c7880e075625fb8a15bac06c54
from combined HIA c76588477d02e69e96fd86c8677430602c0c0f6afae198c07a0e60e906360155.

Single-thread locked CPU Torch and the native candidate agree on all
52 small ordinary cases (50 new and two retained Bool). Each logical
and physical stream has 1,036 I64 words; 26 cases distinguish the
orders. Fresh owning I64 results have [K,2], element strides [1,K],
storage 16*K, zero offset, Version0, and row-major contiguity iff K≤1.
Logical coordinates are row lexicographic, while the backing storage
contains columns. The independent raw reader verifies both and checks
the original owner bytes and distinct output Storage/Version authority
without comparing empty pointers. All 50 new, actually constructed
ordinary inputs hit exact predecessor status7 without owner changes;
the two Bool controls still succeeded. N≤323 and K≤29 for the new
matrix; no fresh cap, OOM, fault, malformed-ownership, rank>2, view,
tuple/out, coexistence or default-suite claim.

Both exact positive historical campaigns replayed byte-for-byte:
1,197 successful prior rank-1 frames (50 nonzero cases across six
dtypes) plus 392 prior rank-2 Bool frames (23 nonzero cases), 1,589
frames total. No unsupported or fault cases were rerun.
