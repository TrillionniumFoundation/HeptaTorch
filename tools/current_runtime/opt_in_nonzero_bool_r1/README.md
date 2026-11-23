# Opt-in native rank1 Bool nonzero on exact main

This is a standalone main-based candidate for aten.nonzero.default:
one fresh I64 tensor, not a tuple/out/static operator. It admits only
ordinary owned zero-offset contiguous CPU rank1 Bool, including true
empty input, canonical bytes and N up to 262,144. No view, rank0,
rank2+, dtype widening, new default or whole-runtime qualification.

Private candidate opcode135 is validated against the linked local
wire/source census and coordinating root's authenticated checked heads.
Two exact cold builds produced the same ELF
95aadc064a5a16a188988b699b81dda287443b0e7172fa163dde15db3857c6c7,
combined source 42cadd9e3f1c7aea5ea391ef32b020efecbc972030a9df3c44a1a689c9ff9f91.

The result has shape [K,1], element strides [1,K], Version0, storage
8*K and no view lineage. During ordinary private construction only,
the newly allocated result's second element-stride slot is assigned K
after verifying its shape/storage/offset and distinct Version/Storage
identities. This is not a global allocator layout flag. The source's
true K0 never enters the payload/address paths and has [1,0] metadata;
zero bytes do not imply shared Storage identity. Sorted coordinates
are the selected input indices as little-endian I64 (K values).

In the locked CPU oracle, 19 rows and six measured 2/4-thread repeat
trials preserve raw ordering, bytes, Version0 and the distinct result
layout. The two N=262145 cases are theoretical out-of-scope PyTorch
values: native old/candidate opcode97 refuses rank1 construction with
status3; neither case was issued to native opcode135. No physical
nonzero-cap or OOM claim and no alternate malformed owner. Higher rank
PyTorch control exhibits distinct logical and physical coordinate
ordering, not implemented here.

The native candidate covers 17 successful rows (263,307 words), four
K0 results, both destruction orders, 17 weak lifetimes, 9 non-weak
controls and exact seeded same-frame response metadata/raw read.
The exact main world physically refused all 17 in-scope opcode135
commands with status4 without owner mutation. A separate pure reader
verifies raw storage, metadata, Version/offset, actual live owner and
unchanged input snapshots.

All 385 comparable exact-main request/response pairs stayed unchanged:
all 290 of the previously archived baseline campaign and all 95 of
the new compatible setup/read/drop frames. The 17 old status4 opcode135
rows alone are intended admissions, not part of that comparator.
This is not the entire hosted default gate.
