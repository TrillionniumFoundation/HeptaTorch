# BF16/F32 rank-positive where: scope and proof plan

Exact predecessor #66 ad37a693f8931f461f29ef1b291010b55f26647a.
Only the existing failed-pair fallback admits {public BF16 tag6, F32 tag0}
with Bool mask and both value ranks >0, in either order. The existing
rank gate checks both, and mixed roots retains each operand's width 2/4
and ordinary live storage/version/raw bounds. Result is fresh public F32
using existing broadcast allocation and exact owned caps. Selected F32
goes through the original same-type raw store. Only selected BF16 with
F32 result uses the existing half_widen_bf16 exact unit04 shift16 helper
imported to unit12 (unchanged definition). This retains sign, payload,
and BF16 signaling state. Existing F32/F64 and Bool pairs, representations,
wire, status/rollback, and default mode are unchanged. No F16 pair admitted,
no rank-zero mixed value admission. Rank-zero condition can still be Bool.

Oracle gate completed as separate locked CPU Torch profiling, before any
candidate compile/native test: 264 BF16/F32 rows and 8,660 words across both
operand orders, two mask phases for each length 1..65 and four complete
selected special-value banks. No mismatch. Companion exact profile found
247 selected F16 NaN mismatches in each of F16/F32 and F16/BF16: 25 quiet,
222 signaling; every mismatch observed 0x7fffffff only at incomplete
eight-word tail positions, versus payload/sign-preserving widen_f16
in all tested complete groups. This nonportable distinction is not
implemented or qualified here. Keep full original profile and tails.

After explicit sole physical authorization: materialize this pinned split,
two separate cold builds, take the 264 immutable BF16/F32 official fixture
rows into a no-Torch process, and gather separate locked geometry/empty
fixtures if desired (broadcast rank2/rank4, zero-last versus zero-prefix,
actual owner/root/stale/refusal traces). Record exact predecessor status7
for those newly admitted. Raw independent coordinate readback covers both
source widths, BF16 signal sign+payload, selected F32 identity, both
release orders and distinct output IDs, initial Version0, same-frame local
reference, and valid prefix/invalid suffix rollback. Actual mixed-rank0
both roles/orders and F16/F32 or F16/BF16 of both orders stay refused.
All prior qualified byte streams replay unchanged unless a genuinely
intended archived BF16/F32 refusal is found and separately documented.
