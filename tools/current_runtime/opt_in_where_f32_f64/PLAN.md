# Explicit dimensioned F32/F64 value pair (source-only)

Exact #66 9bc18fcaff165f3a5fd97258b6392935c1e64bcb. Admit only a Bool mask
plus the unordered public {F32 tag0, F64 tag4} pair, with both value ranks
strictly positive. Scalar means actual rank-zero Tensor; leave it refused
for this pair because wrapped/zero-dimensional type resolution is not this
contract. A rank-zero Bool mask is allowed. Reuse each operand's original
own widths 4/8, bounded ordinary CPU owner/geometry, layout/no-AD safety,
three-way shape core and pinned typed allocation. F64 result uses raw width
8. For the selected F64 source retain unmodified bits; only if selected
F32 is being stored into the F64 result call existing widen32 exactly,
imported by audited signature from unit04 with no change to its source.
No global casts, unrelated mixed dtype/promotion, wire, or default routing
changes. Known Bool pairs and same-dtype modes keep existing bodies except
for a new failed-admission edge and the local conversion dispatch.

## Required release boundary (not yet run)

1. **Official wheel first.** Locked CPU PyTorch, exact rank-positive
   value tensors and rank0 condition allowed; inspect target raw bits, not
   Python floating scalars. Both operand orders, min/subnormal, signed zero,
   finite extrema and signed infinity. Compare same-shape and repeated/
   broadcast inputs; use n=1,2,3,4,7,8,9,15,16,17,31,32,33,63,64,65 and
   vary the selected F32/F64 coordinate before/after potential vector
   boundaries. Verify signed F32 distinct qNaN/sNaN payloads against
   independent integer widening (including IEEE quiet bit); independently
   inspect selected F64 qNaN/sNaN raw payload unchanged. Exercise differing
   lengths and rank2/rank4 nontrivial geometry. Never infer sNaN bit policy
   from ordinary casts or double round-trip.
2. If any selected F32 NaN differs, do not qualify the wide projection as
   general IEEE parity. Keep original mismatch fixture and exact available
   predecessor/candidate observations. Make a separately hash-pinned
   fallback that explicitly refuses selected F32 NaNs using exponent
   0xff and fraction nonzero, allowing only proven classes (and similarly
   reject only the problematic F64-selected NaN classes if measured).
   This must be local to proven rank-positive F32/F64 pair; the broad
   already-qualified Bool/F64 raw path must retain every word, including
   signaling NaNs. F64-selected gating, if needed, must check the actual
   operand pair, not only result dtype. Do not silently drop an unproven
   or mismatched row or infer classes from a header.
3. True old status7 for actual valid pair roots; two independent cold
   final builds, no-Torch native then pure coordinate oracle with exact
   mixed widths, fresh IDs/Version0, input bytes and metadata unchanged,
   both lifetime orders, compound rollback and retained proper caps.
   Actual mixed rank0 both roles/orders still status7; conflicting shape,
   genuine viewed/positive-offset/grad/stale/padding, and F32/I64,
   F16/I64, F16/BF16 or other unsupported pairs remain status7.
4. Original Bool-float campaign f32-f64-forward/reverse contain actual
   now-admitted dimensioned controls. Preserve original refusal/status/
   complete owner, match original scalar Bool-mask/shape [2,4]/values
   0..7 to explicit successor cases, and remove only those exact frames
   from retained replay. Prior cold shape/native raw suites otherwise
   byte-identical. No full/hosted/default parity claim.
