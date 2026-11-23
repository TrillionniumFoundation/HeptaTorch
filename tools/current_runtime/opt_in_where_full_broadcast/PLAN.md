# Full three-operand ordinary CPU shape fallback: source only

Exact #66 predecessor `c2fac4373c8047b7a9a6acf5fadf2f3aeffa2c8b`,
aggregate source `30cd4e10394bc82d7197ba5b1a707ce6d7806e05c0674b8e171addd6de9d34da`.
All already admitted equal, rank-zero condition, trailing and fixed-value-shape
right-aligned mask routes retain their code and normal transaction. Only
`where_self_tensor`'s preexisting final shape rejection calls the fallback.
The two value dtypes must already match exactly (all six public dtypes),
the mask must be Bool, and all inputs already passed live ordinary view-free,
AD-free, zero-offset, contiguous storage admission and byte bounds.

Each input is aligned to max rank ≤4; missing leading axes are one. Each axis
joins equal extents or a one. In particular zero joins zero or one, not a
positive extent bigger than one. Every axis is proved before count reduction,
scratch, allocation or reads. Each planned extent ≤262144, so intermediate
saturating products are ≤262145*262144 and cannot wrap. The element-count cap
is 262144; a valid larger nonempty join is a capacity refusal (status3),
not a PyTorch shape incompatibility. A late zero resets the saturated
product to zero. Rank 0 has all inactive plan extents equal to one and
count one; return values distinguish Tensor identity/status, not rank.
Incompatibility is still status7. The existing typed allocator checks
nonempty products at 2^24 and recognizes empty geometry before multiplication;
the stricter new count cap makes the nonempty allocator limit redundant here.
Empty stride construction treats zero like one. Any published stride has
at most three factors ≤2^18 for rank4, hence at most 2^54.

`where_full_alloc` saves exact existing shape scratch qwords 128/136/144/152
in SSA, stages only from the already-proved four SSA dimensions, invokes
`typed_descriptor(s,rank,128,0,1,dtype)`, then restores all four via exact
unconditional `putw` before testing the return. This extends the already
established bshape→typed_descriptor scratch contract, introduces no other
owner header offset, and preserves first error semantics. The result is
validated from its newly copied descriptor (rank, count, dtype, every
active dimension and existing ordinary/raw byte owner proof), not from
scratch. That descriptor is immutable thereafter. The original checked
index mapper receives the fresh output descriptor and independently maps
mask then chosen value. It validates each positive divisor and coordinate
bound; no data iteration at all for an empty. Selected words are copied
with exact byte width and no conversion.

Physical discriminants for after authorization:
* Six public value tags. Shape fixtures should include distinct axes from
  all three sources ([2,1,1], [1,3,1], [1,1,4] → [2,3,4]); value-only
  expansion with a scalar mask; a condition that grows both scalar values;
  a higher-rank condition [1,3,4] versus [3,4]; mixed missing/interior axes;
  outcomes that distinguish mask, x and y geometry.
* Two empty shapes (zero in different axes); scalar/ones versus zero
  accepted; zero versus non-singleton >1 rejected even when another axis
  is already zero. A supported empty with large prefix such as
  [512,1,2,0] + [1,512,1,1] should not be rejected by a spurious prefix cap.
  If tested, distinguish PyTorch exact bytes (zero words) from downstream
  resource policy. Use boundary [512,1]×[1,512] -> 262144 and overcap
  [513,1]×[1,512] -> 262656 with actual small roots; require actual status3
  and no allocation for overcap; keep historical baseline status7 separately.
* Distinct owners/version0; no source mutation; direct raw metadata from
  independent Cartesian proof; both lifetimes and weak expiry; actual
  full-owner data copy and within-frame valid where + invalid suffix rollback;
  one compound binary op before where creates real preexisting nonzero
  scratch, output plus post-where capture must match control scratch
  bytes (private snapshot offsets 128..159); no hand-edited state.
* Archived reclassification only where legitimately changed: prior
  scalar/trailing/general `broadcast-values` of [1,4] against [3,4];
  prior general `higher-mask-rank` of [1,3,4] vs [3,4]. Keep original
  complete refusals and match source words/geometry against new proofs,
  then compare retained byte streams after excluding only exact retired
  records. Wrong incompatible x/y (same numel), leading mismatch against
  non-singleton, true offset views, marked grad, padding, stale both
  value handles, and mixed dtypes stay refused. Same-shape/rank-zero/
  trailing/general mask and #65 matrices otherwise unchanged.
* Extract oracle separately in official locked CPU Torch; two cold builds
  once assigned; no-Torch observer; independent raw Cartesian reader.
