# where.self explicit Bool/I64 value pair only (source projection)

Parent PR #66 95347ab4031551a386408dadb171d1d6a5f2955e, exact source
a99ffaa8bcfee978016cf6a379a04b1e4ddf9540f4d84f79769d33c520340bf5.
Only when the mask is Bool and the values are precisely one public Bool and
one public I64 Tensor (in either order), bypass the existing same-dtype
path and run the proven cap-bounded three-way shape core in explicit pair
mode. Typed admission checks the two input widths separately (1 and 8) and
the existing ordinary CPU/layout/owner/view/no-AD invariants. Allocate tag2
with width8, same wire, fresh Version/Storage/Tensor; shape, bounds and
scratch unchanged. The core obtains the selected value through the existing
raw checked read. If pair-mode selects I64 it retains every bit; if it selects
Bool it rejects noncanonical >1 (after live bounds/read guard) and explicitly
maps true/false to raw I64 words 1/0. Pair admission never enters the old
same-dtype shortcuts. Existing full fallback calls the shared core with
pair mode false under its original signature; same-dtype fast routes,
unknown commands, global promotion, default graph and all other mixed dtypes
are unchanged. Static local binding and branch arity checks are included.

Physical assessment (not yet run):
* Locked CPU Torch actual dtype and bit oracle, both orders; both typed
  rank-zero, one rank-zero Boolean with non-scalar I64, one rank-zero I64
  with non-scalar Boolean, and each source also as the left-hand value.
  Use true/false scalar condition, alternating vector/actual ternary-broadcast
  cases and both source branches selected. Include I64 min/max, signed
  values adjacent to min/max, and >2^53 words; no conversion to float.
  Include two mixed empty cases from distinct geometry. Determine actual
  pinned Torch type for these rank rules before any claim.
* Exact old image can create same Bool/I64 roots but refuses new 129 with
  status7 and complete owner bytes; native candidate standalone physical
  fields plus independent Cartesian raw reconstruction (decode input words
  separately w=1/8, exact selected word, output w=8). Distinct IDs, initial
  Version0, unchanged input raw bytes and fields, independent weak expiry and
  both lifetime orders; local result use and valid call then invalid suffix
  rollback. If recording noncanonical Bool, accept only protocol-created
  values, not edited owner state; if standard setup rejects, record as
  setup boundary, not a runtime pair negative.
* F32/I64, I64/F32, F16/I64, BF16/Bool, F64/Bool and private/unsupported types
  stay refused; later same shape vs different shape mixed denial test with
  actual input objects. True positive-offset viewed Bool and I64 inputs,
  stale, padding, rank/shape/resource limits and condition dtype remain
  unchanged. Check exact packet and raw owner snapshots.
* Two cold builds; retain unchanged raw traces for all previous known
  shape cohorts and #65 helpers. No prior mixed Bool/I64 129 negative
  found in the existing physical scripts (their mixed negative is
  F32/I64); if a genuine retained changed witness is encountered, archive
  and document that exact control before any replacement. No hosted claim.
