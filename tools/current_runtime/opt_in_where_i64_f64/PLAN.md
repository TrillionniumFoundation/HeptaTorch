# Dimensioned I64/F64 where value pair: source only

Exact #66 2d1d1371c99864394cd4dc1d0dc10528106a446e. Admit only Bool
condition and exact unordered public I64/F64 in either order with both
value ranks >0. Independently validate both original operand widths of
8 and existing ordinary CPU layout/raw bounds/no-grad/lifetime and
shape/cap guards; use the original three-way shape planner, scratch
restore and fresh typed output (public F64 tag 4). On selected F64 the
established raw same-tag path preserves every bit. Only selected I64
with target public F64 enters existing exact integer_i64_to_f64, imported
by its pinned unchanged signature from unit04. It handles signed two's
complement, ties-to-even and mantissa overflow/renormalization.
No global rules/casts, other combinations, mixed rank-zero values,
views, Python scalar, out=, autograd or default activation.

Future physical qualification only after authorization:
- Both actual rank-positive orders in locked official CPU Torch, retain
  original typed words. Build an independent integer quantizer with step
  ULP and quotient/remainder (do not use Python float or HIA source as
  oracle); independently validate exact F64 words before native. Include
  signed near 2^53 ±{0,1,2,3}, signed near 2^62 ±{511,512,513,1535,1536,1537},
  both even and odd halfway directions, min/max I64, mantissa carry across
  exponent, and exact representables. Distinguish x86 vector/tail by
  selecting step-adjacent values with both orders, and rank2/4 broadcast.
  F64 selected raw signed zeros, subnormals, infinities and signed q/sNaN
  payloads remain exact identity, and empty result checks each order.
- Predecessor actual typed roots and status7 whole-owner rollback.
  Two cold builds, no-Torch native then independent raw owner + Cartesian
  checker. Fresh Tensor, Storage, Version0, both release orders, weak
  expiry and same-frame result handle use; valid then invalid suffix
  rollback; real offset/grad/rank0/cap/shape/stale/padding controls.
  Other pairs (especially I64/F32, I64/F16/BF16 and F64/F16/BF16) maintain
  existing status and exact refusal. If any genuine historical I64/F64
  status7 case exists, archive unchanged and retire only its replay frame;
  prior visible guards are I64/F32 and I64/F16, not this pair.
