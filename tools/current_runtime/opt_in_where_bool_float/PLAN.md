# Explicit Bool + F16/BF16/F32/F64 Tensor pairs (source only)

On #66 86bb2bfee8913d1337911db43cfab90a0cab95a4, retain the exact existing
pair-mode scaffold and Bool/I64 route. The failed same-tag check may now
select the other operand tag exclusively among public F32, I64, F64, F16
and BF16 when exactly one operand is Bool and condition is Bool.
Each operand's own widths and ordinary-live-owner guards are original.
The target is the exact non-Bool tag and width; no global type inference,
cast operations, float arithmetic, dtype folding, or wire changes.
The selected raw word of the target dtype is stored without conversion;
only the selected Boolean branch canonical-checks 0/1 then maps it to
positive-zero or true bits. The local true table is exact:
I64 1; F16 0x3c00; BF16 0x3f80; F32 0x3f800000; F64 0x3ff0000000000000.
Unknown table targets do not publish. Existing same-dtype paths and private
tag 3 denial are unchanged. Shape join, scratch save/restore, capacity,
fresh owner allocation, no-view/no-autograd/status/rollback remain pinned.

Physical campaign after separate authorization:
* A separate locked wheel proves dtype and bytes of exact typed Tensor
  operands, never a wrapped Python number. For each dtype, both value
  operand orders and both true/false condition branches, rank-zero/rank-zero
  and each rank-zero role versus a higher-rank operand (including rank-zero
  float with dimensioned Bool and its inverse). A mask grows a coordinate
  and value shapes independently; empty both orientations. Alternate masks
  and all-target branch cases preserve both signed zeros and positive/negative
  subnormals, finite extrema, infinities and distinct signed qNaN/sNaN
  payloads. Ensure at least one selected Bool-False -> +0, Bool-True -> exact
  encoded +1, and selected target numeric negative zero passes through.
* Exact predecessor 129 status7 on actual Bool/float roots. No-Torch
  physical candidate + independent Cartesian reader with separately decoded
  widths and literal sign-bit/one-word witnesses. Exact fresh public Tensor,
  Storage and Version IDs, Version0, unchanged source bytes/metadata;
  input/result release orders, weak expiry; request-local result; real
  accepted pair then invalid suffix rollback. Keep true view/positive-offset,
  stale, padding, shape, dtype and capacity controls. Additional illegal
  pairs include F32/I64, F16/I64, F16/BF16 and F32/F64 in both orders.
* The previous Bool/I64 campaign included four newly admitted refusals:
  bf16-bool-forward/reverse and f64-bool-forward/reverse. Capture exact
  original status7 and complete unchanged owners, and compare the
  unmodified original fixture's input words, geometry and order against
  separate successor positives (its scalar condition true, data shape
  [2,4], low words 0..7 and alternating Bool). For replay remove exactly
  the four now-admitted records, leaving their otherwise-unused setup/drop
  unchanged so all other request/response frame bytes compare exactly.
  No retroactive relabel or invented rollback. Prior #65/#66 shapes and
  Bool/I64 physical positives should replay otherwise byte-identically.
