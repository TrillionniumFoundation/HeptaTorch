# F16 dimensioned mixed where: explicit locked FBGEMM source-tail profile

Exact #66 2ff996a1b2ee5168e15c82f5e708aace741ec731. Only existing pair
admission and local conversion are extended for unordered {F16,F32} and
{F16,BF16}, both with positive-rank ordinary CPU owned tensor values and
Bool condition. The original F16 operand count must be strictly <32768,
regardless of output and mask selection. No direct views, non-contiguous,
negative offsets, autograd, wrapped Python values, rank-zero operands,
different wheel/compatibility mode, or default dispatch are admitted.
Both source-width guards coexist; output F32 width4. Existing raw F32 and
BF16 shift16 copy paths, private type denial, shape caps/bounds and prior
pair/cast helpers remain unchanged.

The compatibility target is the exact pinned official eager CPU wheel,
with accelerated FBGEMM cast prior to broadcasting. The pinned FBGEMM
revision is bf6dce360a4fe133bc779e2fd036277678509f95; full 16 then 8 lanes
use accelerated bit-preserving widening; remaining scalar CPU half
conversions canonicalize signed quiet/signaling NaNs to +0x7fffffff.
In the one-chunk contiguous qualified scope, cutoff is floor(N/8)*8 where
N is the **original** F16 operand element count. The shape planner computes the
actual chosen source index checked by existing coordinate mapping/raw
read. The new local handler carries that index and handle (not output
index), rechecks index < original count and cap, and uses the cutoff only
on selected F16 raw words. Else use unchanged existing widen_f16; this
keeps non-NaNs (including signed zero/subnormals/inf) and full-lane selected
NaN sign/payload as before. No change to global widening or scalar code.

Pre-implementation locked held-out profile: [N,1] values broadcast against
[1,8] repeated rows; 55 qualifying matched records across both orders
and both pairs, 65,695 direct .to(float32) and 2,102,240 where words with
zero mismatches. N=32768 and 32769 diagnostic observations did not grant
admission. Independent 1..65 profile originally disagreed with unconditional
widen_f16 on exactly 247 selected F16 NaNs in each pair and remains archived.

Later physical checks: two cold, separate no-Torch and independent
coordinate reader. Include N=7/8/9, 15/16/17, near 32767, actual source cap
refusal at >=32768 with result shape otherwise supported, same-rank
row-major and repeated source distinguish source vs output cutoff, both
orders/pairs, signed ±q/sNaNs, state gates, fresh metadata/Version0, two
release orders and weak expiry, transaction suffix/refusal rollback.
Old official controlled profile remains separate from native readiness.
Archivally prove only genuine intended retired source/output pair cases:
Bool-float's two F16/BF16 controls, F32/F64 cohort's two F16/BF16 controls,
BF16/F32 cohort's two F16/F32 and two F16/BF16 controls. Preserve existing
physical unadmitted rank0 and other pairs. No native compile executed in
this draft.
