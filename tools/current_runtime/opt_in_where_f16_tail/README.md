# Native CPU where.self: two F16 pairs under a locked FBGEMM profile

Adds exactly F16/F32 and F16/BF16 value tensors in either order when both
values have positive rank, use ordinary contiguous no-grad CPU ownership,
and the original F16 operand has fewer than 32,768 elements. Condition is
Bool; destination is fresh F32. No default dispatch, global casting change,
other CPU/build profile, multichunk cast, larger original F16 operand,
rank-zero mixed value, view input, autograd or out= is admitted.

The compatibility target is the locked official CPU wheel with its
accelerated eager contiguous FBGEMM cast (pinned FBGEMM
bf6dce360a4fe133bc779e2fd036277678509f95). For each chosen F16 value,
the planner's mapped **source** index and original F16 count determine the
profile; flattened output position and size are never used for this.
Within complete eight-word source groups the unchanged widen_f16 helper
preserves sign/payload and quiets signaling NaNs; the scalar tail of the
locked FBGEMM cast canonicalizes any selected quiet/signaling F16 NaN to
positive 0x7fffffff. All selected non-NaNs still use unchanged widening.
Selected BF16 and F32 retain the previously exercised exact raw behavior.

Separate official held-out profiling covered 55 qualifying direct cast
and where records, 65,695 cast words and 2,102,240 paired result words with
zero mismatches. Results beyond the source count cap were recorded only
as diagnostics. Two cold builds then reproduced the exact ELF. No-Torch
native capture and independent Cartesian readback matched all 40 official
fixtures and 264,616 output words, with 24 matching held-out rows, source
lengths 7/8/9 and 15/16/17, four 32767-element source cases, rank-four
broadcast and four empties. The raw audit located 284 outputs that naive
output-index tail classification would get wrong. It verified quiet and
signaling NaNs of both signs in complete groups and the actual source tail,
fresh Tensor/Storage/Version, zero Version, unchanged input owner records,
both release orders, weak expiry and request-local result use.

The predecessor refused all 40 new actual calls with status 7 and no
owner mutation (400 frames, separate). The successor's 870 new frames
included 80 full rollback controls, with eight 32768-source profile
refusals even when that operand was never selected, 12 rank-zero mixed
value denials, actual view and shape guards, result-capacity refusal,
and valid call followed by invalid suffix. Eight earlier F16 pair
denials remain exact archived old witnesses; four distinct successor
cases prove the same words and geometry. All other earlier streams
match byte-for-byte after omitting those eight now-admissible controls.
Total successor: 17,024 frames and 1,221 exact owner rollback checks.

Reproduce projection: isolated Python
`prepare_f16_tail.py EXACT_PARENT EXACT_BUILD_PY`; optional final new
directory materializes the checked split. Use included pin for second
cold build. Replay `heldout_profile.py` and
`oracle_native_fixture.py` in locked official CPU Torch only, or
recover exact references with `expand_fixture.py fixture|heldout INPUT OUT`.
Use `native_f16_tail.py PINNED_HELPERS PROTOCOL_PY ELF FIXTURE OUT
baseline|candidate` separately without Torch, then
`pure_f16_tail.py FIXTURE BASELINE_CAPTURE CANDIDATE_CAPTURE`.
`verify_retired.py OLD_BF16_F32 OLD_WIDE OLD_BOOL_FLOAT RETAINED_ROOT
NEW_CAPTURE FIXTURE` verifies controlled archive comparison.
