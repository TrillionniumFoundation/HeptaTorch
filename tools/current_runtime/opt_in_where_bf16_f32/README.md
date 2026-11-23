# Native CPU where.self: explicit dimensioned BF16/F32 values

Allows exactly BF16 plus F32 value tensors in either order when both
have positive rank and condition is Bool. The individual widths stay
2/4 under existing ordinary contiguous, live, no-grad CPU owner and
shape/cap checks. Selected BF16 bits shift left 16 exactly in the
unchanged helper, preserving sign, payload and signaling state; chosen
F32 passes through unchanged. The result is fresh public F32. Mixed
rank-zero values, F16 mixed pairs, other promotion, arbitrary views,
autograd, out=, default routing and hosted parity are not added.

Two independent cold builds produced identical ELFs. Official locked
Torch supplied 264 rank-one rows covering both orders and both phases at
every length 1–65 (8,660 words) plus six separate rank-2/3/4 shape rows
(96 words, four empties). A no-Torch native capture and independent
Cartesian reader matched every word: 4,378 BF16 widenings and 4,378
unchanged F32 selections. Both signs of zero, subnormal, infinity and
quiet/signaling NaN payloads were covered. Fresh Tensor/Storage/Version,
zero initial Version, unchanged input data/metadata, both release orders,
weak expiry and request-local output use were observed.

The exact predecessor refused 270 actual new setups with status 7 and no
owner mutation (2,700 frames, reported separately). The new 5,206
successor frames include 296 full rollback controls: six actual mixed
rank-zero denials, true offset F32 and zero-offset genuine BF16 views,
other mixed pairs, capacity, geometry, stale/padding/grad and a valid call
followed by an invalid suffix. All previous raw request/response bytes
replayed exactly, with no newly retired control. Total successor 16,162
frames / 1,149 independently checked rollback controls.

The standalone locked companion profile also deliberately examined
F16/F32 and F16/BF16. Each showed 247 selected F16 NaN mismatches to the
unchanged widening model: 25 quiet and 222 signaling, all in incomplete
eight-word tails and all canonicalized to positive 0x7fffffff by Torch.
The 1,619 selected F16 NaNs in tested complete groups matched. Neither
F16 pair is admitted; this is a scoped measured blocker, not a claimed
portable lane behavior. The exact original profile and proofs remain
available separately.

Verify the split/inverse with isolated Python
`prepare_bf16_f32.py EXACT_PARENT EXACT_BUILD_PY`. An optional final
new directory materializes the source; the accompanying manifest pins
the second cold build. The published compressed fixture expands via
`expand_fixture.py bf16-f32.fixture.json.zlib.b85 FIXTURE`.
`profile.py`, `extract_profile.py` and `oracle_geometry.py` reconstruct
locked reference inputs separately. Then run
`native_bf16_f32.py PINNED_HELPERS PROTOCOL_PY ELF FIXTURE OUT
baseline|candidate` separately without Torch and
`pure_bf16_f32.py FIXTURE BASELINE_CAPTURE CANDIDATE_CAPTURE`
as an independent reader.
