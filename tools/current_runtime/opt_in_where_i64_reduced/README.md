# Native CPU where.self: dimensioned I64/F16 and I64/BF16

Adds only these two unordered public value-type pairs, with a Bool mask,
positive value ranks and the existing ordinary contiguous no-grad CPU
root guards. The two source widths remain independent (8 and 2 bytes).
The result is fresh and uses the chosen pair's reduced type and width 2.
Already-reduced selected words pass through verbatim. Selected I64 is
converted with the existing exact I64→F32 bit helper, then with the
existing F16 or BF16 rounding helper. No other runtime definitions,
allocation/broadcast geometry, old mixed rank-zero restrictions, default
or previous F16→F32 FBGEMM profile changed.

The preregistered independent integer model matched locked CPU PyTorch
for 2,940 direct casts and 14,352 actual where words, over 134 fixtures.
BF16 distinguished the F32 intermediate from single reduced rounding
at 1,696 outputs and from a binary64 intermediate at 1,504. Signed F16
overflow is max finite at magnitude 65519 and infinity at 65520.
Selected reduced quiet/signaling NaN payloads and signed zero stayed
raw-identical. Two cold builds reproduced the same ELF. After adding six
frozen historical controls, the official, no-Torch native and offline
owner/readback checks agreed on 140 cases and 14,400 output words.
Input payloads and owner metadata stayed unchanged, with fresh Tensor,
Storage and Version identities, zero initial Version, both destruction
orders, weak expiry and same-frame handles.

The published predecessor refused all 140 exact setups at status 7
with byte-identical full owners. The candidate's 2,753 frames include
171 exact rollback checks, twelve mixed rank-zero guards, eight genuine
views (positive-offset I64 and zero-offset 16-bit views), four compatible
overcapacity refusals, geometry, padding, stale handles and invalid-suffix
rollback. Eight empties passed. All 24,100 other prior frames replayed
byte-identically with 1,616 rejected-frame owner checks, including the
original F32 allocation-pressure and recovery sequence under its exact
archived absolute process-local cap. Only 20 proven old negatives across
seven suites were removed from the success replay and preserved in raw
custody with six exact replacement witnesses. There are
26,853 checked successor frames and 1,787 rejected-frame owner checks.
This is a scoped opt-in candidate, not general or default PyTorch support.

Run `python3 -I -B prepare_i64_reduced.py EXACT_PARENT EXACT_BUILD_PY NEW_SOURCE`.
Use the included source manifest for a second pinned cold build. Expand
the compressed fixture using `expand_fixture.py PACKED FIXTURE` (fixture
basenames are pinned). Run `locked_profile.py PINNED_HELPERS ORACLE_LOCK
INSTALL_RECEIPT prereg.json OUT` and `oracle_suite.py PINNED_HELPERS
ORACLE_LOCK INSTALL_RECEIPT suite.json OUT` only in the locked CPU wheel.
Run `native.py PINNED_HELPERS PROTOCOL_PY ELF suite.json OUT baseline|candidate`
in isolated no-Torch Python, then `pure.py suite.json BASELINE_CAPTURE
CANDIDATE_CAPTURE`. To replay earlier accepted campaigns, use
`retain.py PINNED_HELPERS PROTOCOL_PY ELF PRIOR_I64_F32_ROOT ORIGINAL_WHERE_CAPTURE
suite.json NEW_CAPTURE`. To rebuild the historical additions from their
raw archives, use `derive_suite.py CURRENT_FIXTURES PRIOR_I64_F32_ROOT
HALF_F64_ROOT I64_F64_ROOT NEW_SUITE`.

Pressure replay identifies the unique archived request with status 3,
tests exact pre/post owners, and independently maps lead/failure/recovery
477/478/479 in the frozen predecessor stream. Those are 478/479/480 in
the original pre-filter stream. The original process-local absolute cap
17,321,984 bytes is restored before the recovery request; the first
unrebased harness attempt is retained separately.
