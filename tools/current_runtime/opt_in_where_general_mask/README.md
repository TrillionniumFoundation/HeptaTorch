# Native where.self: right-aligned Bool condition on CPU

Allows a condition whose rank does not exceed the two identical value
shapes. Its omitted leading axes act as ones, and each aligned extent
must equal the corresponding value extent or be exactly one. Rank is
bounded to four and each fallback extent and count to 262,144. Existing
equal, scalar and trailing plans are retained. Both values still have
the same public dtype and equal shapes. All tensors must be ordinary,
zero-offset, no-grad, contiguous, and live in the original owner arena.
No value broadcasting, dtype promotion, views, out=, autograd, default
activation, or hosted aggregate claim.

The fallback proves every aligned axis and exact counts before allocating
even an empty result. Its prefix products saturate at the limit plus one
to preclude overflow while a later zero extent still yields exact zero.
Coordinates are traversed only for nonempty values; every consumed value
extent is checked strictly positive before div/rem. Mask indices and
strides are checked against the real mask count before raw reads, with
the original bounded dtype-byte store unchanged. Result shape stays the
value shape.

Two independent cold builds reproduced the same ELF. Official locked
CPU Torch and independently reconstructed Cartesian-coordinate readback
of a no-Torch native session agreed on 12 rows and 216 bit-exact words
across six dtypes, including lower-rank masks, interior/multiple singleton
axes, and distinct zero-last/zero-interior cases. The exact predecessor
refused every actual new case with status 7 and unchanged complete owner
(118 frames). The successor verified 32 complete rollbacks over 290
frames, unchanged inputs and new Tensor/Storage/Version with initial
Version zero, both release orders, weak expiry and request-local result
reuse. Higher-rank expansion, misaligned axes even with zero output,
mixed value dtypes and actual positive-offset input views still refuse.

Six archived status-7 controls became admissible and map to four separately
verified successor witnesses using the same input words and geometries.
All other raw request/response frames from the same-shape, scalar, and
trailing stages are unchanged after exclusion only of those six records.
All four #65 retained matrices remain byte-identical. In total, the
successor suites verified 3,969 frames and 431 complete rollback controls.
The prior actual allocation refusal and recovery replay is retained; no
new general-broadcast allocation pressure is claimed.

Use isolated Python to run `prepare_general.py EXACT_PARENT EXACT_BUILD_PY`
for an in-memory, exactly reversible projection. Its optional final new
output directory materializes the source. The pinned manifest records
the second cold build pins. Run the locked wheel only for
`oracle_general.py PINNED_HELPERS ORACLE_LOCK INSTALL_REPORT FIXTURE`.
Run `native_general.py PINNED_HELPERS PROTOCOL_PY ELF FIXTURE OUT
baseline|candidate` on each exact image without Torch. Then run
`pure_general.py FIXTURE BASELINE_CAPTURE CANDIDATE_CAPTURE` in an
independent process, and use `verify_retired.py ORIGINAL_TRAILING
ORIGINAL_SCALAR ORIGINAL_SAME_SHAPE RETAINED_ROOT NEW_CAPTURE FIXTURE`.
Retained suites update only the executable pin and the six documented
expected-refusal requests; other fixture and pure reader bytes are unchanged.
