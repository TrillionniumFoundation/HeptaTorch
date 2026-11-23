# Native CPU where.self: dimensioned I64/F64 values

Adds only the public I64/F64 pair in both orders when both value tensors
have positive rank, the condition is Bool, and all roots pass the original
ordinary contiguous/no-grad CPU and owner checks. Both inputs independently
use width 8. The original shape planner, typed allocator and transaction
are unchanged. The output is fresh public F64; chosen F64 bits pass through
directly, and only chosen I64 calls the unchanged integer-to-F64 converter.
No global promotion or default routing is enabled. Rank-zero mixed values,
other mixed pairs, views, autograd and out= remain unqualified.

Two cold builds reproduced the same ELF. Locked CPU Torch, a separate
no-Torch capture and an independent Cartesian/readback oracle agreed on
268 cases and 8,874 words. They include both orders and alternating phases
at every length 1–65, mixed three-way rank4 fixtures and two empty outputs.
Of the new matches, 4,496 chosen I64 words covered signed extrema, carry into
the next exponent, both sides of 2^53 and 2^62 halfway points, and 1,297
exact halfway conversions across both signs and both ties-to-even directions.
The 4,378 chosen F64 words retained exact signed zeros, subnormals,
infinities and signed quiet/signaling NaN payloads. The raw reader uses
adjacent exact integer grid points and compares distances, not Python float
or the native helper. New Tensor/Storage/Version identities, initial Version
zero, unmodified inputs, both release orders, weak expiry and same-frame
result handle reuse were checked.

The exact prior ELF refused all 268 genuine new setups with status7 and
unchanged complete owners (2,680 frames, separate). The new candidate's
5,168 frames include 295 full rollback checks: six mixed-rank-zero guards,
four true positive-offset views, incompatible shape, compatible overcapacity,
other unadmitted mixed types, stale/padding and valid-then-invalid suffix.
F64 gradient marking was itself refused at setup (status 5); it was not
counted as a where-gradient refusal or rollback proof. A stale pin on the
first local observer aborted before native; the corrected observer ran the
same cold-built ELF. All earlier traces passed independent readback with
identical raw bytes and no newly retired control. Total successor 22,192
frames and 1,516 full rollback controls.

Run the reversible split projection with isolated Python
`prepare_i64_f64.py EXACT_PARENT EXACT_BUILD_PY`; an optional final new
directory materializes it. Use the included manifest to pin the second
cold build. The exact fixture can be recovered using
`expand_fixture.py i64-f64.fixture.json.zlib.b85 FIXTURE`
or generated using `oracle_i64_f64.py PINNED_HELPERS ORACLE_LOCK
INSTALL_REPORT FIXTURE` in the locked official wheel only.
Next run `native_i64_f64.py PINNED_HELPERS PROTOCOL_PY ELF FIXTURE OUT
baseline|candidate` without Torch on each exact image, and independently
`pure_i64_f64.py FIXTURE BASELINE_CAPTURE CANDIDATE_CAPTURE`.
