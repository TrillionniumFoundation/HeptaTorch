# Native where.self: scalar Bool condition over same-shape CPU values

A true rank-zero Bool condition with one element now selects between two
existing equal-shape, same-dtype values. Both value tensors and the condition
still must have live owners and be ordinary, contiguous, zero-offset, CPU,
and no-grad. No broadcast of either value, singleton-dimension expansion,
dtype promotion, views, out=, autograd, hosted aggregate, or default activation
is included. Opcode 129 and all its other modes remain unchanged.

The exact previous #66 ELF refused all 19 new valid baseline setups, with
status 7 and complete owner rollback (188 raw frames). Two independent cold
builds reproduced the same successor ELF. Official locked CPU PyTorch and
independent readback of a separate no-Torch native session matched 19 new
cases and 150 output words: six dtypes with both Bool choices, six owned
typed empties, and a repeated-input case. Includes signed zeros, subnormals,
infinities, qNaN/sNaN payloads and integer extrema. Both release orders,
fresh Tensor/Storage/Version identities and zero result Version held.
Independent raw replay verified 38 complete owner rollbacks across 416
frames, including true positive-offset Bool and F32 view inputs (three
positions), dtype/rank/shape failures and accepted selector followed by
invalid command in one trial. An actual request-local result reference passed.

Prior #66 same-shape replay (including its allocation refusal and recovery)
and all four #65 replay classes preserved request and response bytes exactly.
Totals: 3,431 successor frames and 375 independent exact owner rollbacks
across all successor suites. The baseline refusals are recorded separately.
No new allocation-pressure run was added to the scalar variant.

Reproduction uses the exact pinned #66 source. Run
`prepare_scalar_mask.py EXACT_PINNED_SOURCE EXACT_BUILD_PY` with isolated
Python to verify the reversible source projection without compiling; an
optional final fresh directory argument materializes the projection. Perform
the pinned complete build twice, using the new manifest with object and ELF
pins for the second build. `oracle_scalar.py PINNED_HELPERS ORACLE_LOCK
INSTALL_REPORT FIXTURE` is locked CPU-Torch only. Then use `native_scalar.py
PINNED_HELPERS PROTOCOL_PY ELF FIXTURE OUT baseline|candidate` twice with the
appropriate image and no Torch import. Finally run
`pure_scalar.py FIXTURE BASELINE_CAPTURE CANDIDATE_CAPTURE` in independent
Python; it reads raw frames and owner images only. Prior captures replay
with their original fixtures and body/reader bytes, updating the expected
candidate ELF only. The complete local raw captures remain separately retained.
