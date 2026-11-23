# Native where.self: trailing-singleton Bool condition on CPU

Adds only same-rank rank 1–4 trailing-singleton condition broadcasting for
ordinary no-grad owned CPU tensors. The prefixes must match axis by axis,
the Bool condition's last extent must be exactly 1, and the last output
extent and counts are bounded and checked. Values still have identical
shape and public dtype. In the nonempty case the quotient divides an
output index by the checked last extent only if it exceeds 1. Zero last
extent yields zero output and no mask load or division. Zero prefix extent
also returns a fresh zero-element result. No general broadcasting,
promotion, view input, gradient, out=, or default activation.

Two independent cold builds reproduced one ELF. Separate locked official
CPU PyTorch and coordinate-wise independent raw readback of native no-Torch
captures matched 10 cases and 92 words across six dtypes and ranks 1–4,
including both [3,1]→[3,0] and [2,0,1]→[2,0,4]. The exact predecessor
refused every case with status 7, no published output and byte-identical
complete owners. The candidate recorded 31 new complete rollback checks
over 255 frames, preserving real positive-offset input refusal, wrong
prefix even for zero result count, wrong rank/last/dtype, and a valid
selector followed by an invalid same-frame request. Fresh Tensor,
Storage and Version, unchanged inputs, zero result Version, both release
orders, weak expiry and same-frame result reference were verified.

The original scalar campaign’s frame 385 remains a recorded, byte-exact
status-7 refusal, and the independent reader proved the same element
inputs and geometry are now a positive F32 fixture. The 415 other scalar
replay frames (37 rollbacks) match byte-for-byte after removal of only the
retired refusal. The same-shape #66 and all #65 replay streams matched
byte-for-byte. Across new plus retained successor suites: 3,685 raw frames
and 405 full-owner rollback controls. Old predecessor runs are separate.

Use isolated Python to run `prepare_trailing.py EXACT_PINNED_SOURCE
EXACT_BUILD_PY` for source/partition verification in memory; the optional
final fresh directory materializes the exact reversible projection. Lock
the second cold build with the included pinned manifest. Generate the
official fixture separately with `oracle_trailing.py PINNED_HELPERS ORACLE_LOCK
INSTALL_REPORT FIXTURE`. Capture with `native_trailing.py PINNED_HELPERS
PROTOCOL_PY ELF FIXTURE OUT baseline|candidate` separately on the two exact
native images. Run the raw-only `pure_trailing.py FIXTURE BASELINE_CAPTURE
CANDIDATE_CAPTURE` and `verify_retired.py ORIGINAL_SCALAR_CAPTURE
RETAINED_SCALAR_CAPTURE NEW_TRAILING_CAPTURE FIXTURE`. Historic suites use
identical reader/fixture bytes, updated executable pin and, for the retained
scalar native observer only, exactly the identified retired refusal omitted.
