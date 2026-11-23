# Native CPU where.self: explicit Bool + F16/BF16/F32/F64 Tensor pairs

Adds exactly one Bool and one public floating value Tensor, in either order,
for an exact Bool condition. Output is the exact non-Bool dtype. Each input
has independent width and live ordinary contiguous/no-grad CPU owner checks.
The previous Bool/I64 and all same-dtype modes are preserved, as are the
wire ABI, global casts and promotion, and prior shape caps. The selected
floating word stays raw; only selected canonical Bool is converted, to +0
or the exact target +1 bits. Actual rank-zero tensors are included; wrapped
Python scalars, unrelated non-Bool mixed types, views, autograd, out=, hosted
aggregate and default routing remain outside this slice.

Two independent cold binaries were identical. Locked official CPU PyTorch,
separate no-Torch physical capture and independent Cartesian reader agreed
on 60 cases and 816 words: 408 unchanged floating raw words and 408 Bool
conversions (196 exact positive zeros, 212 exact ones). In every floating
dtype the physical selected words cover both signs of zero, subnormal,
infinity, quiet/signaling NaN payloads, and finite values. Includes both
operand orders, 16 actual rank-zero outputs, 16 cases with exactly one
rank-zero value, eight empty outputs and rank-four cases. Fresh distinct
Tensor, Storage and Version identities, initial Version zero, both release
orders, weak expiry, unchanged input bytes/metadata, and local-result
consumption were verified.

The exact previous image refused 60 new valid setups with status 7 and
unchanged complete owner bytes (600 frames, separate). The successor's
new 1,206 frames include 80 full-owner rollback checks: true positive-offset
input views, incompatible shape, compatible overcapacity (status 3), stale/
padding/grad refusals, eight still-unsupported non-Bool mixed orders, and an
accepted new operation followed by invalid suffix. Four previously recorded
BF16/Bool and F64/Bool refusals remain archived, with identical original and
positive-successor input words and geometry independently proved. Only
those four records were removed from retained replay. All other #66 and #65
frames matched byte-for-byte: total successor 5,752 frames and 559 exact
rollback checks.

Run the source projection without compiling using isolated Python:
`prepare_bool_float.py EXACT_PARENT EXACT_BUILD_PY`. An optional final
directory materializes the exact checked split. The included manifest pins
the second cold build. `oracle_bool_float.py PINNED_HELPERS ORACLE_LOCK
INSTALL_REPORT FIXTURE` uses the official wheel separately.
`native_bool_float.py PINNED_HELPERS PROTOCOL_PY ELF FIXTURE OUT
baseline|candidate` runs without Torch. Independently read the resulting
raw artifacts with `pure_bool_float.py FIXTURE BASELINE_CAPTURE
CANDIDATE_CAPTURE`. Compare exact original status/owner witnesses and
retention independently with `verify_retired.py ORIGINAL_BOOL_I64_CAPTURE
RETAINED_BOOL_I64_CAPTURE NEW_CAPTURE FIXTURE`.
