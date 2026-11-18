# Original Integral24 comparisons on the current image

`run.py` is the compatibility entry for the complete original Integral24
suite. The original reader is preserved in the mode- and SHA-bound source
fixture at `validation/current-runtime-baseline`. The current identity
adaptation and outer admission are in `tests/current_runtime_historical`.

All 24 original cases, 260 reference events, and five historical dtype
controls execute. The original report remains `FAIL`: four unchanged refusal
assertions at indices 0–3 fail on the added Bool/F32 successes, while the fifth
refusal passes. The original 700 classifier's failure is preserved. The outer
admission accepts only those exact documented failures, their actual wire
successes, and complete passing numerical, ownership, input-custody, and EOF
checks. The new dtype contract is independently qualified by the F32 suite.

The physical-path adapter admits only the 44 exact original-manifest members
needed by the native source packages. All other paths still use the original
path guard and the unchanged original read/metadata checks.

Use `make test` or `tests/qualify_current_runtime.py`. The compatibility entry
uses the explicit arguments of `current_runtime_historical/run.py`;
`--prepare-only` remains `PREPARED_NOT_RUN` and cannot qualify native execution.
