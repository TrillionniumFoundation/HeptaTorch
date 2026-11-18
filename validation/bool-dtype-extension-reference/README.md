# Observed Bool dtype extensions

This independent two-case package records actual pinned-Torch results for
`sum(dtype=torch.bool)` on binary32 `[1, -1]` and the genuine saved binary64
scalar `+1.0`. Both produce a fresh, detached rank-zero Bool result with byte
`01`. Input state and the full Torch RNG state remain unchanged.

`report.json.xz` restores the exact observed JSON; the manifest binds its
bytes, the literal case source, the unchanged logical reference producer, and
the actual remote execution receipt. All eleven repository-pinned
wheel/build/install provenance checks passed. The receipt preserves exit zero
and both complete output streams. This is reference evidence, not a native
execution or a project-wide closure claim.

The original logical147 report and the original PR38 F32 reader remain
byte-exact. `tests/sum_f32_review/fusion.py` executes the original reader,
retains the original dtype-refusal failures, and checks these two observed
successes separately on the current HIA image.
