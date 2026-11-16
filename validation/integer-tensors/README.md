# Int64 producer evidence, candidate 07

The source-only manifest in `tools/integer_tensors` remains exactly the manifest
used by the real cold execution. Its historical preparation status is not
rewritten. `qualification.json` and `evidence/portable-cold07-report.json`
record the later actual cold result on that exact package.

All ten named top-level gates passed, including a new eleven-module native
build, the seven producer reference gates, the independent unchanged actual40
runner and twelve legacy behavior gates. The legacy coordinator includes the
original sixteen gates and their unchanged explicit partial/open outcomes.
These are overlapping evidence groups, not a new API denominator.

The cold archive preserves the actual source, executable, every compiled unit,
commands, stdout/stderr, original reports and completed raw archives. No native
execution is replaced by a Python model. Extract an outer archive, then follow
its preserved nested `RESTORE.json` and member manifests to restore raw gate
streams; restoration indexes retain the original execution paths as context.

History is deliberately additive. Candidate03 and candidate06 checkpoint
archives and manifests remain original bytes, including their original pending
scope labels. They contain the initial host observer failures, candidate04
as_strided/view and promotion-layout failures, later fixes and source-specific
results. Candidate06's independently discovered three in-place failures and
candidate07's independent actual40 success remain in the separate root-owned
`validation/int64-inplace-review` namespace. The new initial07 archive also
keeps both separately executed legacy06 and legacy07 suites; neither is
renamed as the other's execution.

Resource and owned-entry gates, legal prior-profile owner transfer, live32,
strict35 and copied-snapshot negative controls have separate owners, source
pins and manifests. They are required by integration and are not inferred from
the producer status. Mixed-dtype AD, public Tensor.to, a Python exception-text
frontend and full PyTorch completion remain unclaimed. The overall project
remains HOLD with denominator 2590.
