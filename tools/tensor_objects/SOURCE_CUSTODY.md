# Source and evidence custody

`source-map.json` records all 237 generated core function bodies. Of the 143
original canonical bodies, 99 remain byte-identical and 44 have explicit reasons
and per-function before/after hashes. `canonical-to-v3.diff` records the complete
44-function change. The three-function native owner module is checked separately
in `profile.json`.

The scalar binary32 arithmetic functions are unchanged. Additional backward
changes control which values are read and which gradient branches execute.
Actual numerical/metadata tests, the independent model and pinned observations
qualify those changes. The object adapter reuses the existing Storage service
authoring with a checked namespace and enclosing-buffer base. Allocation is
performed by HIA, not by the authoring program.

The builder checks ten named historical authoring baselines byte-for-byte
against canonical functions. Two historical source snippets require explicit
normalization: `back_unary` reproduces the canonical select expression exactly;
`pool2d` includes the existing 16,777,216-term guard and matches the canonical CFG
exactly after a bijective SSA/block-name renaming. The emitted v3 guard is never
removed or weakened. `baseline-checks.json` records both the original snippet
discrepancy and the successful normalization.

The canonical numerical source, generated v2 source and authoring, Storage
authoring, optimizer-version helper/producer, imported upstream authoring,
compiler, linker and all generated unit/object snapshots are pinned before use
and rechecked after the build. The build refuses an unrecognized final source,
object or executable. ABI2 and ABI3 are separate profiles. ABI3 adds stack
interface metadata and startup policy while retaining identical application
bytes from offset 4,096 onward.

## Upstream and actual observation basis

The PyTorch source pin is
[`31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`](https://github.com/pytorch/pytorch/tree/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457).
The independent review's `upstream/custody.json` binds 20 inspected files, and
`save-policy-review.json` records the 30 currently implemented AD opcode kinds.
The source basis includes TensorImpl/VariableVersion, StorageImpl, SavedVariable,
AccumulateGrad, derivatives.yaml and its generator, TensorFactories clone, and
the dense CompositeImplicitAutograd matmul routes in LinearAlgebra.cpp.

The actual 22-case lifetime observation has SHA256
`43aadde8aef7f6406b6d0bd96f7b72fc724583e9bdcee823fa971ef8558f4d0c`,
from hosted run 43801012881, job 133081011984. Its eleven provenance checks passed.
The separately delivered independent replay consumes those actual results.
Local source reading or replay is not reported as a new local PyTorch execution.

## Failed attempts remain distinct

Candidate06's empty-clone and two materialized-matmul failures remain in the
independent review handoff with their original identities and executed sources.
Candidate08's scalar-dispatch failure is preserved here under
`evidence/scalar-classification-failure-08/`, including the exact executed test,
input/response streams and report. Candidate09 changes two classification
constants, each removing bit14, and passes the unchanged failing request.
`evidence/scalar-classification.diff` contains the complete two-constant HIA diff.
No previous report is relabelled as a test of candidate09.

Earlier source/compiler admission refusals remain in the isolated development
archive. SSA/block-argument limits and a module dependency cycle were resolved
by factoring and ordering source, without raising native limits. One portable
source-custody assertion initially demanded raw byte identity from a manually
numbered historical pool guard; the final explicit alpha-renaming check records
the exact reason for that refusal. No native test tolerance was weakened.
