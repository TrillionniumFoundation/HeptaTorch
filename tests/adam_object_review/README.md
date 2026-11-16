# Independent review of native Adam objects

This package binds the independent execution of the unchanged actual PyTorch
observations to the native Adam02 source and executable. It is an independent
review package, not the producer's implementation or build recipe.

## Result and limits

The new replay executes original cases 21–24: lazy Adam state, lazy AdamW state,
duplicate parameter occurrences, and externally retained state Tensor lifetime.
All four pass: 908 native request/response frames, 3,469 exact checks and
1,223,232 bytes compared for rollback. All 210 parameter/moment numeric positions
also match the reference binary32 words exactly. The predeclared numeric policy
remains `rtol=4e-5, atol=5e-6`; this review did not relax it.

The original replay, with its source and assertions unchanged, independently
executes its 19 cases against the same Adam02 image: 18 qualified, one partial
retained-nonleaf case, no unexpected failures. Its five original OPEN rows remain
unchanged in that report. Combining only the disjoint executed indices from the
two reports gives 22 qualified cases, one partial case and one unexecuted public
gradient-object case out of the fixed 24. This is not complete Optimizer or
PyTorch compatibility.

The source is `253438648a437a75ef2dbc77138eca74cb5fe6a6e2ea322c7c38e5a535e5d2a6`;
the executable is `fa92356954dbb9e0c454f2a58c2f78aa33950103d377c57e7b659ec9904183af`.
The reference is `7ad0a2e718300a33c91a9212f14ada3c0c7997be7fc77f64bb557c2ecd858160`.
Its actual official PyTorch wheel provenance has all 11 checks true. These
replays execute the HIA-produced native ELF and compare that fixed observation;
they do not run a fresh PyTorch instance or provide a host numerical target.

Case 24 uses the native command 95 to add 0.5 to the real externally retained
step and exp_avg Tensors. It verifies one Version increment for each complete
operation, unchanged Tensor/Storage/Version identities, actual weak-storage
expiration, and the saved-Version failure on later backward. Reference values
are never written into those state Tensors.

The entry remains restricted to CPU binary32, ordinary non-capturable,
non-differentiable, non-foreach, non-fused Adam/AdamW without AMSGrad. An actual
external step Tensor is used. Incremented step values must be positive integral
binary32 values below 2^32; arbitrary real bias-correction exponents remain open.
Public `.grad` Tensor identity/Storage/Version, retained-nonleaf membership,
Python warning objects, arbitrary late-error Python Optimizer side effects and
authentication of fabricated foreign owned state are not closed by this review.

## Repeat native qualification

Use a source and executable rebuilt by the producer's Adam02 recipe. Run from
the repository root; replace the two paths below with those actual artifacts.
The scripts check every input hash before loading helpers and again after
execution. They refuse an existing output directory.

```sh
ADAM_BINARY=/absolute/path/to/tensor-adam-session
ADAM_SOURCE=/absolute/path/to/runtime.hepta

python -B tests/adam_object_review/replay_adam_objects.py \
  --binary "$ADAM_BINARY" --source "$ADAM_SOURCE" \
  --controls tests/adam_object_review/native_release_controls.py \
  --transport-observer tests/adam_object_review/client.py \
  --legacy-replayer tests/adam_object_review/replay_optimizer_objects_v2.py \
  --observations validation/adam-object-review/torch-optimizer-objects.json \
  --output build/adam-object-review-new \
  --expected-sha256 fa92356954dbb9e0c454f2a58c2f78aa33950103d377c57e7b659ec9904183af \
  --expected-source-sha256 253438648a437a75ef2dbc77138eca74cb5fe6a6e2ea322c7c38e5a535e5d2a6

python -B tests/adam_object_review/replay_optimizer_objects_v2.py \
  --suite optimizer --binary "$ADAM_BINARY" --source "$ADAM_SOURCE" \
  --controls tests/adam_object_review/native_release_controls.py \
  --transport-observer tests/adam_object_review/client.py \
  --observations validation/adam-object-review/torch-optimizer-objects.json \
  --output build/adam-object-review-original \
  --expected-sha256 fa92356954dbb9e0c454f2a58c2f78aa33950103d377c57e7b659ec9904183af \
  --expected-source-sha256 253438648a437a75ef2dbc77138eca74cb5fe6a6e2ea322c7c38e5a535e5d2a6
```

The new replay is pinned to the exact original observations and three existing
helper files. The original replay and helper files here are byte-for-byte
copies; no original assertions or first-20 expectations were edited.

## Source audit

`audit_native_source.py` performs read-only checks without importing authoring or
numerical modules. It verifies the exact source snapshots, 308 unchanged
baseline function bodies, retained SGD dispatcher/auditor bodies, the isolated
Adam moment allocator, four explicit numerical-access changes and their exact
inverse, and the unchanged owned bridge and RNG modules. The numerical inverse
restores the frozen arithmetic function byte-for-byte.

```sh
python -B tests/adam_object_review/audit_native_source.py \
  --baseline /absolute/path/to/optimizer-candidate-08 \
  --previous /absolute/path/to/adam-candidate-01 \
  --build /absolute/path/to/adam-candidate-02 \
  --upstream-sources /absolute/path/to/pinned-adam-upstream \
  --output build/adam-source-audit.json
```

The primary-source files are individually SHA256/Git-blob pinned to PyTorch
commit `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`; their exact URLs and hashes are
recorded in `source-audit.json`. The read-only findings are not described as a
fresh compiler bootstrap or as proofs about arbitrary machine code. The
producer's native build receipt and all concrete compiled units are retained
in the evidence archive for inspection.

## Original evidence and failure history

`evidence/native-observations.tar.gz` preserves all request/response streams,
process observations, per-case reports, executed script copies, both candidate
histories, and the exact Adam02 source, unit objects, link request and ELF.
`evidence/archive-manifest.json` records every decoded file's size, SHA256 and
mode; every member was decoded and checked before freezing.

The first attempted new-four run on candidate01 failed to create its output
directory with ENOSPC, before any native process started. Its explicit NOT_RUN
record is retained. The original 19 cases did execute on candidate01 and their
separate unchanged report is preserved. The first actual new-four execution
was on candidate02 and passed. No failure was reclassified as a native pass.
An earlier source-audit receipt is also retained; the final audit only changes
one metadata key to English and adds spaces in two prose fields, with all
checks and conclusions unchanged.
