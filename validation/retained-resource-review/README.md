# Retained-gradient independent resource review

This package preserves the original independent resource and ownership review of
the 446-function retained-gradient candidate. Its native execution reports remain
**8/8 resource controls and 2/2 typed handoff directions PASS**. The source review
is separately a read-only PASS. These results do not replace the producer's
17 suites or the other reviewer's 12 actual-PyTorch variants.

The original evidence archive was interrupted when the Cloud environment became
unavailable. The report bytes survived. All eight resource-control raw streams,
both runtime builds, and the typed handoff executable were recovered. The tail
containing the typed handoff raw streams did not survive. This package therefore
does **not** claim complete original raw custody. A new cold replay must have its
own output directory and receipt; it cannot replace or relabel the old run.

## Fixed identities

| Item | SHA-256 |
| --- | --- |
| Retained 446 source | `666230f05cc6942470006d1f184135321d8cbc26a6c6ba0bf7e3cf9857fe515b` |
| Retained ELF | `7568c0a4ec40d82d3a0b46d1e8a27499dd96e8fe3fd15433523131a196da89b5` |
| Previous 428 source | `1fed8ea7ba5bf755a9df9ca1e61d34cfd4712cc2fd66e0728493091c81c24dae` |
| Previous ELF | `6773485896d961b6c2e62784ebb66320290df47128bd1331aecc0c8b7f6611fe` |
| Native compiler | `b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f` |
| Native session linker | `aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16` |
| Fixed actual12 reference | `ca8b5d3a7c679dcc5ad9a3d7b2bbd38f664c90e6474cf3e3c28496d5b716d1b3` |

The new numeric/backing state tags are `HTRET001` / `HTORET01`, with 576-byte
descriptors and eight object kinds. The previous tags are `HTGAD002` / `HTOGAD02`.
The test controller transfers genuine HIA owners between exact checked entry
signatures. It imports no arbitrary host pointer and copies no owner descriptor.

## Three portable entry points

Run from the repository root after importing this package's `tests/` and
`validation/` trees. Python is test transport and observation only. The compiler,
linker, and all tensor operations execute as native HIA programs. Linux x86-64,
read access to the runner's own child's `/proc` state, and `resource.prlimit`
support are required. No Torch import is performed by these three runners.

Set `RETAIN_BUILD` to a freshly built retained candidate, `PARENT_BUILD` to the
previous 428 build, `STACK_BUILD` to the native stack toolchain build, and
`REVIEW_OUTPUT` to a new output root. The two runtime build directories must
include `build.json`, `runtime.hepta`, their executable, and all nine `.hepta`
and `.o` units. The stack directory contains `native/stack-module-compiler` and
`native/stack-linker-session`. Each output leaf below must not already exist.

```sh
python -B tests/retained_resource_review/test_resources.py \
  --binary "$RETAIN_BUILD/tensor-retained-session" \
  --source "$RETAIN_BUILD/runtime.hepta" \
  --expected-source-sha256 666230f05cc6942470006d1f184135321d8cbc26a6c6ba0bf7e3cf9857fe515b \
  --expected-native-sha256 7568c0a4ec40d82d3a0b46d1e8a27499dd96e8fe3fd15433523131a196da89b5 \
  --controls tests/retained_resource_review/native_release_controls.py \
  --observer tests/retained_resource_review/owner_client.py \
  --reference validation/retained-resource-review/actual/torch-retained-nonleaf.json \
  --output "$REVIEW_OUTPUT/resources"

python -B tests/retained_resource_review/test_typed_handoff.py \
  --build "$RETAIN_BUILD" --previous-build "$PARENT_BUILD" \
  --stack-build "$STACK_BUILD" \
  --protocol tests/retained_resource_review/protocol.py \
  --controller tests/retained_resource_review/state_handoff.hepta \
  --output "$REVIEW_OUTPUT/typed-handoff"

python -B tests/retained_resource_review/source_review.py \
  --build "$RETAIN_BUILD" --previous-build "$PARENT_BUILD" \
  --peer-runner tests/retained_nonleaf_review/replay_retained_nonleaf.py \
  --review-inputs validation/retained-resource-review/source-inputs \
  --output "$REVIEW_OUTPUT/source-review"
```

The last command requires the separately frozen actual12 runner with SHA
`c925e33872dc87ead29ecc2125b1799763a87b13c8ae2eba85e966b4202f1d8e`.
It reads that source; it does not run the actual12 suite. The checked authoring
snapshots are also read only, never imported or executed.

## Original results and scope

`resources.json` is the original 320-frame report: 34,190 exact checks and
13,848,608 bytes of whole-owner rollback comparisons. The eight cases cover weak
retain with and without a separately held old gradient, version-error and
released-Saved prehook prefixes, unvisited reservation cleanup, ordinary atomic
rollback, real allocation failure and recovery, Optimizer strong roots across
growth/reset/release, and empty fresh-gradient cleanup. Commands 7, 29, 57, and 58
are exercised with the new retain flag. The allocation-failure case lowers only
its own child's address-space limit, verifies a same-capacity query still works,
observes failure status 3, and restores the limit before proving recovery.

`typed-handoff.json` records two directions, 10 frames, 19 real linked modules,
and 1,398,048 bytes of rollback comparisons. Each foreign state is rejected by
the other profile before mutation, then returned to its original profile and
successfully used. Real close calls release the owner mapping. Symbol prefixes
are the only source adaptation for linking both runtimes; the test proves an
exact inverse for every changed unit.

`source-review.json` binds all 446 core bodies to exactly one real compilation
unit, checks 410 unchanged parent bodies, the 18 changed parent bodies and
18 added functions, preserved owner/RNG modules, weak-self and temporary-root
cleanup order, and the separate observer's public-token lifetime handling.
It does not certify arbitrary externally forged state or arbitrary machine code.

## Interrupted archive and recovery

`validation/retained-resource-review/raw-evidence.tar.gz` remains the exact
interrupted 6,249,696-byte artifact, SHA
`45deb8ecdba66e41dbea7807c1cf8d4e4db87b62323a0a4dff3a102e518d29b6`.
It lacks a gzip end marker. It must not be treated as an ordinary complete
archive. The original process session 61505 has unknown completion status.

The `recovery/` directory contains the forensic report, transparent recovery
scripts, a valid `complete-members.tar.gz`, its member manifest, and a binding
check against the unchanged execution reports. The valid archive contains only
the 114 fully readable members, totaling 30,506,771 uncompressed bytes; every
member's length, SHA and mode and the new gzip CRC were checked. Its SHA is
`b9b07c55a51f25be748d3614c42d62458d252bd4939ae022eed666903d784dfa`.

The recovered set includes complete previous/new sources, nine objects and
executables, all 320 resource request/response frames and owner snapshots,
the source-review inputs, and the real typed executable
`8c19f79c0f491dae5dc59f5d4510749fe4afb6c85c3e7d926b70b7f0883c12ed`.
The recovery checker compared each resource frame with its original report hash.

The first incomplete member is
`review-runs/typed-handoff-01-first/link.request`, which declared 5,473,308 bytes.
Its remainder, the subsequent renamed unit files and typed raw transport are
unavailable from the interrupted archive. The original typed report still
contains all ten request/response hashes and its PASS observations. None of the
missing original bytes have been reconstructed or claimed recovered. Recovery
did not rerun any native program or Torch.

## Reusing the previous 428 regressions

The old mathematical mixed2, actual mixed2, and actual scalar-NaN48 runners can
use their unchanged source with the retained source/image pins supplied through
their existing CLI. The strict extreme8 observer and comparator likewise accept
the new pins without a tag patch. These are invocation observations, not new
execution results on the retained candidate.

Only `test_joint_gradients_v2.py` reads the old numeric/backing tags directly.
A separate derived copy needs exactly one `HTGAD002` to `HTRET001` replacement
and one `HTOGAD02` to `HTORET01` replacement, with an exact byte inverse to the
frozen original. Its protocol, observer, numerical assertions, tolerances,
counts and expected values stay unchanged. Do not transfer its previous PASS
to the new image without executing it. The separate producer's legacy controls
for previously unknown commands 93/94 have an explicit contract adaptation;
the original failures must remain historical evidence.

These resource controls add no API-completion credit. Full PyTorch compatibility,
all dtypes, all RNG behavior and the whole rewrite remain incomplete. The global
denominator remains 2,590 and the overall decision remains HOLD.
