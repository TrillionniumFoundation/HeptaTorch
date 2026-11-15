# Validation evidence

The final native implementation is bound by `native.json` to both its HIA source
and executable SHA-256. It includes 123 tensor/integration cases, 17 optimizer
cases, seven spatial/training integration cases, 894 scalar special-value
vectors, and 19 independently reviewed adversarial groups. These counts describe
different sets and units; they are not a count of upstream PyTorch tests.

The native packet corpus contains 76 input/expected-output pairs. Its exact
files, sizes and hashes are recorded in `fixture-manifest.json`. `make test`
executes all of them through the native HIA hexadecimal decoder and tensor
executable with empty PATH. Responses were recorded only after independent
numerical/analytical/finite-difference checks or explicit rejection checks.

`components.json` retains the larger component campaigns, their seeds, reference
methods, numerical errors, and distinct executable identities. They include the
2,314,020 basic arithmetic comparisons, 3,424,359 elementary-function comparisons,
20,025 high-precision controls, and 144 spatial gradient finite-difference
coordinates. Component adapter results are separate from the final integrated
executable's observations.

`bootstrap.json` records a fresh compiler bootstrap. The native compiler's source
and binary pins are in `compatibility/hia-toolchain-lock.json`; the full recipe is
the executable GitHub workflow. The finite bootstrap passes do not claim the
separate upstream full cold-qualification suite or a formal compiler proof.

## Independent review findings

Four concrete implementation issues were found and corrected before publication:

1. Empty gradient reads bypassed nonleaf/requires-grad validation.
2. A detached alias could modify a saved output used by backward.
3. A squared denominator caused avoidable extreme-value division-gradient errors.
4. Adam's early gradient squaring and early variance debiasing could suppress
   valid large-gradient updates.

The final review reran malformed inputs, invalid IDs, shape/payload bounds,
empty tensors, storage aliasing, stale versions, graph isolation, fan-out,
repeated backward, optimizer state/configuration contracts, large gradients,
and descriptor/workspace/output/native-wrapper limits. All reported groups
passed against unchanged source and executable hashes.

## Hosted PyTorch differential gate

The CI workflow installs the pinned **CPU PyTorch 2.15.0+cpu** only as an
independent test oracle. Its inline test program serializes inputs and invokes
real PyTorch forward operations, autograd, and optimizers. It never implements
HeptaTorch's runtime operations. The gate requires exactly 76 cases, zero
failures, the exact wheel version, source revision and artifact hash, and an
unchanged native executable. Numerical tolerances are recorded per case.

`compatibility/pytorch-oracle-lock.json` fixes the official Python 3.12 Linux
x86-64 wheel URL and SHA-256. Pip verifies the downloaded wheel against that
hash. The installed distribution's source URL and archive hash, exact package
version, CPU backend and build Git SHA are checked independently of numerical
comparisons.

The inventory release commit is `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`.
The official wheel was built at its sole parent,
`d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`. The release adds only five lines of
Ruff lint configuration and comments to `pyproject.toml`. Comparing the complete
74-entry root trees established identical Git objects for every other entry,
including every source, generator, test and third-party subtree. Both pins and
the exact source-equivalence evidence are retained in `hosted.json`.

The [first hosted run](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43737347959)
passed compiler bootstrap, 76 native packet pairs and all 76 real-PyTorch
numerical cases (2,754 values). Its overall result remains **FAIL** because the
original gate incorrectly required the wheel build SHA to equal the release tag
SHA. That failed result is retained in `hosted.json`; the provenance correction
must pass a fresh hosted run before a hosted PASS is recorded.

Local execution lacked PyTorch and remains recorded as NOT_RUN for that oracle.
A hosted result is tied to its tested GitHub commit and run URL; none of these
scoped checks establish whole-schema or whole-project parity.
