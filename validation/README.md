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

The CI workflow installs the pinned **CPU PyTorch 2.15.0** only as an independent
test oracle. Its inline test program serializes inputs and invokes real PyTorch
forward operations, autograd, and optimizers. It never implements HeptaTorch's
runtime operations. The gate requires exactly 76 cases, zero failures, the
expected Torch source revision, and an unchanged native executable.

Local execution lacked PyTorch and was correctly recorded as NOT_RUN for that
oracle. A hosted PASS is claimed only after reading the corresponding Actions
result and logs. Its result, when available, is recorded with the tested GitHub
commit and run URL; it does not establish whole-schema or whole-project parity.
