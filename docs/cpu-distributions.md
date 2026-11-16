# Native CPU Bernoulli and dropout

This is a bounded native HIA component over the seven-kind Tensor/Generator
owner. Its compatibility status is **PARTIAL** and the whole-project acceptance
status remains **HOLD**. No full upstream schema or full subsystem gate is
closed by counting these finite observations.

## Two explicit contracts

| Profile | Source SHA256 | Native ELF SHA256 |
| --- | --- | --- |
| Atomic 03 | `a72a98232c7b7f10efa67ce29b21e417c58201383bef9aacc266fe247ff356ce` | `515790a191b3db1ecbfcb3f2afab644ef980b51ad85c5d6daf1b8c5a9a5a6e28` |
| Explicit partial 01 | `55ed0c6e16317add03974c01ffd85e38684fc535a3e5cb6423ed043f4217f05b` | `bb4c25e35a170f08276d464db6b630cf03765ff816e6a9ca9aa1651360a1ccf4` |

The [atomic protocol](../tools/cpu_distributions/PROTOCOL.md) provides commands
80–85: scalar-probability Bernoulli, Tensor-probability Bernoulli, functional
Bernoulli, ordinary functional/in-place CPU dropout, and a checked alias of
live Saved dropout noise. Its ordinary errors roll back the whole application
frame. The [successor protocol](../tools/cpu_distributions_partial/PROTOCOL.md)
preserves all 19 predecessor `dist_*` bodies and all old command semantics,
then adds commands 86 and 87 with explicitly different error results.

| Command | Operation | Explicit late-error result |
| --- | --- | --- |
| 86 | In-place float32 Tensor-probability Bernoulli | Invalid per-element probability keeps the valid prefix's writes and MT32 consumption; the invalid element consumes nothing. |
| 87 | Ordinary CPU in-place dropout | A tracked-leaf or definite-overlap error after noise creation keeps that RNG advancement, writes no input element and retires the temporary noise. |

Both return a normal record containing the operation error, actual public
Tensor handle, completed write count and consumed raw MT32 count. A nonzero
operation error commits the already executed prefix and stops the suffix.
The failed operation does not increment Tensor Version. Earlier successful
commands can have independently changed that Version or Generator state.
Invalid handles/flags/state, unsupported graph cases, output/work limits,
Version exhaustion and allocation failure retain whole-frame rollback.
An ordinary later failure also rolls back an earlier successful 86/87.

The outer framing remains `HTOWNR01` / `HTOWNS01`; application bodies remain
`HTORCH03` / `HTORSP03`. This does not make owned states interchangeable.
The atomic tags are `HTDIST01` / `HTODIST1`, and the successor tags are
`HTDIST02` / `HTODIST2`. Real owner handoffs check both refusal directions,
complete owner preservation, recovery by the original profile and actual
unmapping on close. No process pointer or owner descriptor is accepted from
the wire or copied into the numerical pool.

## Numerical and autograd behavior

The [source inventory](../tools/cpu_distributions/primary-sources.json) binds
the official PyTorch source used for distributions, TensorIterator, overlap,
preserved layout and derivatives. Scalar probabilities retain their binary64
bits. Float32 Tensor probabilities use the output-first multioperand iterator
order, broadcasting, nonzero-overlap cases and per-element alias feedback.
The complete 5,056-byte Generator state, including normal caches, stays inside
its existing private Hold.

Nonempty scalar-probability Bernoulli consumes two MT32 words per element;
float32 Tensor-probability Bernoulli consumes one per valid element. Probabilities
zero and one do not eliminate Bernoulli draws. Ordinary dropout has separate
identity paths for evaluation, zero probability and empty input; probability
one uses real floating multiplication by zero without RNG consumption.
For intermediate probability it creates actual noise, scales it, and multiplies
through the input's layout. These are distinct from `native_dropout` fast paths,
mask dtype and scaling order.

Functional Bernoulli uses a true zero derivative, including positive-zero
results for negative-zero, infinite and NaN incoming gradients. Functional
dropout retains a real Saved noise Tensor with shared Storage/Version; mutation
and released-Saved checks remain active. Normal-gradient-mode in-place graph
rewriting for tracked nonleaf Tensors is still unsupported. Details and exact
limits remain in the two immutable protocols.

## Fixed actual evidence

The [original hosted report](../tools/cpu_distributions/evidence/torch-bernoulli-dropout.json)
has SHA256
`c624f5c4762b1c71e7053aba9af55aaa3d8bbaae5aa100b0a6df5f44f58da467`:
183 observations, no setup errors and all 11 official-wheel provenance checks.
The report and its probe are unchanged. Local native replay compares those
recorded observations; it does not claim to run PyTorch again.

| Native replay | Strict PASS | Transaction difference | Unrepresented |
| --- | ---: | ---: | ---: |
| Atomic 03, commands 80–85 | 161 | 4 | 18 |
| Successor, unchanged commands 80–85 | 161 | 4 | 18 |
| Successor, explicit commands 86/87 where required | 165 | 0 | 18 |

Only zero-based rows **101, 103, 105 and 179** change classification under the
new explicit interfaces. The first three are late invalid Tensor probabilities;
the last is a tracked-leaf in-place dropout error after noise consumption. The
original atomic differences are still reported, rather than deleted or
reclassified under the old interface.

The 18 unrepresented rows remain exact and individually named:

- Eight float64 probability Tensor observations: indices 4, 9, 14, 19, 24, 29,
  34 and 39. A double scalar probability does not implement a double Tensor.
- Two int64 probability Tensor observations: indices 109 and 110, including
  an empty-output case. Their output Tensor is float32.
- Eight distinct `native_dropout` observations: indices 167–174. A float
  payload is not substituted for the missing bool-mask Tensor interface.

The explicit replay executes 1,908 frames before EOF, compares 49,085 float32
words exactly and checks 182 complete Generator states. Its rollback comparison
total is 9,051,360 bytes. Independent native controls cover one million active
elements, exact probability rounding, strides and aliases, zero/empty paths,
late-error prefix/suffix behavior, output bounds, real allocation refusal and
typed owner handoffs. Repeated controls are not added to upstream coverage.

## Build and CI admission

See [the build commands](build.md#native-cpu-bernoulli-and-dropout).
`make cpu-distributions` rebuilds atomic03 from the exact nullable Tensor/RNG
predecessor and the common/materialization authoring layers.
`make cpu-distributions-partial` rebuilds the explicit successor from that
atomic build. `make test-cpu-distributions` actually executes the original five
suites plus the successor's six suites, then binds their reports.

The 37-file and 28-file frozen manifests remain unchanged. Every production
unit passes the existing native compiler; the final application passes the
native linker, resource interfaces and whole-call-graph stack admission.
The per-function host authoring adapter only handles the aggregate document
capacity boundary. It is not an assertion that the whole document passed the
original aggregate host Core admission, and it introduces no synthetic callee.

The report checker enforces concrete source/image/peer identities, the complete
ordered reference rows, exact PASS/OPEN partitions and every historical archive
member's bytes, SHA256 and mode. CI separately requires the unchanged live32
Tensor/RNG/backward oracle and frozen35 Tensor oracle on the final explicit
successor image. Those existing oracle scripts, assertions, tolerances and
provenance checks are retained.

Historical aggregate-authoring rejection, replay-observer ordering mistakes,
wrong wire fixtures, the killed in-memory compression attempt and integration
ENOSPC events remain preserved in their respective records. Host test failures
and environment failures are distinguished from native contract failures.

## Remaining scope

The profile has rank at most four, nonnegative strides, a per-axis bound of
262,144, total Tensor numel at most 16,777,216, and at most 1,000,000 active
elements per distribution. It uses one bounded growable owned pool and seven
object kinds. Kind-8 Optimizer state and the public-gradient-object successor
have not yet been composed into this particular ELF.

Other Tensor dtypes, `native_dropout`, default-Generator singleton behavior,
other devices/distributions, feature/alpha dropout, complete in-place autograd,
higher-order differentiation, Python frontend compatibility and all remaining
schemas retain their original goals. All RNG, dtype and full-rewrite completion
flags remain false. The 2,590-schema denominator and whole-project HOLD are
unchanged.
