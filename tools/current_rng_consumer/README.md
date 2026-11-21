# Current RNG object consumer

This consumer runs the complete original CPU RNG packet qualification against
the **actual `rng03.o` produced by a complete current-runtime cold build**.
It does not restore `src/heptatorch.hepta`, compile the old eighteen-function
standalone RNG runtime, or execute an old application image.

The current runtime source, image, function count and partitions come from
`tools/current_runtime/source/manifest.json`. Before building or reading any
result, the consumer calls the existing
`tools/current_runtime/qualification.py:admit_build` on the entire cold build.
The RNG source/object and the packet compatibility contract retain exact pins.
A stale current build is refused even when its RNG object happens to match.

## Native implementation

The original `tools/rng-cpu/rng.hepta` contains eighteen functions. Its seventeen
non-entry function bodies are byte-identical to the current RNG support unit.
The consumer compiles only the original `entry`, `getr`, `getw`, `putw` and
`rng_error` bodies, with eight exact external declarations. It links that
five-function packet adapter with the **existing current build's object**.
No host numeric kernel or host RNG is linked into the executable.

The module compiler and flat linker are fixed native HIA tools. Two adapter
compilations and two native links must be byte-identical. The independent
source/object stack inspector checks the complete imported call graph. The
resulting entry uses ABI3 argument/result arrays containing the same six
borrowed input, output and workspace words as the old flat contract.

The adapter source hash is
`f94d146b56da4c5d2b93272ee8c335a6c6a664ed26805acaac8df958aa4b8432`.
The reused RNG object hash is
`d462d241e7c3082167b61ccb7b4e6a60eeb94c242dbb69e242fd57b822b9f2a1`.
The consumer ELF hash is
`084c855df6fa5bd387d051228744a9ded60a598334f1c237ae61d4759ad782a6`.
Its entry offset is 28,752 and complete native stack path is 19,952 bytes.
These identify a compatibility entry over the current object; they do not
identify the full current-runtime ELF.

## Complete retained qualification

| Original suite | Cases | Retained checks |
| --- | ---: | --- |
| CPython and C++ MT19937 | 74 | 110 packets, 718,619 exact values, 434,816 state bytes; fifteen independent engine seed proofs |
| Pinned PyTorch CPU | 76 | 114 packets, 719,337 exact values, 455,040 state bytes; all original eleven provenance checks and both actual normal-cache imports |
| Flat borrowed-buffer boundary | 10 | All original capacities, return values, input preservation, output extent and workspace atomicity |
| Independent state/admission review | 25 | Inverse tempering, rounding/clamping, state canonicalization, consumption thresholds, 23 actual stdin packets and fifteen atomicity/limit observations |

These are **185 original cases**, including the two deliberately separate
reference suites. They are not 185 unique PyTorch operators. The raw reader
binds all 247 actual stdin request/response pairs and 25 direct borrowed-buffer
observations. The complete original historical 35-case row payload must remain
identical. The original frozen independent package, including its ten former
failure records, is also preserved unchanged.

`prepare.py` applies only explicit build, identity, ABI and evidence changes to
the original comparator files. Every generated file must invert to the exact
original bytes. The entire numerical/state/rejection sections remain unchanged.
`prepare_custody.py` redirects the six original source143 helper comparisons to
identical current RNG bodies; it keeps all eight body hashes, both original
dtype comparisons, six licensed source files and MT19937 notice checks.

The qualification reader regenerates those precise permitted adaptations as
data. It rechecks raw packets, complete output/workspace/input buffers, actual
ABI returns, source inverses and process receipts. It executes no native image,
C++ reference or Torch operation. All original C++ reference executables,
generated comparators, source snapshots, compiler/linker inputs and outputs,
raw packets, physical observations and process streams remain in the output.

### Process and reference custody

Each of the eight Python child processes has a saved pre-spawn intent, a
supervisor result with both complete streams, and a receipt written by the
actual child. The reader independently reconstructs the exact argv, repository
working directory, interpreter bytes and mode, Python flags, parent/child PIDs,
timeout, and complete explicit environment. Children receive only the recorded
PATH, locale, timezone, home and evidence destinations; Python/loader overrides
and unrelated inherited environment variables are excluded. All four native
compiler/linker processes similarly retain exact command, working directory,
environment, executable identity, input, output, error stream, PID and EOF.
Original producer paths remain labels in the archive: moving the evidence does
not rewrite the original commands or require recreating the producer's paths.

The PyTorch child receives an explicit `TORCHINDUCTOR_CACHE_DIR` under its new
evidence directory before it starts. The original strict oracle imports Torch
compiler support, which otherwise adds this variable after startup. Recording
the path in the initial intent keeps the complete startup, child and reference
environments equal. The reader still compares every environment field; the
earlier strict CI rejection and its original process records remain preserved.

The actual Torch child saves the interpreter, imported package and extension,
version source, distribution METADATA/WHEEL/RECORD/INSTALLER, PEP610 direct URL,
original pip install report, fixed source and locks, and complete import
streams. The pure reader checks these raw entities, the distribution RECORD,
the pinned official METADATA hash, and their relationship to the actual process.
It extracts the eleven CHECKS expressions from the byte-pinned original oracle
and evaluates those expressions using the reconstructed raw-input facts. It
requires the exact eleven original names, booleans and results; an arbitrary
set of eleven true flags is not accepted. Missing original pip or PEP610 data
cannot become production provenance by changing a report's status or flags.
The fixed NumPy support-install receipt is also checked against its requirement.

The same actual reference process also records all 12,567 immutable installed
wheel members before and after its numerical work. An independent fixed member
index and the complete original official wheel anchor these 751,277,549 bytes,
including twelve computation libraries. The pure reader checks the complete
shared snapshot, the actual mapped library paths, devices, inodes and executable
segments, and the connection to the original eleven-check reference process.
Changing an installed file together with its RECORD and receipt hashes cannot
replace the official member identity. Foreign or deleted Torch libraries,
invalid mappings, a different process, and omitted members are rejected.

The shared wheel, independent member index and every immutable installed member
form exactly 12,569 declared reference inputs. Their complete byte identities
and modes are checked again after all four suites. The original raw eleven-check
reader and every original numerical comparison remain unchanged.

All twenty-five direct calls bind their requests to the original case packet,
including the three independent short-capacity/header literals. Every saved
output and workspace must contain exactly 8,192 bytes, every return exactly
four bytes, and the complete input before/after must equal the original request.
The successful 5,080-byte flat boundary additionally binds all 5,056 written
workspace bytes to the returned state and retains the untouched suffix; all
other workspaces retain their entire original sentinel contents.

The running manifest starts with `same_current_rng_object_executed=false`.
Only completion of all four actual suites and the source/build checks permits
that flag to become true. Production qualification remains false on a failed
run or any explicitly exploratory reference run.

## Run

The caller supplies a successful complete current cold build and the fixed
native stack-runtime linker. The normal invocation uses the strict original
PyTorch source/installation checks:

```bash
python3 -B tests/pinned_reference_custody.py \
  --prepare-installed --repository .

python3 -I -S -B tools/current_rng_consumer/run.py \
  --repository . \
  --current-build "$CURRENT_BUILD" \
  --flat-linker "$STACK_BUILD/native/stack-linker-runtime" \
  --output "$RNG_REVIEW"

python3 -I -S -B tools/current_rng_consumer/qualify.py \
  --repository . \
  --current-build "$CURRENT_BUILD" \
  --evidence "$RNG_REVIEW" \
  --output "$RNG_QUALIFICATION"
```

The dedicated `rng-cpu.yml` workflow rebuilds the existing native toolchain,
cold-builds the current manifest, freshly installs the exact pinned Torch wheel,
runs all four original suites, reruns the pure reader, and preserves complete
consumer and native tool evidence. Install the fixed Torch and NumPy wheels with
their actual pip reports before preparing the shared snapshot; see the
[reference environment instructions](../../docs/current-runtime.md#independent-reference-environment).
Retain the original official wheel, shared installed snapshot, fixed member
index and shared reader in the same artifact as the consumer evidence.
The workflow never restores historical143.

For explicitly exploratory local work, `run.py --local-observed-oracle` records
the observed installed Torch identity without inventing a pip report. Such a
result is `PASS_EXPLORATORY_NATIVE_ONLY`; the default reader refuses it.
`qualify.py --allow-unverified-local` permits inspection of that evidence while
keeping both production qualification and complete installation provenance
false. Neither local option appears in the CI workflow.

## Relationship to Generator131 and remaining scope

The current owned Generator consumer shares the same RNG kernels and covers
handle creation, clones, releases, ownership, entropy entry and dispatcher
behavior. Its 24 Torch scenarios plus 11 native, 59 entry and 37 dispatcher
cases do not replace this packet protocol's complete 74/76/10/25 comparisons.
The packet suite adds its own batching, exact 100,000-value and 1,000,000-word
limits, all ten output capacities, full-packet rollback, independent injected
state/rounding boundaries, and explicit malformed-packet checks. Both consumers
remain required for their different contracts.

| Packet operation | Owned Generator operation | Shared behavior |
| ---: | ---: | --- |
| 0 | 9 | Raw 32-bit words |
| 1 | 10 | Raw 64-bit words |
| 2 | 11 | Unit F32 values |
| 3 | 12 | Unit F64 values |
| 4 | 15 | Signed I64 range |
| 5 | 13 | Unit F16 values |
| 6 | 14 | Unit BF16 values |
| 7 | 2 | Reseed |
| 8 | 8 | Discard raw 32-bit words |

This migration establishes neither normal-distribution generation/cache
consumption nor complete distributions, devices, tensor RNG or PyTorch RNG
compatibility. It does not authorize deleting still-used historical consumers
or their independent reference packages. Full project closure remains governed
by the repository's complete acceptance contracts.
