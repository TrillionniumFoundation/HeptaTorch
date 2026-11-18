# Build the complete current HIA runtime

`tools/current_runtime/build.py` builds the complete runtime directly from its
published HIA source and ordered module manifest. It needs the fixed ABI3 HIA
compiler and session linker. It does not load a parent executable, reconstruct
the sequence of historical authoring overlays, or reuse a compiled application
object.

The current source package is `tools/current_runtime/source/`. It contains the
701-function F32-sum and empty-cat-gradient successor, its fourteen compilation units and
`manifest.json`. Tensor arithmetic, allocation, gradients and session execution
remain HIA implementations. Python partitions source, invokes native build
tools and checks their output; it does not execute Tensor operations.

## Source checks

This command checks every source hash, the complete function-body partition,
exports, imported signatures/effects, module dependencies and fixed unit limits.
It starts no subprocesses and needs no compiler installation:

```sh
python3 -I -S -B tools/current_runtime/build.py \
  --manifest tools/current_runtime/source/manifest.json \
  --source-root tools/current_runtime/source \
  --check-only --output build/current-source-check
```

The output directory must be new. A successful check reports
`SOURCE_CHECKED_NATIVE_NOT_RUN`; it is not a native compilation or runtime test.

## Cold native build

The default Make target selects this current profile:

```sh
make HEPTA_CURRENT_COMPILER=/absolute/path/to/stack-module-compiler \
     HEPTA_CURRENT_LINKER=/absolute/path/to/stack-linker-session
```

GNU Make 4.3 or newer is required for the paired build-report/executable target.
The final tools otherwise default to `HEPTA_STACK_BUILD/native/`. Make compiles
in a new staging directory, publishes only a successful complete build and keeps
any recognized previous build in that staging directory. Failed builds retain
their original reports. Deleting either generated target causes both to rebuild.

The underlying standalone build command is:

```sh
python3 -I -S -B tools/current_runtime/build.py \
  --manifest tools/current_runtime/source/manifest.json \
  --source-root tools/current_runtime/source \
  --compiler /absolute/path/to/stack-module-compiler \
  --linker /absolute/path/to/stack-linker-session \
  --output build/current-runtime
```

The builder reads the exact tool bytes, checks their SHA-256 values and uses
private executable copies for this build. Every `.o` is produced in a new
compiler process from its full `.hepta` unit. The final `heptatorch` executable is
produced by the native HIA linker using `HIASTKS1`, entry `step`, and zero startup
workspace. The production `step` and `close` ownership contract is unchanged.
The builder does not run the resulting executable.

The direct builder first cold-compiled the following three historical profiles. R4 and
Integral699 reproduced every original object and the original executable SHA;
Bool700 produced its first native image from the new complete source. All
compiler/linker processes exited zero with complete stdout/stderr EOF.

| Profile | Core functions / units | Native executable SHA-256 |
| --- | ---: | --- |
| R4 Scalar07b | 658 / 13 | `0799496559ffdfb051e34bc7a543db238b81d09c970a5cb4a8235151e0335b74` |
| Integral699 | 699 / 14 | `fec4f5cd7f80591fee98378cb5a8a85b6e71c5e4a7f0a00a739dad16afc24c28` |
| Bool700 | 700 / 14 | `c6afeedc14ba8cc408a62d6fbef0abb5b78dff0039b8099f3449d813e290a90e` |
| Current F32-sum / empty-cat | 701 / 14 | `80445a8c433f20314d3bc7b00ebb68f6a70fa200ee2e6aba193cff21aabfbc17` |

Those three builds each measured a 131,824-byte maximum native call path and
135,920 bytes including the existing reserve. These are build results; the
builder did not execute any of the Tensor applications. Eight separate source
controls also rejected changed core bodies, changed support code, missing
units/exports, mismatched imported effects, cyclic imports, path traversal and
an uncorrected source hash.

| Tool | Required SHA-256 |
| --- | --- |
| ABI3 compiler | `b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f` |
| ABI3 session linker | `aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16` |

The original [stack profile](../tools/stack_budget/profile.json) and
[ABI3 contract](../tools/stack_budget/CONTRACT.md) remain authoritative for tool
reproduction. Initial reproduction of those tools still uses their existing
bootstrap recipe. A normal application build only needs the two final tools;
it does not need the older Tensor executables.

The compiler retains its 1 MiB source/object limit, 256 function slots including
imports, 2,560 blocks, 256 blocks per function and 8,192 instructions per unit.
The linker retains its 64-object and 8 MiB input/output limits. Its native
interface, ownership, machine-call and stack checks are preserved. The usable
private stack is 8 MiB with guard pages; admitted paths are at most 8,384,512
bytes, leaving the unchanged 4,096-byte reserve. The pinned independent stack
inspector additionally compares the new object metadata and decoded frames
against the actual source call graph before linking.

`build/current-runtime/build.json` records source and tool identities, each new
object, the final image, stack demand and the exit status and EOF of every
compiler/linker process. Successful compilation reports
`PASS_NATIVE_BUILD_ONLY`. Runtime and PyTorch comparison suites must bind this
exact source, executable and report before granting their own qualification.
The full-project [acceptance ledger](../compatibility/full-acceptance.json)
continues to track the separate whole-PyTorch target.

## Current-image runtime qualification

```sh
make test HEPTA_CURRENT_COMPILER=/absolute/path/to/stack-module-compiler \
          HEPTA_CURRENT_LINKER=/absolute/path/to/stack-linker-session \
          HEPTA_TORCH_PYTHON=/absolute/path/to/pinned-env/bin/python
```

Make selects a new qualification directory. The equivalent standalone command
is:

```sh
python3 -I -S -B tests/qualify_current_runtime.py \
  --build-directory build/current-runtime \
  --torch-python /absolute/path/to/pinned-env/bin/python \
  --output build/current-runtime-qualification
```

The output directory must not exist. The explicit Torch interpreter must be the
fixed CPython 3.12 Linux x86-64 environment described by
`compatibility/pytorch-oracle-lock.json`. Its original pip installation receipt
must exist at `build/pytorch-install.json`. The workflow prepares that exact
environment; the local gate does not install packages or access the network.
The interpreter's original path is used to preserve virtual-environment
selection; its resolved executable bytes are also bound as an input.

The same coordinator runs seven mandatory groups against one complete source,
image and actual cold-build receipt: Bool27, original view/scalar76, Integral24,
F32 Sum32, Bool/I64-to-F32 Sum17, original live35, and original live270. Their
case total is 481. Extra event, frame, numeric, lifetime, RNG and control counts
retain their own denominators.

The five recorded-reference groups (Bool27, original76, Integral24, Sum32 and
Sum17) use exact frozen independent reference observations.
The original35 and original270 run the fixed real Torch oracle and all eleven
provenance checks. All original numerical inputs, expectations, tolerances,
training loops, exception phases and comparison bodies remain fixed apart from
the documented complete, reversible transport/identity adaptations.

The original270 bridge keeps 256 complete successful requests. Its fourteen
refusals use the same process for the genuine success prefix and one genuine
terminal refusal frame because the current owner rolls back a failed complete
frame and returns count0. It retains both actual replies; it does not synthesize
an old-format error record. Qualification requires 270 new native processes,
284 frames, the exact actual executable, every exit0, empty native stderr and
both actual EOFs. The original35 additionally executes its strict report checker
and the unchanged four positive/eighteen negative copied-observation controls.

The old Bool27 dtype controls still report FAIL at indices2/3 after F32 output
is implemented. The old Integral24 controls still report FAIL at indices0/1/2/3
after Bool and F32 output are implemented. Their original reports and nonzero
reader exits are retained. A strict outer gate requires every original numeric
case, unchanged refusal, exact expected extension and final native EOF; the new
Sum17 controls independently qualify the extended contract.

Each runner's complete stdout/stderr, hashes, exit and EOF are retained. The
supervisor tracks exact process incarnations and sends cleanup signals through
pidfds; it does not authorize a signal using a recycled PID or session number.
The original SID-reuse counterexample and corrected pure-Python cleanup checks
are retained. A timeout, prepared-only/missing report, incomplete stream,
changed input or ordinary case failure blocks the overall focused PASS, while
the remaining groups still run.

The focused workflow has a 180-minute job budget, archives complete available
build/source/reference/native evidence, and reports whole-project HOLD even
when `PASS_CURRENT_RUNTIME_FOCUSED` is obtained. Actual source-bound results
are indexed in `validation/current-runtime701/evidence.json`; historical
qualification is never copied to a changed executable.

## A single complete source, reproducible units

The aggregate `runtime.hepta` is the authoritative collection of core function
bodies. `partition.py` reproduces module sources without invoking the old
authoring stack:

```sh
python3 -I -S -B tools/current_runtime/partition.py \
  --runtime tools/current_runtime/source/runtime.hepta \
  --rng tools/current_runtime/source/units/rng03.hepta \
  --owners tools/current_runtime/source/units/owners.hepta \
  --name sum-f32-cat-empty-01 --prefix cat_f32 \
  --output build/current-repartitioned
```

Dependency traversal, name sorting, 65-function / 3,600-instruction /
1,000-block grouping and declaration generation follow the existing R4 and
Integral699 partition algorithm. These grouping thresholds are more
conservative than the unchanged native compiler limits. Every complete core
function occurs once in the partition and must match the aggregate bytes.
Private RNG helpers remain scoped to the fixed RNG unit. The two support unit
identities are fixed in the builder; changing their ownership or RNG contract
requires a reviewed profile change.

When reproducing an already built profile, `--expected-build path/to/build.json`
requires the original aggregate SHA, all unit names and source SHAs in order,
the same native tools and successful link status. It carries the original
object and executable SHAs into the new manifest. The cold build then requires
every newly compiled object and linked executable to match those identities.
No historical object or application ELF is read or executed.

The R4 comparison manifest at
`tools/current_runtime/profiles/r4-scalar07b.json` exercises this same direct
builder against the existing 658-function source package. It references the
already published source and original output identities without copying them.
Use the repository root as `--source-root` for that manifest.

## Updating the current source

Apply an implementation change to the complete runtime, generate fresh units
with `partition.py`, review the source/manifest changes and run the cold builder
in a new directory. A first build of a new source has no claimed historical
object or ELF hash; its actual outputs are bound by the new build report.
Functional tests use independently fixed reference results and execute that
same image. Updating a source manifest never transfers the earlier image's
runtime qualification to the changed image.

This direct source path removes historical application builds from the current
build dependency graph. It does not by itself authorize deleting older source
packages, original failure records or test inputs that other acceptance suites
still reference. Those references can be consolidated after the current
runtime passes their relevant regression suites.
