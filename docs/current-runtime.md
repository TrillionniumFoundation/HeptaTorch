# Build and qualify the current HIA runtime

The canonical application is `tools/current_runtime/source/`: one complete
712-function HIA program, its ordered 15-unit manifest and current support units.
The profile name is `tensor-generator-native01`. Tensor arithmetic, allocation,
autograd, RNG, persistent Generator operations and session dispatch execute in
HIA. Python organizes builds, records independent observations and checks evidence.

A fresh execution has passed all nine current groups and all 644 cases on the
same actual source/image/build. Independent readback verified 65,627 bound inputs
and every runner exit, both EOFs and empty stderr. The corresponding new
source-head hosted qualification remains **PENDING**. Exact reports, the original
retained failures and separately scoped component results are indexed in
[current-runtime712/evidence.json](../validation/current-runtime712/evidence.json).

## Source checks

The following checks every source SHA, complete function-body partition,
imported signature/effect, export, dependency and fixed unit limit. It starts no
native process and requires no compiler installation:

```sh
python3 -I -S -B tools/current_runtime/build.py \
  --manifest tools/current_runtime/source/manifest.json \
  --source-root tools/current_runtime/source \
  --check-only --output build/current-source-check
```

The output directory must be new. `SOURCE_CHECKED_NATIVE_NOT_RUN` means exactly
that; it is not a build or runtime qualification.

## Native tools and cold build

Reproduce the fixed stack-budget tools and retain the fixed Hepta authoring
checkout using [the toolchain recipe](build.md). Then build the current application:

```sh
make HEPTA_SOURCE_ROOT=/absolute/path/to/pinned/Hepta \
     HEPTA_STACK_BUILD=/absolute/path/to/stack-budget-tools
```

GNU Make 4.3 or newer is required for grouped report/executable targets.
`HEPTA_SOURCE_ROOT` must contain the pinned Hepta source contract and all 33
authoring inputs; the existing source identity checks remain mandatory.
`HEPTA_STACK_BUILD` must retain the complete fixed tool layout, including the
module compiler, tool linker, runtime linker, old three-argument session linker
and two original linker authoring files.

The `current-entropy-linker` prerequisite uses only the original native compiler
and tool linker to reproduce the four-argument entropy session linker. It does
not rebuild or execute the retired Generator application. Its report and full
raw streams are kept in the original staging directory, then copied to
`HEPTA_ENTROPY_BUILD`, which defaults to `build/current-runtime-entropy`.
Keep the original staging directory: `build.json.processes[].directory` binds
those absolute `input.bin`, `stdout.bin` and `stderr.bin` paths.

| Tool | Required SHA-256 |
| --- | --- |
| Module compiler | `b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f` |
| Current entropy session linker | `761a941fd7a2d53eb9b45869fcfc20e970640e9dd11d60f7c61a89812b61d74e` |
| Original tool linker, used for entropy bootstrap | `fa3065b5dc42bfe09fd29cf773c5adc5a9296c125a6a161daecf2fa885ea160b` |
| Original runtime linker, used by entry qualification | `0c5176e3aa968631ae592b887c5dc3cf1e8e3f48f73593f076e9d7965ffacfd7` |
| Original three-argument session linker, used by entry qualification | `aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16` |

The current build selects the new linker; the original three-argument linker is
retained as a distinct tool. Do not overwrite it with the entropy linker.
The existing [stack profile](../tools/stack_budget/profile.json) and
[ABI3 contract](../tools/stack_budget/CONTRACT.md) retain their original tool scope.

Equivalent explicit tool-only and application build commands are:

```sh
python3 -I -S -B tools/current_runtime/entropy_linker/build_tool.py \
  --upstream /absolute/path/to/pinned/Hepta \
  --stack-build /absolute/path/to/stack-budget-tools \
  --output build/current-runtime-entropy

python3 -I -S -B tools/current_runtime/build.py \
  --manifest tools/current_runtime/source/manifest.json \
  --source-root tools/current_runtime/source \
  --compiler /absolute/path/to/stack-budget-tools/native/stack-module-compiler \
  --linker build/current-runtime-entropy/generator-session-linker \
  --output build/current-runtime
```

Both output directories must be fresh. The builder verifies exact tool bytes,
makes private executable copies, compiles every full `.hepta` unit in a new
process and links all 15 actual objects. The current request is `HIACGS01`, entry
`step`, with the four-argument contract
`step(readbuf, mutbuf, buffer, readbuf) -> (buffer, u32)`. The final read buffer
carries trusted startup entropy. The runtime routes Tensor and Generator frames
to their native owners; it does not make their handles interchangeable.

`build.json` records all source/object/image identities, stack demand and the
exit and both EOFs of all 16 compiler/linker processes. It must state
`objects_reused=false`, `all_inputs_unchanged=true` and
`runtime_qualified=false`. Successful compilation is `PASS_NATIVE_BUILD_ONLY`;
the builder never executes the resulting Tensor/Generator application.

The original cold build produced source
`fac3d413c6c0f6874d224cbea312108ee2c8baa027102d41890d85094a5faa38`
and a 3,723,264-byte ELF with SHA
`5b7aac203d2c913d43781f0da45d40e9bac2037c2bc074ec30f8dd663710a805`.
The actual maximum native call path was 133,744 bytes, or 137,840 bytes with the
unchanged 4,096-byte reserve. These measurements describe that fixed build.

The compiler's original per-unit source/object, function, block and instruction
limits remain checked. The new independent inspector reads actual object frames
and verifies the complete source dependency graph before linking. Historical
builds and failures retain their own source/image identities in their evidence
indexes; they are not inputs to the current application build.

## One fixed interpreter and one qualification entry

Prepare the CPython 3.12 Linux x86-64 environment in the
[oracle lock](../compatibility/pytorch-oracle-lock.json). The real installation's
original pip report must be `build/pytorch-install.json`. A historical fixture's
pip report cannot substitute for this installation. The local test target does
not install packages or access the network.

```sh
make test HEPTA_SOURCE_ROOT=/absolute/path/to/pinned/Hepta \
          HEPTA_STACK_BUILD=/absolute/path/to/stack-budget-tools \
          HEPTA_TORCH_PYTHON=/absolute/path/to/pinned-env/bin/python
```

Make selects a fresh evidence directory outside the repository. To reuse an
already completed current cold build and tool build, invoke the coordinator
directly with explicit locations:

```sh
/absolute/path/to/pinned-env/bin/python -I -S -B tests/qualify_current_runtime.py \
  --build-directory /absolute/path/to/current-runtime-build \
  --torch-python /absolute/path/to/pinned-env/bin/python \
  --generator-tool-build /absolute/path/to/current-entropy-build \
  --stack-build /absolute/path/to/stack-budget-tools \
  --output /absolute/path/to/new-evidence-directory
```

These are the coordinator's five CLI parameters. The output must not exist and
must neither contain nor lie within the repository or an input build tree.
The coordinator reads each actual build receipt's SHA and passes it to every
reader. There is no reuse-report, group-filter or prepare-only route to overall
qualification. Existing cold receipts also retain their original absolute
`inputs` paths; moving an ELF alone does not preserve admission.

Before reading the build or starting a suite, the coordinator requires CPython
3.12, `cpython-312`, the same resolved executable as `--torch-python`, enabled
assertions and `-I -S -B`. Live readers use that selected interpreter with site
packages enabled. This preserves the exact AST serialization used in the source
inverse proofs. It does not remove a digest comparison or accept a cross-version
hash whitelist.

## Nine mandatory groups

| Execution order | Cases | Timeout budget |
| --- | ---: | ---: |
| `bool27` | 27 | 600 s |
| `original76` | 76 | 900 s |
| `integral24` | 24 | 900 s |
| `sum_f32_17` | 17 | 600 s |
| `current35` | 35 | 900 s |
| `base270` | 270 | 900 s |
| `live32` | 32 | 900 s |
| `generator131` | 131 | 600 s |
| `sum32` | 32 | 3,600 s |
| Total | **644** | **9,900 s / 165 min** |

The workflow budget is 210 minutes. Budgets are timeout limits, not observed
execution times or evidence of qualification. The slow Sum32 group runs last;
an ordinary group failure is recorded while later independent groups are still
attempted. Interpreter or source preflight failures stop before any case.

The original 513 numerical cases and all original controls remain complete.
Five groups use exact recorded independent reference observations. The original
76 fixture contains only its eight required original reference/probe files,
retaining their bytes and modes. Original live35, live270 and live32 use real
fixed Torch and all eleven provenance checks. The full original comparison
bodies, numeric tolerances, training loops and exception phases are preserved
with explicit reversible source/identity/transport adaptations.

Generator131 consists of original Torch24, native11, entry59 and 37 new dispatch
controls. It requires 37 actual persistent children, 102 actual tool/probe
processes and the independent C++ reference processes. The entry harness builds
fresh probes and a reordered image using every current object; reuse of the
completed primary cold build does not skip these required real tool calls.
Linux `/proc`, the original resource/entry controls and a usable `g++` are
required, not optional skips.

The coordinator restores the following into its fresh output:

| Input | Output subtree | Exact custody |
| --- | --- | --- |
| Original Generator26 fixture | `references/generator-original26` | Original 26 members and original SHA manifest, including original modes |
| Candidate metadata + 22 current canonical files | `references/generator-candidate` | Complete original 31-payload package and manifest; no permanent second implementation |
| Portable Generator source04 | `generator131-source` | Whole-source inverse to source03; only six physical-root/CLI edits |

The original26 source and package manifest are both read from the explicit
restored root. Actual Torch oracle/support/lock and the current installation
report remain repository inputs. Portable source04 retains every source03
numeric comparison, all original 94 cases and the corrected 37-control contract.

Old Bool dtype indices 2/3 and Integral dtype indices 0/1/2/3 retain their
original FAIL reports for explicitly added Bool/F32 sum outputs. The unchanged
remaining refusals, all original numeric cases and new dtype/entry/OOM controls
must pass the strict outer classification. A historical report is never edited
to create PASS.

Each runner retains complete stdout/stderr, hashes, exits and EOFs. Cleanup
uses exact process incarnations and pidfds. Missing reports, prepared-only
reports, timeouts, forced cleanup, incomplete EOF or changed inputs prevent
`PASS_CURRENT_RUNTIME_FOCUSED`. The known fixed NumPy warning allowance is
limited to the independently bound live35 Torch source and exact single-warning
shape; all native stderr and other runner contracts remain strict.

## One canonical source and reproducible partition

The aggregate contains the original 701 Tensor function bodies unchanged plus
11 Generator bodies. The original twelve `cat_f32_*` Tensor units remain exact.
The support units implement the combined native entry and fixed Generator
contract; the original source inverse does not transfer runtime qualification.

Reproduce the current manifest and all units with:

```sh
python3 -I -S -B tools/current_runtime/partition.py \
  --runtime tools/current_runtime/source/runtime.hepta \
  --rng tools/current_runtime/source/units/rng03.hepta \
  --owners tools/current_runtime/source/units/owners.hepta \
  --generator tools/current_runtime/source/units/cg_model.hepta \
  --name tensor-generator-native01 --prefix cat_f32 \
  --output build/current-repartitioned
```

The generator-aware path checks the entire inverse source, the exact new model,
all 15 unit names and the original current manifest SHA. It emits a source-only
partition receipt. `--expected-build` additionally binds original object/image
identities when reproducing an already built profile. It never reads an old
application ELF as a substitute for compiling current objects.

## Evidence and remaining scope

[The current evidence index](../validation/current-runtime712/evidence.json)
distinguishes completed component executions, preserved failures, pure readback
the completed current644 gate and the still-pending source-head hosted gate. It also links the predecessor's
source-bound qualification; no predecessor PASS is copied to this image.

The complete original Generator26 package and checker are now retained in fixed
fixtures. Its nineteen old source paths, old checker and standalone manifest
have no active default-build role. Two duplicate old Generator CI steps and old
Make targets are replaced by the mandatory current group. The fixed toolchain,
original Torch entry and independent legacy consumers remain. See the
[retirement index](../validation/current-generator-original26/retirement.json).

F64 and later combined candidates remain separate source/image qualifications.
The 17 whole-system gates remain open in
[full-acceptance.json](../compatibility/full-acceptance.json), after this
finite focused suite has passed.
