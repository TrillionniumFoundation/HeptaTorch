# Reproducible native build and historical toolchain profiles

The default `make` and `make test` now build and qualify the complete current
700-function, fourteen-unit HIA runtime. Start with [current-runtime.md](current-runtime.md)
for that path. This page preserves the original compiler bootstrap and historical
application recipes used by the separate legacy regression workflow.
The 143-function source and limits in the first sections describe that historical
baseline, not the current default runtime.

The large frozen Adam/distribution observation archive is stored as two
byte-exact parts so each connector write fits its request-size limit.
`tools/restore_frozen_archive.py` verifies both parts and reconstructs the original
15,066,536-byte gzip file with SHA-256
`0ec89d5c56f5bd3a39db09bf3a9ed56a2c13e759648327f8388488ec0f25a907`.
The native Make targets restore it before checking the unchanged frozen package
manifest. Direct package-manifest checks need the same command first:
`python3 -I -S -B tools/restore_frozen_archive.py --manifest validation/adam-distributions/evidence/archive-parts.json`.
Original archive bytes, members and frozen manifests are preserved. The restored
file is ignored by Git because its exact contents are represented by the tracked
parts and their manifest.

## Complete hosted report export

The native workflow exports the original top-level `build/*.json` report files
as an Actions artifact, in addition to the existing compressed log envelopes.
`ci-report-manifest.json` records each original file's byte length and SHA-256,
the checkout commit and tree, the workflow and collector hashes, and the actual
run, attempt, event and source-head identity. Collection never parses or
reserializes report contents, including unsigned 64-bit state observations.
A successful job must contain every report named by the existing envelope
steps and have an unchanged tracked checkout. Failed jobs can export partial
evidence, explicitly marked with the prior job status and missing report names.
Artifact transport does not qualify a report or change any numerical gate.

## Target and toolchain

The executable target is Linux x86-64. The historical 143-function implementation is
`src/heptatorch.hepta`, using HIA primitive Core v8 with semantic function names
and explicit typed SSA instructions. `docs/functions.json` indexes the native
functions. The complete current source is instead published under
`tools/current_runtime/source/`.

The `legacy` build partitions those function bodies into two independent HIA
modules, compiles each with the native module compiler, and admits and links
their objects with the native HIA linker. The partitioner preserves every body
byte; it does not generate numerical implementations. A separate monolithic
reference build emits the same authoritative source with `selfhost_exec`.

| Input | Pinned identity |
| --- | --- |
| HIA repository | `TrillionniumFoundation/Hepta` |
| HIA commit | `f33d8da316de8fdf27b504371f1a5310997ff548` |
| Original compiler source | `examples/hia_bootstrap/selfhost.hepta` |
| Original compiler source SHA-256 | `388b182e42675f11435867dd99520e9452a4ff48e23c0766a3952b662acc7627` |
| Original bootstrap compiler SHA-256 | `87c247e4d1c3c9a449d2947fd4227fdc186d519bffd203a04cafce7861a4ed54` |
| Expanded compiler profile | `heptatorch.hia.compiler.f256-b2560-w2m.v1` |
| Expanded compiler HIA SHA-256 | `5cf8a4286d0759d2368e2d8d5658ce7de0b2ad72435ce75915cf07c0a23df7bf` |
| Expanded ordinary compiler SHA-256 | `c5a9c2b95a3061e080a21298f5b26000936ed9dac519690d845b0699a15744c6` |
| Expanded ordinary compiler size | 674,085 bytes |
| Module profile | `heptatorch.hia.borrowed-modules.v1` |
| Native module compiler SHA-256 | `e250d93bc0b2a5ff878d2188c95780abf7e7ebaf186e4531b136fa2735b2af59` |
| Native module linker SHA-256 | `1ba40ffe025abfcc2edd8d69f1ad689817598571138c11806af7aa60047fc7b8` |

The Makefile checks each selected tool's hash before invoking it. The module
compiler emits actual ELF objects with imports, exports and `.hia.interface`
metadata. The module linker performs interface admission, symbol resolution
and relocations and emits a syscall-only executable. The ordinary reference
compiler accepts `source.hepta` on stdin and writes a complete executable on
stdout. Neither runtime product uses `PT_INTERP`, a dynamic loader, libc,
Python or tensor-library calls.

The original compiler accepts 128 functions and 1,536 basic blocks. The
expanded ordinary and borrowed-module profiles admit 256 functions and 2,560
total blocks per compilation unit, with unchanged 8,192-instruction, 1 MiB
source and 1 MiB generated-image limits. External declarations also count
toward a module's function limit. This historical baseline has 143
functions, 1,783 blocks and 7,064 instructions in 481,516 source bytes.
Its partition has 63 kernel functions and 80 runtime functions, with 52
declared imports from runtime to kernel.

| Historical baseline artifact | SHA-256 | Bytes |
| --- | --- | ---: |
| Authoritative HIA source | `264e9c5c4e8ea1834724cc0d8d859952016daffe4af330a8daa7745e7a5e5233` | 481,516 |
| Legacy linked runtime | `cf78556c3c512227ffe123635b76e5a6af97697606e8ba3535c56f5101f4f80d` | 733,184 |
| Monolithic reference runtime | `5b992e5f07fe1167aacf31d9fd07a053b3b02fc1bd1149c06f8ccd164ef96bb0` | 729,233 |

The two runtime files have different ELF organization and hashes. Their packet
behavior is compared against the same independently qualified fixture corpus.
The expanded and module compilers, both in ordinary mode, emit byte-identical
current monolithic runtime and unhex outputs. The original 128-function
compiler cannot admit the complete current source; its historical comparison
is defined below. These local profiles remain distinct from the unchanged
[upstream support profile](https://github.com/TrillionniumFoundation/Hepta/blob/f33d8da316de8fdf27b504371f1a5310997ff548/docs/HIA_SELFHOST_CORE_SUPPORT.json).

## Starting with existing native tools

```sh
make HEPTA_COMPILER=/absolute/path/to/hia-native-compiler \
     HEPTA_MODULE_COMPILER=/absolute/path/to/hia-module-compiler \
     HEPTA_MODULE_LINKER=/absolute/path/to/hia-module-linker \
     legacy test-legacy
```

`HEPTA_COMPILER` names the expanded ordinary compiler; it builds unhex and the
monolithic reference. `HEPTA_MODULE_COMPILER` and `HEPTA_MODULE_LINKER` name the
two module tools. `legacy` produces the historical linked `build/heptatorch` and
`build/unhex`. Each variable can be an absolute path or a command resolved by
the build environment; the selected bytes must match the corresponding pin.

Build and run the same regression corpus through the monolithic reference:

```sh
make HEPTA_COMPILER=/absolute/path/to/hia-native-compiler \
     monolithic test-monolithic
```

This produces `build/heptatorch-monolithic`; it does not replace the default
linked executable. Build files are written through temporary files and atomic
rename. Module partitioning and packet/file transport use Python 3 and shell
build tools, while compilation, admission, relocation and execution are native
HIA. The resulting executables do not load those host build tools.

## Starting from the pinned HIA source

The workflow in `.github/workflows/native.yml` defines the complete build and
qualification pipeline. It checks out the exact upstream commit with a narrow
source selection and uses the upstream `build_seed` and `compile_stage` routines:

1. The upstream Python assembler/verifier/direct emitter and GNU ld produce B0.
2. B0 executes the HIA compiler source to produce B1.
3. B1 produces B2, and B2 produces B3; their machine image bytes must agree.
4. B3 compiles the same HIA source with the `selfhost_exec` header.
5. The resulting native executable rebuilds itself twice with an empty PATH;
   both outputs must match the pinned original compiler bytes.
6. Repository adapters verify the exact upstream import closure, reproduce the
   original HIA source, and generate the separate capacity profile and its HIA
   executable packer under build scratch.
7. The original native compiler produces the expanded compiler's flat image.
   A native HIA packer supplies its 2 MiB compiler workspace. Four native
   compiler/image generations must match exactly; the successor reproduces
   the packer. All compiler and packer hashes are checked.
8. The expanded compiler passes the 16-case capacity gate. Original/expanded
   code generation is compared on the pinned historical runtime witness below.
9. Module adapters verify their fixed upstream closures and source pins. The
   expanded compiler bootstraps the module compiler and linker; three module
   compiler generations and both HIA packers must reproduce identically.
10. Both runtime paths use the same 461 fixture pairs. Ordinary-mode output
    from expanded/module compilers must agree on current runtime and unhex
    source. Module contract, relocation, resource and guarded-buffer gates run
    independently of application regression tests.
11. The same 268-case real-PyTorch oracle is configured for each runtime image.
    Fresh hosted receipts must record actual results and both identities;
    configuring this gate does not assert that it has already passed.

The initial Python bootstrap belongs to the external language toolchain's
trust boundary. Source partitioning and build transport also use host Python,
but do not execute tensor operations or replace a native compiler/linker gate.
No Python source file is loaded by the HeptaTorch executable. These finite
bootstrap checks are not the upstream project's separate full cold-qualification
campaign or a formal compiler-correctness proof. Capacity/module authoring uses
pinned upstream generators and the unchanged Core verifier; native successor
compilation and ELF packing use HIA programs with an empty PATH.

Fresh local bootstraps from a clean source checkout and a narrow source export
previously reproduced the same original compiler hash. The sparse checkout
includes root `pyproject.toml`, used by the upstream compiler as a source-root
marker. Historical receipts retain the source they actually exercised.

## Reproducing the capacity profile from the original compiler

After the upstream bootstrap has produced the original pinned compiler, run:

```sh
python3 -I -S -B tools/compiler-profile/adapt.py \
  --source-root /absolute/pinned-hepta-checkout \
  --output /absolute/build/generated-profile
python3 -I -S -B tools/compiler-profile/generate_packer.py \
  --source-root /absolute/pinned-hepta-checkout \
  --output /absolute/build/generated-profile
bash tools/compiler-profile/rebuild.sh \
  /absolute/original-native-compiler \
  /absolute/build/generated-profile \
  /absolute/build/new-native-profile
python3 -I -S -B tools/compiler-profile/check_capacity.py \
  --compiler /absolute/build/new-native-profile/hia-native-compiler \
  --output /absolute/build/new-capacity-checks
```

The native-profile output directory must not already exist. Generated HIA
files are derived from the fixed external checkout and are not vendored in
this repository. `compatibility/hia-expanded-profile.json` records exact
products, limits and application wrapper invariants.
[Compiler-profile evidence](../validation/compiler-profile.json) retains the
original per-edit source receipt, table layout and independent native checks.

The expanded compiler has a 2 MiB workspace. Ordinary `selfhost_exec`
applications, including the monolithic runtime and unhex, keep their 1 MiB
input, output and workspace buffers. Per-function blocks (256), SSA names
(2,048), parameters (16), Core admission limits and scalar-only explicit exports
retain their existing contracts in this ordinary profile. The borrowed-module
profile adds its own explicit import/export contract; general ownership and
foreign APIs, and device code generation remain separate work.

## Reproducing the borrowed-module tools

First reproduce the expanded compiler and keep its generated `adaptation.json`
and `compiler-packer.hepta`. The capacity adaptation receipt must hash to
`1926437c96c413b891b4c0b61d1555434598a40067855241e0d2a53b1bee1292`.
The following commands use absolute caller-selected paths. The generated and
native output directories must be new; both source adapters write to the same
generated directory:

```sh
python3 -I -S -B tools/modules/build_module_compiler.py \
  --source-root /absolute/pinned-hepta-checkout \
  --capacity-adaptation /absolute/build/generated-profile/adaptation.json \
  --output /absolute/build/generated-modules
python3 -I -S -B tools/modules/build_module_linker.py \
  --source-root /absolute/pinned-hepta-checkout \
  --capacity-adaptation /absolute/build/generated-profile/adaptation.json \
  --output /absolute/build/generated-modules
cp /absolute/build/generated-profile/compiler-packer.hepta \
   /absolute/build/generated-modules/compiler-packer.hepta
bash tools/modules/rebuild.sh \
  /absolute/build/new-native-profile/hia-native-compiler \
  /absolute/build/generated-modules \
  /absolute/build/native-modules
```

The adapters verify the upstream source closure before and after imports,
reproduce unchanged baseline generation, and emit HIA with pinned hashes.
This host authoring stage does not claim native compilation has already
happened. `rebuild.sh` then compiles and packs with native HIA under an empty
PATH. It checks three identical compiler image/executable generations, both
packers and the exact final module compiler/linker hashes.

The resulting tools are `/absolute/build/native-modules/module-compiler` and
`/absolute/build/native-modules/module-linker`. Pass those paths as the two
Make variables above. [The module recipe](../tools/modules/README.md) supplies
exact direct split/compile/link commands and independent qualification commands.
[Its profile](../tools/modules/profile.json) defines 64 objects, 8 MiB linker
input/output buffers and 16 MiB linker workspace, separately from the compiler's
2 MiB workspace and the application's three 1 MiB buffers.

The native module boundary admits exact scalar types and borrowed `readbuf` /
`mutbuf` arguments, with zero or one scalar/unit result. It preserves native
signature, effect, alias and lifetime checks, and adds interface and module-DAG
admission before relocation/output publication. Owned resources, allocation,
resource returns, arbitrary foreign C resource entry points and a whole-program
stack-cost proof remain outside this profile. The
[contract](../tools/modules/CONTRACT.md) and
[native module evidence](../validation/native-modules.json) state those limits.

## Historical and current compiler comparisons

The original compiler hash beginning `87c247e4` is compared with the expanded
`c5a9c2b9` compiler on the runtime from commit
`31d5b1483714b32254730ba56787884c5c76e2fb`:

| Historical witness | SHA-256 |
| --- | --- |
| HIA source | `4c1d17240dd8616798e238cdd54641ee6f2e42c02635c82b3bdeea0a530b5ca5` |
| Executable | `ecca265b088e0688df8ad963613f6bdf460efd324d0c480e6395e4ef67372779` |

Those 121 functions fit the original profile. The historical 143-function source
does not; its expected old-compiler refusal is not a code-generation regression.
The current ordinary-mode comparison instead uses expanded `c5a9c2b9` and
module `e250d93b` compilers on the full current source and unhex. Historical
compiler/application receipts retain their original source/image identities
rather than being overwritten with current hashes.

## Test execution

`make test-legacy` decodes each of the 461 `tests/fixtures/*.input.hex` requests and its
expected response using `tools/unhex.hepta`, runs the native tensor executable
with an empty environment and PATH, and compares response bytes. The shell only
orchestrates processes, byte transport and file comparison. The hexadecimal
decoder itself is HIA. `make test-monolithic` runs this same collection through
`build/heptatorch-monolithic`. The legacy linked image has passed all 461 pairs;
source-bound validation records separately state each runtime's final replay.

Golden responses were captured only after independent numerical, analytical,
finite-difference or explicit-error checks. Their origins and measured
tolerances are retained in `validation/`. They are regression data, not a
substitute for the full upstream compatibility matrix.

The decoder's own input is bounded by its 1 MiB native input wrapper, so a
single decoder invocation accepts at most roughly 512 KiB of binary data in
hexadecimal text. A whole fixture may be larger. For text above that wrapper
limit, the runner removes only the four whitespace characters already admitted
by unhex (space, tab, carriage return and newline), divides the remaining text
into 524,288-character chunks at complete byte-pair boundaries, invokes the
same native HIA decoder on each chunk and concatenates the decoded bytes.
This is shell/file transport; fixture bytes, decoder code and tensor runtime
are unchanged. The tensor executable still accepts at most 1 MiB of binary
input. A roughly 2 MiB text fixture can therefore exercise near-capacity binary
index payloads without weakening either native input bound.

## Independent hosted oracle

After native bootstrap and regression execution, CI uses Python 3.12 to install
one official CPU PyTorch wheel by its exact URL and SHA-256. This dependency is
pinned separately from the inventory because the release tag adds a Ruff
configuration change after the otherwise identical wheel source revision.
`compatibility/pytorch-oracle-lock.json` and `validation/hosted.json` preserve
the two pins and their relationship.

The oracle checks real PyTorch forward results, autograd gradients and optimizer
updates against each already-built HIA executable. The tensor-composition gate recorded below had 268
cases: the preceding 159 plus 35 cat/stack and 74 indexed-addition cases. The current gate has 270 cases after two integrated SGD/AdamW trajectories were added; its separate receipt is `validation/hosted-cross-feature-training.json`. It
records the wheel's source URL, archive hash and Git revision and verifies that
the native executable is unchanged by the test. Both builds passed all 268
cases, 12,932 value comparisons and 11 provenance checks in
[run 43788929713](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43788929713),
at commit `724cbf2eb19d2ef8d400615d7196f5397cb402f6`.
[The complete receipt](../validation/hosted-tensor-composition.json) includes
the actual toolchain/module gates, both 461-pair suites and both full oracle
reports. Historical 159-case receipts and boundary probes keep their identities.

PyTorch is used only for this independent oracle. Python also supports initial
toolchain authoring/bootstrap, source partitioning, build transport and audit
drivers as described above. Neither Python nor PyTorch is a runtime dependency
of either native executable or of the regression loop on already-built products.

## Building native persistent sessions

`make session session-monolithic` builds the optional interactive runtimes.
`make test-session test-session-monolithic` observes 477 native cases per image,
including all 461 existing packet pairs and full-state rollback. Set
`HEPTA_COMPILER`, `HEPTA_MODULE_COMPILER`, `HEPTA_FLAT_LINKER`, and
`HEPTA_SESSION_PACKER` to the pinned native tools. The complete source-adaptation
and native reproduction commands are in [tools/session/README.md](../tools/session/README.md).

The workflow rebuilds both session tools from the same fixed upstream checkout,
checks 18 flat-linker / 22 packer / 3 actual module-growth cases, then builds and
exercises both sessions. It separately compares each image with 21 real pinned
PyTorch cross-frame cases. The existing 270-case ordinary-request gate and 461
fixture pairs remain required. Native source and pre/post producer hashes bind
each build; no Python or upstream tensor runtime executes production requests.

## Reproducing the owned-resource ABI2 profile

Run the ordinary `make legacy monolithic` build first. The owned qualification
compares its actual linked output against `build/heptatorch` and its ordinary
output against the corresponding pinned binaries; a fresh checkout without
those artifacts cannot perform that comparison.

[tools/owned_modules/README.md](../tools/owned_modules/README.md) supplies the
exact `rebuild.py` and `qualify.py` commands. Use the existing fixed upstream
checkout, capacity adaptation receipt and borrowed native tool directory.
Each output directory must be new. The compiler is split into two legal units,
then bootstrap ABI1 and three ABI2 generations are compiled and linked by native
HIA. The 1 MiB application startup and 2 MiB compiler startup are separately
pinned. `tests/check_owned_module_reports.py` binds the 385-case campaign, full
source partition and current runtime byte equality to the original receipts.

CI runs this owned profile after the ordinary runtime exists. It keeps the
existing borrowed toolchain pins and both ordinary and interactive PyTorch
oracle gates. The public entry of this owned-module profile remains the
three-borrowed-buffer entry; an owned persistent-state entry needs a separate
bridge and qualification.

## Building owned interactive storage

First reproduce the owned ABI2 toolchain as above. Use its full build directory,
the same pinned upstream source and the existing borrowed native tool directory:

```sh
python3 -I -S -B tools/owned_sessions/rebuild.py \
  --source-root /absolute/path/to/pinned-hepta \
  --owned-build /absolute/path/to/owned-module-build \
  --owned-profile tools/owned_modules/profile.json \
  --borrowed-tools /absolute/path/to/borrowed-native-tools \
  --output /absolute/path/to/new-owned-session-tools

make storage-objects test-storage-objects \
  HEPTA_SOURCE_ROOT=/absolute/path/to/pinned-hepta \
  HEPTA_OWNED_COMPILER=/absolute/path/to/owned-module-build/native/owned-module-compiler \
  HEPTA_OWNED_LINKER=/absolute/path/to/owned-module-build/native/linker-abi2-w1m \
  HEPTA_OWNED_SESSION_BUILD=/absolute/path/to/new-owned-session-tools
```

The first command reproduces three distinct native linkers: production, a
post-close observer, and a signal observer. Only production is used for the
service. Qualification adapters have their own pins and are never substituted
for the production image.

Make creates `build/storage-native/storage-trace` and
`build/storage-interactive/storage-session`. The test target recompiles the real
five-module service for transport qualification, runs all 131 entry/transport
cases, 12 scenario groups through each entry, five current extreme-state guards,
and three 300-event independent model campaigns. Failed review directories are
retained under `build/`; earlier observations are not overwritten. The combined
actual receipt is `build/hosted-owned-storage-gates.json`.

The historical sixth guard is a preserved failure of candidate 03. It is not
counted as a current passing production witness. The original 456-file storage
package and all 23 owned-session package files are verified byte-for-byte.
`tests/check_storage_object_reports.py` also checks exact generated modules,
executables, response corpora, independent transcript hashes and loaded report
identities. Native HIA performs all target allocation, copies, lifetime changes,
rollback, linking and execution; Python authors source, transports bytes and
computes independent test expectations.

CI repeats these commands after the owned-module toolchain and keeps both
ordinary and both interactive pinned PyTorch gates required. Passing the byte
service does not qualify its still-pending numerical Tensor integration.

## Building growable numerical Tensors

After reproducing the owned module compiler and owned session tools above:

```sh
make owned-tensors test-owned-tensors \
  HEPTA_COMPILER=/absolute/path/to/expanded-hia-native-compiler \
  HEPTA_SOURCE_ROOT=/absolute/path/to/pinned-hepta \
  HEPTA_OWNED_COMPILER=/absolute/path/to/owned-module-build/native/owned-module-compiler \
  HEPTA_OWNED_SESSION_BUILD=/absolute/path/to/new-owned-session-tools
```

The resulting native service is `build/owned-tensors/owned-tensor-session`.
The checker verifies the exact 23-file implementation package, 15-file independent
review, and 5-file PyTorch-oracle package; it binds all generated units, the native
producer, the 150-function source mapping and the resulting executable. It then
checks the 11 native scenario groups, six real typed-owner probes and the separate
reviewer's 16-frame numerical/lifetime campaign. Failed and successful review
directories are preserved. The actual aggregate is
`build/hosted-owned-tensor-native.json`.

After installing the exact locked CPU wheel as described above, run:

```sh
python3 tests/owned_tensor_oracle/pytorch_owned_tensor_oracle.py \
  build/owned-tensors/owned-tensor-session build/owned-tensor-pytorch.json \
  --session-reference tests/pytorch_session_oracle.py \
  --reference tests/pytorch_oracle.py \
  --lock compatibility/pytorch-oracle-lock.json --repo .
```

This distinct gate requires 28 actual cases and all 11 wheel provenance checks.
It preserves the old numerical assertions and tolerances by an AST reconstruction
check, while adding absolute parameter-version assertions. A native scoped PASS
cannot replace this gate. The current candidate still couples allocation and
version identity in its Tensor descriptor; the independently qualified byte
Storage/Version object service has not yet been connected to these kernels.

### Corrected optimizer version profile

The original owned Tensor profile is preserved as the exact first reviewed
candidate. The current successor is built and tested with
`make owned-tensor-versions test-owned-tensor-versions` and the same native tool
variables. Its executable is `build/owned-tensor-versions/owned-tensor-session`.
See [the native version correction](../tools/owned_tensor_versions/README.md)
for the unchanged real PyTorch oracle command, the explicit image identity and
the complete first-failure receipt. CI runs both original native qualification
and corrected qualification before executing the strict oracle against the
corrected image. It does not count replaying one corpus as new upstream coverage.

## Guarded stack and native call-path admission

The separate ABI3 profile in [tools/stack_budget](../tools/stack_budget/README.md)
reproduces a native compiler and nine native linker variants from the unchanged
ABI2 prerequisites. Its native linker checks the real machine call sites and
function prologues against the emitted metadata, computes the longest resolved
call path, and rejects a program that exceeds its declared stack budget. The
production executable allocates an 8 MiB private stack between two inaccessible
pages; normal owned-state cleanup releases this stack and the I/O mappings.

```sh
python3 -I -S -B tools/stack_budget/rebuild.py \
  --source-root /absolute/path/to/pinned-hepta \
  --owned-build /absolute/path/to/owned-module-build \
  --session-build /absolute/path/to/owned-session-tools \
  --borrowed-tools /absolute/path/to/native-module-tools \
  --output build/stack-budget-tools

make stack-tensors test-stack-tensors \
  HEPTA_COMPILER=/absolute/path/to/expanded-hia-native-compiler \
  HEPTA_MODULE_COMPILER=/absolute/path/to/native-module-tools/module-compiler \
  HEPTA_MODULE_LINKER=/absolute/path/to/native-module-tools/module-linker \
  HEPTA_SOURCE_ROOT=/absolute/path/to/pinned-hepta \
  HEPTA_OWNED_COMPILER=/absolute/path/to/owned-module-build/native/owned-module-compiler \
  HEPTA_OWNED_SESSION_BUILD=/absolute/path/to/owned-session-tools \
  HEPTA_STACK_BUILD="$PWD/build/stack-budget-tools"
```

The output is `build/stack-tensors/owned-tensor-session`, with SHA256
`e172bc81dfbe9ee2c9853cf9221ebf7236e8099a207947bccff66b0b01dc1f5f`.
All 151 numerical core function bodies and all three module sources are identical
to the corrected ABI2 Tensor application. Its resolved maximum call path is
61,488 bytes, plus a 4,096-byte reserve. Both programs have byte-identical linked
application text after the distinct 4096-byte startup area.

The Make gate runs the 86 native stack/resource cases, a separately authored
nine-case metadata attack/execution review, and the current Tensor's unchanged
11 native groups and 16 peer frames. It retains the full aggregate in
`build/hosted-stack-budget.json`. The CI workflow additionally runs the same
strict 28-case PyTorch oracle against the guarded image with its explicit SHA;
the oracle and its numerical assertions are unchanged. Actual run
[43803875841](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43803875841)
passed all 40 steps, including 28 strict guarded-image cases, 6,233 frames,
2,590,207 compared values and all 11 provenance checks. The full receipt is
`validation/hosted-stack-budget.json`; the 283 explicit momentum-state lifecycle
observations remain open and are not counted as parameter-version mismatches.

This profile assumes verified source/object custody through the pinned native
producer. It is not a general verifier for arbitrary substituted x86 code,
asynchronous signal recovery, or every future compiler backend. The original
matrix-only false-admission witnesses are retained in the validation history;
the new implementation reconstructs actual calls before accepting the matrix.

## Independent numerical Tensor objects

After reproducing the owned-session and ABI3 toolchains, build and execute both
object profiles with:

```sh
make tensor-objects test-tensor-objects \
  HEPTA_SOURCE_ROOT=/absolute/path/to/pinned/hepta \
  HEPTA_OWNED_COMPILER=/absolute/path/to/owned-module-build/native/owned-module-compiler \
  HEPTA_OWNED_SESSION_BUILD=/absolute/path/to/owned-session-tools \
  HEPTA_STACK_BUILD=/absolute/path/to/stack-budget-tools
```

The two executables are `build/tensor-objects/abi2/tensor-object-session` and
`build/tensor-objects/abi3/tensor-object-session`. Their SHA256 values are
`4bd22ab115a5428be6ee1ab77fd31a1e20baf14a80756eb38f000574cbad531f` and
`2b39ff1a5a258c5fae7be1fbca5dcf0764787b43fa72ad8fdc3ab6fd119e2a91`.
The qualification verifies 61 implementation files, 9 independent review files,
39 oracle files and the complete historical independent receipt before and after
execution. It reruns all six native suites and the native bridge smoke separately
on each executable. Full current observations are written to
`build/hosted-tensor-object-native.json`; this report does not claim Torch ran.

The frozen real-PyTorch runner is
`tests/tensor_object_oracle/pytorch_tensor_objects_oracle.py`. Its positional
arguments are the native executable and an output JSON path; pass
`--owned-reference tests/owned_tensor_oracle/pytorch_owned_tensor_oracle.py`,
`--session-reference tests/pytorch_session_oracle.py`,
`--reference tests/pytorch_oracle.py`,
`--lock compatibility/pytorch-oracle-lock.json`, `--repo .` and the exact
`--expected-native-sha256`. The same 35 cases run on both profiles in CI. The
28 older numerical case bodies and assertions are frozen, and the transport
adaptation is checked by reversible AST and packet proofs. Native object IDs
come from actual responses or explicit same-request references. The bridge does
not compute target numerical outputs, gradients or version counters.

The full-scope acceptance gates remain open. In particular this candidate keeps
Saved data until explicit graph release/reset, and its internal logical
reference counts do not claim parity with PyTorch intrusive pointer use counts.

### Complete hosted object reports

The exact source-bound native qualification and the two actual 35-case Torch
results are recorded in `validation/hosted-native-tensor-objects.json`. Each full
original Torch JSON report is stored without changes in
`validation/hosted-tensor-object-oracles/abi2.json.zlib` or `abi3.json.zlib`.
The receipt records compressed and original byte lengths and SHA256 hashes.
These are complete reports, not replacement summaries. For example:

```sh
python3 -c 'import pathlib,zlib; p=pathlib.Path("validation/hosted-tensor-object-oracles/abi2.json.zlib"); pathlib.Path("/tmp/tensor-object-abi2-report.json").write_bytes(zlib.decompress(p.read_bytes()))'
```

The CI log uses the same byte-preserving zlib/base64 envelope to stay within the
log transport limit. The original oracle exit status remains mandatory; report
compression neither changes numeric JSON values nor reclassifies failures.
The separate `backward-observations.yml` workflow records actual pinned reference
behavior for the next release/failure implementation; reference observations do
not count as native parity.

## Persistent native CPU Generator service

Persistent CPU Generators now execute in the canonical Tensor/Generator image.
Use the [current runtime build and qualification guide](current-runtime.md):

```sh
make HEPTA_SOURCE_ROOT=/absolute/path/to/pinned/Hepta \
     HEPTA_STACK_BUILD=/absolute/path/to/reproduced/stack-budget-tools
make test HEPTA_SOURCE_ROOT=/absolute/path/to/pinned/Hepta \
          HEPTA_STACK_BUILD=/absolute/path/to/reproduced/stack-budget-tools \
          HEPTA_TORCH_PYTHON=/absolute/path/to/pinned-env/bin/python
```

The entropy prerequisite reproduces only the four-argument native session
linker. The current application cold-compiles all fifteen HIA units and executes
the complete mandatory Generator131 group: original Torch24, native11, entry59
and37 new dispatch controls. Native step, full state, epoch/allocation, real
entropy, rollback, original ABI refusal and actual probe/reordered-image checks
remain required. The original three-argument and runtime linkers still have
independent qualification consumers and must remain available.

The independent source03 Generator131 execution and the subsequent fresh
portable04/current644 aggregate both passed on the current712 image. Independent
readback verified all 65,627 bound inputs and nine original runner streams.
Source-head hosted qualification remains PENDING. Original failure and pure-read
records remain separate in
[the execution index](../validation/current-runtime712/evidence.json).

The old standalone Make targets and two duplicate old CI calls have been
replaced by this current mandatory group. Their original nineteen source files,
seven reports and original package manifest are preserved once in the exact
[original26 fixture](../validation/current-generator-original26/manifest.json).
The old checker is preserved in the
[qualification fixture](../validation/current-generator131-source/manifest.json),
and the [retirement map](../validation/current-generator-original26/retirement.json)
records all old bytes, modes and recovery locations. The original Torch oracle,
actual pip installation report, fixed toolchain and unrelated legacy qualifiers
remain live inputs. Historical hosted results still belong to their original
source/image and retain their complete raw evidence.

This integration does not close the full RNG gate. Global/default and concurrent
bindings, every distribution, all dtype/device variants and Tensor/Generator
handle interoperability require separate implementation and qualification.

## Default backward release foundation

`make backward-release test-backward-release` builds the separately named
command-57 component from the frozen guarded Tensor object core. Supply
`HEPTA_SOURCE_ROOT` and `HEPTA_STACK_BUILD` as above. By default its baseline is
`build/tensor-objects/abi3`; `HEPTA_TENSOR_OBJECT_BASELINE` may identify an already
verified build of exactly the same pinned image. The builder checks that source
closure, and the root qualification driver checks the complete 30-file frozen
package before and after execution.

The native executable is `build/backward-release/tensor-release-session`, SHA256
`f146b71b5614b7d1022ef41875f5fbfd857e80028d33b2d634ecd255163f320b`.
The driver replays all native contracts, all 36 actual historical reference
observations and a real 11-module owner-handoff executable. The first 20
reference operator cases match exactly; the remaining 16 contain 15 matching
cases and one explicit partial-failure side-effect difference. The 15 tanh
comparisons keep their original declared tolerances. The aggregate status is
`QUALIFIED_WITH_OPEN_DIFFERENCE`; the old whole-frame rollback interface is not
reclassified as PyTorch-style partial execution.

The next implementation will be admitted separately with its unchanged strict
reference tests, including the preserved signed-zero failure. The lightweight
reference workflow also observes sparse uniform interval-admission boundaries
using the exact fixed wheel. Reference observations remain distinct from target
execution and are not accepted as a full subsystem closure.

## Partial backward and common Tensor RNG execution

The following targets use the same pinned HIA source checkout and reproduced ABI3
stack tools as the default-release foundation:

```sh
make gradient-materialization test-gradient-materialization \
  HEPTA_SOURCE_ROOT=/path/to/pinned-hepta \
  HEPTA_STACK_BUILD=/path/to/stack-tools
make tensor-rng-composition test-tensor-rng-composition \
  HEPTA_SOURCE_ROOT=/path/to/pinned-hepta \
  HEPTA_STACK_BUILD=/path/to/stack-tools
```

`HEPTA_TENSOR_OBJECT_BASELINE` defaults to `build/tensor-objects/abi3`; it may
point to an already reproduced, exact Tensor09 build. The dependency chain
rebuilds release04, partial04, materialization05 and the common image. The
standalone Tensor-RNG03 build remains separate for the real cross-profile owner
handoff. No previously frozen producer or oracle is edited.

The materialization gate requires the original partial04 signed-zero replay to
exit 1 with its exact nine failing cases, then requires the successor to pass
all thirteen. It also executes nineteen native partial/presence contracts,
fourteen legacy57 controls, four independent58 controls, all sixteen actual
partial-error recipes, eight actual presence recipes, four raw-gradient tests
and read-only committed-owner staging checks. The eight presence recipes retain
their eight state-Version and sixty-five lazy-state-creation differences.

The common image is `build/tensor-rng-composition/native/tensor-rng-session`.
Its frozen38-file package is bound by
`validation/tensor-rng-composition-package.sha256`; core and final image SHA256
are recorded in its profile and reproduction receipt. The integrated gate
reexecutes the explicit RNG admission adaptation, exact actual88 range checks,
entry/work bounds, three joint transaction/growth recipes, 12-step training
through both57 and58, eight legal cross-profile owner directions and the
unchanged thirteen/sixteen/eight gradient replays.

CI additionally installs the original locked PyTorch wheel and requires the new
independent32-case live comparison and original frozen35-case Tensor oracle on
this same image. Their numerical assertions, original tolerances and eleven
wheel-provenance checks remain mandatory. The standalone reference observation
workflow does not substitute for these live differential runs.

The preparation report is intentionally `NOT_RUN`; native replay reports state
that they consume actual hosted observations rather than execute Torch locally.
All full subsystem gates and the original2590-schema denominator remain open.

## Nullable gradients in standalone and Tensor RNG owners

Use the same pinned source checkout and ABI3 tools:

```sh
make gradient-none tensor-rng-nullable test-gradient-none \
  HEPTA_SOURCE_ROOT=/path/to/pinned-hepta \
  HEPTA_STACK_BUILD=/path/to/stack-tools
```

The dependency chain reproduces the original Tensor objects, release and
materialization profiles before compiling the new standalone image at
`build/gradient-none/tensor-nullable-session` and the combined image at
`build/tensor-rng-nullable/tensor-rng-session`. The frozen 25-file standalone
package and 20-file common package remain byte-identical. The latter combines
authoritative HIA function bodies, then invokes the unchanged native compiler
and linker on all seven real units.

`tests/qualify_gradient_none.py` verifies every packaged input before and after
execution. Each image runs six native storage/rollback groups, the ten actual
gradient-clear observations and unchanged thirteen/sixteen/eight gradient
replays. The combined image also runs the previous three joint recipes, four
new clear/RNG/training cases and real owner handoffs with both predecessors.
The standalone barrier separately executes both ownership directions. No raw
native pointer is serialized into production. The small observer adapter only
changes private state magic and table offsets; its exact source transformation
must match the frozen receipt.

The complete local qualification is preserved losslessly through
`validation/gradient-none-root-native.json`. Earlier integration-driver schema
and output-directory errors are also preserved there, separately from native
contract failures. The command writes `build/hosted-gradient-none-native.json`
for lossless CI emission. Hosted CI additionally runs the unchanged live32 and
frozen35 oracles against the new common image, retaining all original numerical
assertions, tolerances and eleven reference-provenance checks.

The ten clear recipes keep six raw Adam bit differences under the original
4e-5 relative / 5e-6 absolute policy, plus their explicit 11/9/30 lifecycle
observations. The older eight presence recipes retain their separate 8/65
observations. Repeated execution does not add API coverage. Public gradient
Tensor identity/Version, externally retained gradient aliases and true
Optimizer-owned state remain open.

## Owned SGD objects and captured gradient edges

`make owned-optimizers test-owned-optimizers HEPTA_SOURCE_ROOT=/path/to/pinned/Hepta HEPTA_STACK_BUILD=/path/to/stack-build` builds on the existing ABI3 Tensor, partial-backward, gradient-materialization and nullable-RNG prerequisites. It writes `build/owned-optimizers/tensor-optimizer-session`.

The source-bound qualifier verifies all frozen implementation and independent replay manifests, all seven module sources and objects, the ELF and compiler/linker identities before and after execution. Its 19 reports include the 24/6/3 actual-reference suites, independent private-root migration controls, native payload/allocation guards, the unchanged 13/16/8/10 backward replays, joint RNG regressions and four directions of real owner transfer. It also extracts the source-pinned historical candidate06 and reproduces both known native failures with expected exit 1; these failures remain separate from candidate08 qualification.

A cold rebuild must match source `7c68a96b08aa9a21a52e2fd12be912c5f47dbde889e462f9a0dbb38cb3c09edf` and ELF `b6634158e079512e797d8237cca12d65052db18df23b12b8414adeb8d0b4b0e1`. Full aggregate host-authoring admission is not claimed: its actual refusal remains recorded. Complete production modules and the actual linked call graph pass the unchanged native checkers. Hosted CI additionally runs the unchanged live32 and frozen35 oracles on this image.

The formal result is scoped to exposed SGD, ownership and leaf flags. The 24-case reference is recorded as 18 qualified, one partial and five open, never as 24 native passes. Full optimizer, public-gradient-object and PyTorch completion gates stay open.

## Owned Adam and AdamW state Tensors

The frozen Adam component consumes the exact reproduced Optimizer08 source.
It creates actual owned step, first-moment and second-moment Tensors, retaining
the same guarded ABI3 compiler/linker and the existing SGD algorithm. Build and
qualify it after the prerequisite stack tools are available:

```sh
make owned-adam test-owned-adam \
  HEPTA_SOURCE_ROOT=/path/to/pinned/Hepta \
  HEPTA_STACK_BUILD=/path/to/stack-build
```

`owned-adam` depends on `build/owned-optimizers/reproduction.json`, so the
existing Tensor/object, partial-backward, nullable-gradient and RNG dependency
chain remains explicit. The target invokes the frozen
`tools/owned_adam/rebuild.py` with the predecessor, repository, pinned upstream
source root and both native tools. It produces
`build/owned-adam/tensor-adam-session`, `runtime.hepta`, all eight complete
module sources/objects, `build.json` and `reproduction.json`. Source pins are
checked before and after the producer. The exact identities are:

| Artifact | SHA-256 |
| --- | --- |
| Generated core | `253438648a437a75ef2dbc77138eca74cb5fe6a6e2ea322c7c38e5a535e5d2a6` |
| Final executable | `fa92356954dbb9e0c454f2a58c2f78aa33950103d377c57e7b659ec9904183af` |
| ABI3 compiler | `b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f` |
| ABI3 session linker | `aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16` |

The assembled document contains 352 functions, 4,598 blocks and 17,185
instructions. 308 of the 315 predecessor function bodies remain byte-identical.
The native checker admits each complete module, and the linker checks the
actual whole-program call graph and stack bound. This does not claim that the
aggregate host authoring document fits the single-document budget or that
those upstream limits were increased.

The root qualification driver can also be run directly with a fresh output
directory:

```sh
python3 -I -S -B tests/qualify_owned_adam.py \
  --build build/owned-adam \
  --baseline build/owned-optimizers \
  --stack-build /path/to/stack-build \
  --output build/owned-adam-qualified
```

It verifies both frozen package manifests, all dependencies, source/object/ELF
identities and before/after source custody. Its 19 reports cover the independent
old 19 recipes and new four Adam recipes from the unchanged 24-case report;
6 flag and 3 accumulator-lifetime cases; private-root migration; native
ownership, admission and actual float32-counter boundaries; unchanged legacy
SGD and 13/16/8/10 backward replays; both RNG joint suites; and a real old/new
owner handoff executable. The two accumulator-observer tag edits are exactly
reversible. Numerical inputs, tolerances and original report identities are
preserved. The Make target copies the final result to
`build/hosted-owned-adam-native.json` for lossless CI emission.

[The root native receipt](../validation/owned-adam-root-native.json) records a
successful cold build and all 19 actual qualification reports. The original
24-case reference now has 22 cases qualified within exposed scope, one partial
retained-nonleaf recipe and one unexecuted public-gradient-object recipe.
The old 18/1/5 report is unchanged. These are recipe results, not a count of
fully closed PyTorch schemas.

CI adds three steps: native reproduction/qualification, the unchanged 32-case
live joint oracle on the final Adam ELF, and the unchanged 35-case Tensor
oracle on the same ELF. The new reports are
`build/pytorch-owned-adam-backward.json` and
`build/pytorch-owned-adam-frozen35.json`. Those two new-image hosted runs are
pending; earlier executables' passes are not transferred to this one.

For command layouts, ownership and remaining exponent/mode boundaries, see
[the component description](owned-adam.md) and
[the frozen protocol](../tools/owned_adam/PROTOCOL.md). The private tags
`HTADAM01` / `HTOADAM1` deliberately reject nonempty predecessor state; the
external `HTORCH03` / `HTORSP03` wire protocol is retained. Production state is
created and transferred through checked native owner calls. No user pointer or
copied owner descriptor is introduced.

## Native CPU Bernoulli and dropout

The distribution targets use the reproduced nullable Tensor/RNG predecessor,
the common and gradient-materialization authoring layers, and the unchanged
pinned ABI3 compiler/linker:

```sh
make cpu-distributions cpu-distributions-partial test-cpu-distributions \
  HEPTA_SOURCE_ROOT=/path/to/pinned-hepta \
  HEPTA_STACK_BUILD=/path/to/stack-tools
```

The dependency chain first provides `build/tensor-rng-nullable`,
`build/tensor-rng-composition/native/overlay-authoring.py` and
`build/gradient-materialization/overlay-authoring.py`. The atomic producer
compiles seven real units into
`build/cpu-distributions/tensor-distributions-session`; the successor consumes
that exact build and produces
`build/cpu-distributions-partial/tensor-distributions-session`. Each recipe
checks its frozen package manifest, authors into an isolated temporary
directory, verifies source/object/image identities and copies the successful
result into its build directory. Exact pins and state barriers are listed in
[the component contract](cpu-distributions.md).

`test-cpu-distributions` executes both frozen qualification drivers, in order:
five atomic suites and six successor suites. It does not substitute stored
PASS reports for native execution. Only after both finish does
`tests/check_cpu_distribution_reports.py` bind their eleven reports to the
actual builds and write `build/hosted-cpu-distributions-native.json`. Raw suite
outputs remain under the printed `.cpu-distributions-review.*` build directory.

The checker verifies both 37-file and 28-file frozen manifests; complete HIA,
unit-object and ELF identities; each concrete report's source/image or
qualification-only identity; typed-peer source bindings; all 183 reference
indices, names, families and status counts; the exact 18 unrepresented indices;
and every member's bytes, SHA256 and mode in both historical archives. The
original checker and its earlier accepted source-mismatch/truncated-row
witnesses are preserved separately in the integration history. No production
or frozen qualification file was changed to strengthen report admission.

The original commands retain 161 PASS, four explicit transaction differences
and 18 unrepresented observations. Commands 86/87 strictly match 165 of the same
183 observations, closing only zero-based indices 101, 103, 105 and 179. This
scoped native qualification returns success while its compatibility status
stays PARTIAL; it does not turn HOLD into whole-PyTorch acceptance.

CI adds one step for both rebuilds and all eleven native suites. After the
locked PyTorch wheel is installed, two further steps run the unchanged live
32-case joint oracle and frozen 35-case Tensor oracle on the final partial
image. Their reports use separate `pytorch-cpu-distributions-*` filenames.
All previous steps and historical reports remain unchanged. These added live
runs are required gates, not results inferred from the fixed-observation replay.

The remaining 18 observations require float64 probability Tensors (eight),
int64 probability Tensors (two), or the distinct `native_dropout` boolean-mask
interface (eight). The seven-kind owner also has not yet been composed with the
kind-8 Optimizer or public-gradient-object successor. Full RNG, dtype and
PyTorch-rewrite flags remain false and the original 2,590-schema denominator is
unchanged.

## Combined owned Adam and CPU distributions

The combined profile consumes the complete frozen Adam02 and explicit partial
distribution01 builds. Its single native owner contains the actual Tensor,
Generator, graph, gradient, Saved and Optimizer objects. It has 378 core
functions, 4,971 blocks and 18,539 instructions across eight actual native units.
The unchanged ABI3 compiler checks each complete unit and the native linker
checks their real interfaces, resource contracts and whole-program stack bound.

```sh
make adam-distributions test-adam-distributions \
  HEPTA_SOURCE_ROOT=/path/to/pinned-hepta \
  HEPTA_STACK_BUILD=/path/to/stack-tools
```

`adam-distributions` depends on both
`build/owned-adam/reproduction.json` and
`build/cpu-distributions-partial/reproduction.json`. Their existing dependency
chains provide the exact source and authoring inputs first. The new recipe
verifies `validation/adam-distributions-package.sha256`, calls the frozen
`tools/adam_distributions/rebuild.py` in a fresh temporary directory, and copies
the result only after successful reproduction. The producer checks its input
and imported upstream source closure before authoring and again after execution.

The resulting `build/adam-distributions/` contains
`tensor-adam-distributions-session`, `runtime.hepta`, all eight real unit
sources/objects, `build.json`, `partition.json` and `reproduction.json`. The
expected complete identities are:

| Artifact | SHA-256 |
| --- | --- |
| Combined runtime source | `5d3a3c047d024a3880d257a317a04c2166a2e87bbc6b248bef62649a88546873` |
| Combined native ELF | `3fc59ab7ed9c32dd75ff440ef9bfbec31756f50edd77715d665bdac098cc3f22` |
| ABI3 native compiler | `b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f` |
| ABI3 native session linker | `aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16` |

The private state tags are `HTADIST1` and `HTOADIS1`; they deliberately
distinguish this eight-kind state from both frozen predecessors. Public framing
remains `HTOWNR01` / `HTOWNS01`, with the existing `HTORCH03` / `HTORSP03`
application protocol. The native checked `step` and `close` ownership signatures
are unchanged. See [the shared-owner protocol](../tools/adam_distributions/PROTOCOL.md)
for operation-error prefix commits and ordinary full-frame rollback.

The root qualification CLI is:

```sh
python3 -I -S -B tests/qualify_adam_distributions.py \
  --build build/adam-distributions \
  --adam-build build/owned-adam \
  --atomic-build build/cpu-distributions \
  --distribution-build build/cpu-distributions-partial \
  --stack-build /path/to/stack-tools \
  --output build/adam-distributions-qualified
```

`test-adam-distributions` verifies both the component and independent-review
package manifests, runs that CLI in a new directory, and copies its aggregate
report to `build/hosted-adam-distributions-native.json`. The printed
`.adam-distributions-review.*` path retains the executed subreports and raw
evidence. Stored reference reports are inputs to real execution, not a
substitute for running the native image.

The frozen component qualification replays the original 183 distribution rows
through both interfaces and the original optimizer recipes. It keeps the
161/4/18 atomic and165/18 explicit partitions, and the 22 qualified/1 partial/1
open optimizer ledger. The old atomic negative control expected unknown-opcode
status 4 for 86/87; the successor correctly reaches typed admission and returns 1
for its empty arguments. That original FAIL is retained, while a separately
identified derivative checks the new command contract. The qualification must
not count the old control as an unchanged pass.

The [independent review](../validation/adam-distributions-review/README.md)
checks exact source inheritance and six additional mixed-owner contracts. Its
actual 462 interactive frames compare state identity, captured edges, final
strong-reference release, full RNG states and full-owner rollback. Two short
training paths execute real dropout, backward, Adam or AdamW, gradient clearing
and graph reset. These are finite native source/model contracts; they do not
increase the number of hosted PyTorch observations or implemented schemas.

CI adds three steps: the new root native qualification, the unchanged 32-case
live joint oracle, and the unchanged 35-case Tensor oracle against the exact
combined ELF. The latter reports are
`build/pytorch-adam-distributions-backward.json` and
`build/pytorch-adam-distributions-frozen35.json`. Their results require actual
hosted execution; passing predecessor reports do not qualify this new image.
Every previous workflow step and its 21 literal report-envelope paths remain
unchanged. The existing artifact collector discovers the three new paths from
the workflow and consequently requires 24 original JSON reports, without
reserializing their bytes.

The public-gradient-object successor, remaining dtype/distribution cases,
general optimizer modes and exception behavior, higher-order differentiation
and all other full-rewrite gates remain open. The component stays PARTIAL and
the whole-project decision stays HOLD with the same 2,590-schema denominator.


## Public gradient objects and combined owner

The new `test-gradient-objects` and `test-gradient-adam-distributions` targets
cold-build the exact355-function public-gradient and428-function combined
profiles, respectively. Their root qualification preserves all original
reference and failure evidence. See
[public gradients and composition](gradient-objects-and-composition.md) for
commands, source/image identities, the separate suites, the six new hosted gates
and the remaining boundaries. These profiles do not replace the older frozen
builds or close a complete PyTorch subsystem.

## Native int64 Tensor profile

`make integer-tensors` verifies the frozen source package and actually rebuilds
all eleven native modules. The resulting executable is
`build/integer-tensors/tensor-integer-session`. The build depends on the exact
523-function native-dropout parent; it does not use an existing int64 image.

Run the complete native integration gate with:

```sh
make test-integer-tensors \
  HEPTA_SOURCE_ROOT=/absolute/path/to/fixed-hepta-source \
  HEPTA_STACK_BUILD=/absolute/path/to/complete-stack-tools \
  HEPTA_COMPILER=/absolute/path/to/hia-native-compiler
```

The stack directory must contain the pinned module compiler, session linker
and runtime linker under `native/`. The coordinator creates a new directory,
executes the ten-step cold producer, six independent resource cases, five owned
entry groups and two typed handoff directions against that same new image.
The producer's sibling imports use a disclosed `-S -B` child invocation; the
stdlib-only coordinator itself uses `-I -S -B`. Source package hashes and modes
are required before and after execution.

The aggregate report is `build/hosted-integer-tensors-native.json`; complete
original evidence is `build/integer-tensors-ci-raw.tar.xz` with
`build/integer-tensors-ci-raw-manifest.json`. On failure, the actual failure
status and available original evidence remain. Successful native integration
does not substitute for the unchanged live32, complete35, copied observer
controls or the new source head's hosted artifact admission. See
[the int64 profile](integer-tensors.md) for exact scope and remaining work.


## Native mixed integer autograd

The separate 556-function profile adds genuine mixed F32/int64 differentiation
and original Saved Tensor access to the qualified 544 parent. All eleven native
units are rebuilt by the pinned HIA compiler and linker.

```sh
make test-mixed-integer-autograd \
  HEPTA_SOURCE_ROOT=/absolute/path/to/pinned/Hepta \
  HEPTA_STACK_BUILD=/absolute/path/to/complete/stack-build \
  HEPTA_COMPILER=/absolute/path/to/expanded/hia-native-compiler
```

The stack build must contain `native/stack-module-compiler`,
`native/stack-linker-session` and `native/stack-linker-runtime`. The parent target
first supplies a real `build/integer-tensors/` image. The new qualification
creates a fresh output directory and executes seven outer commands: producer5,
source custody, full owner accounting with copied refusals, resources, owned
entry, new Saved entry, and actual previous/new owner handoff. The Saved-entry
gate explicitly reuses only the preceding gate's newly compiled probe.

The aggregate is `build/hosted-mixed-integer-autograd-native.json`. Complete
original evidence is `build/mixed-integer-autograd-ci-raw.tar.xz` with
`build/mixed-integer-autograd-ci-raw-manifest.json`. Failure paths preserve
their original reports, stdout/stderr, completed commands and partial evidence.
Only a successful full native qualification copies the new image from
`producer5/native` to `build/mixed-integer-autograd/`.

The build-only `make mixed-integer-autograd` target reproduces the locked image;
it does not claim the seven qualification gates. Complete live32, accounted35,
copied18 refusals and hosted artifact admission are separate obligations.
See [the exact contract and evidence](mixed-integer-autograd.md).
