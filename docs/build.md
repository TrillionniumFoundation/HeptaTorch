# Reproducible native build

## Target and toolchain

The executable target is Linux x86-64. The canonical implementation is
`src/heptatorch.hepta`, using HIA primitive Core v8 with semantic function names
and explicit typed SSA instructions. `docs/functions.json` indexes the native
functions. This remains the only hand-maintained runtime implementation.

The default build partitions those function bodies into two independent HIA
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
toward a module's function limit. The current authoritative runtime has 143
functions, 1,783 blocks and 7,064 instructions in 481,516 source bytes.
Its partition has 63 kernel functions and 80 runtime functions, with 52
declared imports from runtime to kernel.

| Current artifact | SHA-256 | Bytes |
| --- | --- | ---: |
| Authoritative HIA source | `264e9c5c4e8ea1834724cc0d8d859952016daffe4af330a8daa7745e7a5e5233` | 481,516 |
| Default linked runtime | `cf78556c3c512227ffe123635b76e5a6af97697606e8ba3535c56f5101f4f80d` | 733,184 |
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
     all test
```

`HEPTA_COMPILER` names the expanded ordinary compiler; it builds unhex and the
monolithic reference. `HEPTA_MODULE_COMPILER` and `HEPTA_MODULE_LINKER` name the
two module tools. `all` produces the default linked `build/heptatorch` and
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

Those 121 functions fit the original profile. The current 143-function source
does not; its expected old-compiler refusal is not a code-generation regression.
The current ordinary-mode comparison instead uses expanded `c5a9c2b9` and
module `e250d93b` compilers on the full current source and unhex. Historical
compiler/application receipts retain their original source/image identities
rather than being overwritten with current hashes.

## Test execution

`make test` decodes each of the 461 `tests/fixtures/*.input.hex` requests and its
expected response using `tools/unhex.hepta`, runs the native tensor executable
with an empty environment and PATH, and compares response bytes. The shell only
orchestrates processes, byte transport and file comparison. The hexadecimal
decoder itself is HIA. `make test-monolithic` runs this same collection through
`build/heptatorch-monolithic`. The default linked image has passed all 461 pairs;
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

Run the ordinary `make all monolithic` build first. The owned qualification
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
