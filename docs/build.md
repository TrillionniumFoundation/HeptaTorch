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
updates against each already-built HIA executable. The current gate has 268
cases: the preceding 159 plus 35 cat/stack and 74 indexed-addition cases. It
records the wheel's source URL, archive hash and Git revision and verifies that
the native executable is unchanged by the test. A successful complete current
hosted run must be recorded in `validation/hosted-tensor-composition.json`;
it is pending at this handoff. Historical 159-case receipts and actual current
boundary probes retain their distinct identities.

PyTorch is used only for this independent oracle. Python also supports initial
toolchain authoring/bootstrap, source partitioning, build transport and audit
drivers as described above. Neither Python nor PyTorch is a runtime dependency
of either native executable or of the regression loop on already-built products.
