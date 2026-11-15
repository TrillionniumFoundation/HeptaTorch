# Reproducible native build

## Target and toolchain

The current executable target is Linux x86-64. The canonical implementation is
`src/heptatorch.hepta`, using HIA primitive Core v8 and the `selfhost_exec` output
profile. Functions have semantic names and explicit typed SSA instructions.
`docs/functions.json` indexes those native functions. The checked-in HIA source
is authoritative; no source generator is required to edit, compile, or run it.

| Input | Pinned identity |
| --- | --- |
| HIA repository | `TrillionniumFoundation/Hepta` |
| HIA commit | `f33d8da316de8fdf27b504371f1a5310997ff548` |
| Compiler source | `examples/hia_bootstrap/selfhost.hepta` |
| Compiler source SHA-256 | `388b182e42675f11435867dd99520e9452a4ff48e23c0766a3952b662acc7627` |
| Original bootstrap compiler SHA-256 | `87c247e4d1c3c9a449d2947fd4227fdc186d519bffd203a04cafce7861a4ed54` |
| HeptaTorch compiler profile | `heptatorch.hia.compiler.f256-b2560-w2m.v1` |
| Expanded compiler HIA SHA-256 | `5cf8a4286d0759d2368e2d8d5658ce7de0b2ad72435ce75915cf07c0a23df7bf` |
| Current native compiler SHA-256 | `c5a9c2b95a3061e080a21298f5b26000936ed9dac519690d845b0699a15744c6` |
| Current native compiler size | 674,085 bytes |

The Makefile checks the compiler hash before invoking it. Compilation is ordinary
stdin/stdout: `hia-native-compiler < source.hepta > executable`. The compiler
emits the ELF header, machine code, and syscall entry wrapper. The resulting
program has no `PT_INTERP`, dynamic loader, libc, Python, or tensor-library calls.

The original upstream compiler accepts 128 functions and 1,536 basic blocks.
HeptaTorch's explicitly named capacity profile accepts 256 functions and 2,560
total blocks. Both retain the 8,192-instruction, 1 MiB source and 1 MiB image
limits. The current runtime uses 121 functions, 1,439 blocks and 5,751
instructions; its source is 383,387 bytes and its executable is 570,551 bytes.
Old and expanded compilers produce this executable and unhex byte for byte.
The local profile is recorded separately from the unchanged
[upstream support profile](https://github.com/TrillionniumFoundation/Hepta/blob/f33d8da316de8fdf27b504371f1a5310997ff548/docs/HIA_SELFHOST_CORE_SUPPORT.json).
See [capacity and remaining architecture work](upstream-audit.md#compiler-capacity-profile).

## Starting with an existing native compiler

```sh
make HEPTA_COMPILER=/absolute/path/to/hia-native-compiler
make HEPTA_COMPILER=/absolute/path/to/hia-native-compiler test
```

Build files are written under `build/` through temporary files and atomic rename.
`make clean` removes only those known build products.

## Starting from the pinned HIA source

The executable workflow in `.github/workflows/native.yml` is the complete build
recipe. It checks out the exact upstream commit with a narrow source selection
and uses the upstream `build_seed` and `compile_stage` routines:

1. The upstream Python assembler/verifier/direct emitter and GNU ld produce B0.
2. B0 executes the HIA compiler source to produce B1.
3. B1 produces B2, and B2 produces B3; their machine image bytes must agree.
4. B3 compiles the same HIA source with the `selfhost_exec` header.
5. The resulting native executable rebuilds itself twice with an empty PATH;
   both outputs must match the pinned compiler bytes.
6. The repository's source adapters verify the exact upstream import closure,
   reproduce the original HIA source, and generate the separate capacity profile
   and its HIA executable packer under build scratch.
7. The original native compiler produces the expanded compiler's flat image.
   A native HIA packer supplies its 2 MiB compiler workspace. Four native
   compiler/image generations must match exactly; the successor also reproduces
   the packer. All compiler and packer hashes are checked.
8. The expanded compiler passes the 16-case capacity gate and builds HeptaTorch
   and its hexadecimal decoder. CI also requires byte-identical application
   output from the original and expanded compilers.

The bootstrap Python code belongs to the external language toolchain's initial
trust boundary. It does not translate HeptaTorch, execute its tensor operations,
or replace a successor compiler. No Python source file is shipped as part of the
HeptaTorch runtime. The finite bootstrap check is recorded as such; it is not the
upstream project's separate full cold-qualification campaign or a formal
compiler-correctness proof. Capacity-profile authoring uses the same pinned
upstream Python source generator and unchanged Core verifier; native successor
compilation and ELF packing use HIA programs with an empty PATH. The runtime
does not load any of these Python build-time adapters.

Fresh local bootstraps from both a clean source checkout and a narrow source
export reproduced the same compiler hash. The sparse checkout includes root
`pyproject.toml`, which the upstream compiler uses as a source-root marker.

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

The native-profile output directory must not already exist. The generated HIA
files are derived from the fixed external checkout and are not vendored in this
repository. `compatibility/hia-expanded-profile.json` records the exact products,
limits and application wrapper invariants. `validation/compiler-profile.json`
retains the per-edit source receipt, table layout and independent native checks.

Only the compiler executable has a 2 MiB workspace. Ordinary `selfhost_exec`
applications, including HeptaTorch and unhex, keep their 1 MiB input, output and
workspace buffers. Per-function blocks (256), SSA names (2,048), argument count
(16), Core admission limits and scalar-only explicit exports retain their
existing contracts. General module imports, resource-bearing public APIs and
device code generation remain separate work.

## Test execution

`make test` decodes each `tests/fixtures/*.input.hex` and its expected response
using `tools/unhex.hepta`, runs the native tensor executable with an empty
environment and PATH, and compares response bytes. The shell only orchestrates
processes and file comparison. The hexadecimal decoder itself is HIA.

Golden responses were captured only after independent numerical, analytical,
finite-difference, or explicit-error checks. Their origins and measured
tolerances are retained in `validation/`. They are regression data, not a
substitute for the full upstream compatibility matrix.

The decoder's own input is also bounded by the 1 MiB native input wrapper;
therefore one text fixture can encode at most roughly 512 KiB of binary data.
The tensor executable itself accepts binary requests up to 1 MiB.

## Independent hosted oracle

After native bootstrap and regression execution, CI uses Python 3.12 to install
one official CPU PyTorch wheel by its exact URL and SHA-256. This test dependency
is pinned separately from the source inventory because the release tag adds a
Ruff configuration change after the otherwise identical wheel source revision.
`compatibility/pytorch-oracle-lock.json` and `validation/hosted.json` preserve the
two pins and their relationship.

The oracle checks real PyTorch forward results, autograd gradients and optimizer
updates against the already-built HIA executable. It records the wheel's source
URL, archive hash and Git revision, and verifies that the native executable is
unchanged by the test. Python and PyTorch are needed only for this independent
comparison and the initial external compiler bootstrap described above; neither
is a dependency of the native build products or `make test`.
