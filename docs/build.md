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
| Native compiler SHA-256 | `87c247e4d1c3c9a449d2947fd4227fdc186d519bffd203a04cafce7861a4ed54` |
| Native compiler size | 673,903 bytes |

The Makefile checks the compiler hash before invoking it. Compilation is ordinary
stdin/stdout: `hia-native-compiler < source.hepta > executable`. The compiler
emits the ELF header, machine code, and syscall entry wrapper. The resulting
program has no `PT_INTERP`, dynamic loader, libc, Python, or tensor-library calls.

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
6. The native compiler builds HeptaTorch and its HIA hexadecimal decoder.

The bootstrap Python code belongs to the external language toolchain's initial
trust boundary. It does not translate HeptaTorch, execute its tensor operations,
or replace a successor compiler. No Python source file is shipped as part of the
HeptaTorch runtime. The finite bootstrap check is recorded as such; it is not the
upstream project's separate full cold-qualification campaign or a formal
compiler-correctness proof.

Fresh local bootstraps from both a clean source checkout and a narrow source
export reproduced the same compiler hash. The sparse checkout includes root
`pyproject.toml`, which the upstream compiler uses as a source-root marker.

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
