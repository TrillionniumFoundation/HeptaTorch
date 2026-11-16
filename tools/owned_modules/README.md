# Reproduce the native owned-resource profile

This directory provides the separate ABI2 owned module toolchain. The detailed
ownership, allocator, effect, producer-custody and startup contract is in
[CONTRACT.md](CONTRACT.md); exact identities and limits are in
[profile.json](profile.json). Existing `tools/modules` remains the frozen
borrowed ABI1 profile.

## Prerequisites

Use Linux x86-64 and the existing pinned upstream checkout and expanded-source
adaptation receipt. Reproduce `tools/modules` first. Its output directory must
contain the pinned `module-compiler`, `module-linker` and `linker-packer`.
The build uses Python only for checked source authoring and byte transport;
all target compilation, object admission, relocation and executable packing
run in native HIA. The qualification observer uses Python's standard library
and native memory protection, with no target libc allocator or numerical VM.

The following commands use absolute paths supplied by the build environment.
`owned_build` and `owned_review` must name directories that do not yet exist.

```sh
python3 -I -S -B tools/owned_modules/rebuild.py \
  --source-root "$hepta_source" \
  --capacity-adaptation "$capacity_receipt" \
  --borrowed-tools "$borrowed_build" \
  --output "$owned_build"

python3 -I -S -B tools/owned_modules/qualify.py \
  --build "$owned_build" \
  --source-root "$hepta_source" \
  --borrowed-tools "$borrowed_build" \
  --baseline-compiler "$expanded_compiler" \
  --repo-root "$heptatorch_repo" \
  --output "$owned_review"
```

The rebuild writes generated source, authoring receipts, complete function
partition hashes, stage0 ABI1 objects, three generations of ABI2 objects and
executables, and `rebuild.json`. It also rebuilds every linker and the native
packer using the final compiler and checks byte identity.

The final compiler is `native/owned-module-compiler`. The ordinary runtime
linker is `native/linker-abi2-w1m`; `native/linker-abi2-w2m` is explicitly for
the compiler's 2 MiB workspace. The bootstrap-only ABI1 variant is
`native/linker-abi1-w2m`. These startup identities are not interchangeable.

The qualification writes every HIA source, relevant object/request/output,
per-suite JSON report and an aggregate `qualification.json`. It checks 385
counted cases plus byte equality of the complete current runtime and unhex.
The tests have no dependency on a scratch path. They retain the existing
`tools/modules/tests/contracts.py` and `tools/modules/split_runtime.py` as
explicit repository inputs rather than copying the old tools.

The currently pinned runtime equality witness is source `264e9c5c…`, with
monolithic executable `5b992e5f…` and linked executable `cf78556c…`.
Changing that witness requires an explicit evidence update; passing these
toolchain tests does not claim full PyTorch coverage.

## Compile independent modules

The wrapper below only snapshots files and transports native requests. The
compiler and linker perform actual admission. The selected public entry must
remain `(readbuf,mutbuf,mutbuf)->(u32)`; owners may cross its private module
calls and be consumed, returned and dropped there.

```sh
python3 -I -S -B tools/owned_modules/compile_link.py \
  --compiler "$owned_build/native/owned-module-compiler" \
  --linker "$owned_build/native/linker-abi2-w1m" \
  --entry entry \
  --output "$new_application_build" \
  caller.hepta factory.hepta transform.hepta
```

Owners come exclusively from the checked ABI2 producer. No foreign pointer,
raw owner serialization, arbitrary ELF import or implicit resource conversion
is introduced by this recipe. `owner.drop` and `chain.pop` require `alloc,trap`
in this profile because native release failure has an explicit trap outcome.
