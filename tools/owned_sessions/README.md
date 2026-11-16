# Native owned sessions

This separate profile links real ABI2 HIA modules to an interactive Linux
x86-64 entry bridge. The native linker admits exactly
`step(readbuf,mutbuf,buffer)->(buffer,u32)` with effects 15 and
`close(buffer)->()` with effects 9. It retains the existing native object,
signature, effect, module graph and relocation checks. The bridge begins with
a trusted live-empty owner, transfers it into each step, adopts the returned
owner, and closes that owner on normal transport termination.

The previous borrowed and owned module profiles retain their own unchanged
identities. This directory contains adaptation recipes and instruction-template
authoring, not a vendored copy of the upstream compiler. Compilation, linking,
allocation, storage mutation and release execute in native HIA-produced code;
Python performs deterministic source authoring, file transport and independent
observations. The trusted startup instruction backend is qualified separately
from the compiler's source-level owner checker. The linker assumes the pinned
producer custody; it does not authenticate arbitrary foreign machine code.

## Rebuild

First rebuild the existing `tools/modules` and `tools/owned_modules` profiles.
Pass the pinned upstream checkout and those build directories explicitly:

```sh
python3 -I -S -B tools/owned_sessions/rebuild.py \
  --source-root "$HEPTA_SOURCE" \
  --owned-build build/owned-modules \
  --owned-profile tools/owned_modules/profile.json \
  --borrowed-tools build/modules/native \
  --output build/owned-sessions
```

`owned-build` contains `sources/` and `native/owned-module-compiler`.
`borrowed-tools` contains `module-compiler` and `linker-packer`. Outputs must be
new directories. The recipe verifies the upstream source closure before and
after authoring, reproduces the exact ABI2 baseline source, and compiles each
new linker through both pinned native compilers. Their flat images must match.
The existing native HIA packer then produces three distinct pinned executables:
production, post-close observation and actual-signal qualification. No test
hook enters the production prefix.

`source-pins.sha256` binds this package; `profile.json` binds generated HIA,
native images, executables and all three startup templates. `rebuild.json`
records actual source and producer checks. The entry bridge uses 1 MiB framing
buffers; owned allocations are separate from those buffers.

## Link and qualify a service

```sh
python3 -I -S -B tools/owned_sessions/compile_link.py \
  --compiler build/owned-modules/native/owned-module-compiler \
  --linker build/owned-sessions/production-native/owned-session-linker \
  --output build/owned-service \
  SERVICE_BYTES.hepta SERVICE_RECORDS.hepta SERVICE_AUDIT.hepta \
  SERVICE_OPS.hepta SERVICE_OWNERS.hepta

python3 -I -S -B tools/owned_sessions/qualify.py \
  --build build/owned-sessions \
  --compiler build/owned-modules/native/owned-module-compiler \
  --output build/owned-session-qualification \
  SERVICE_BYTES.hepta SERVICE_RECORDS.hepta SERVICE_AUDIT.hepta \
  SERVICE_OPS.hepta SERVICE_OWNERS.hepta
```

The qualification service must implement the separately specified owned
storage-object protocol used by the tests. The package does not infer those
business semantics from entry types. The aggregate runs 103 native entry cases,
20 transport/cleanup cases, six returned-length boundaries and two actual
kernel interruption/short-write cases, for 131 cases. It preserves raw native
sources, objects, requests, subprocess observations and individual receipts.
The test-only profiles inspect their own live child through `/proc`; the
signal observer also verifies the Linux x86-64 ucontext layout from system
headers. See [QUALIFICATION.md](QUALIFICATION.md) for precise evidence and
test scope.

Application errors must still return exactly one owner. They do not determine
whether the wrapper adopts a result or calls close. The service owns rollback;
the wrapper never duplicates a native owner descriptor. Native traps and
fatal external signals are outside the explicit-close guarantee. Whole-program
stack-budget admission, concurrency, restart persistence and full PyTorch
compatibility remain separate work. Old profiles are not relabeled as covering
those dimensions.
