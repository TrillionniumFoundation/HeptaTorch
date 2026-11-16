# Native borrowed HIA modules

This directory implements `heptatorch.hia.borrowed-modules.v1`, a separately
identified adaptation of pinned Hepta source. The actual HIA compiler admits
resource-bearing external declarations. The actual HIA linker matches exact
symbol names, signatures, effects and a conservative module dependency graph,
validates symbolic relocations, and links executable code. No Python tensor
kernel, host ELF linker or executable external-function stub is involved.

Read [CONTRACT.md](CONTRACT.md) for the trust and ownership boundary and
[profile.json](profile.json) for exact capacities and hash pins. This profile
supports borrowed `readbuf` and `mutbuf` arguments. It explicitly refuses owned
resources, allocation, borrowed or owned module results, multiple module
results, and cyclic module dependencies. It does not establish full PyTorch
compatibility or a whole-program stack-cost proof.

## Files and responsibility

| File | Responsibility |
| --- | --- |
| `build_module_compiler.py` | Apply exact changes to verified upstream authoring source and render the complete HIA compiler into build storage. Verify the 32-file upstream closure before and after imports. |
| `module_compiler_additions.txt` | HIA authoring additions for external declarations, signatures, ELF symbols, interface records and real PLT32 call fixups. |
| `build_module_linker.py` | Author the native HIA linker and its native HIA packer; verify the 34-file upstream closure and reproduce the unchanged upstream generators first. |
| `module_linker_additions.txt` | Native HIA object/interface/signature/module-graph admission functions. |
| `source-pins.sha256` | Exact generated compiler, linker and two packer HIA source hashes. |
| `rebuild.sh` | Native compilation and native packing through three identical compiler generations, pinned results and packer reproduction. |
| `split_runtime.py` | Host source organization only: account for every function and preserve all function-body bytes in a two-unit partition. |
| `compile_link.sh` | File and wire transport only; source compilation, contract admission, symbol resolution and ELF linking run in pinned HIA executables. |
| `tests/` | Independent packet construction, expected results and guarded observation of native compilation, linking and execution. |

The compiler's own source remains ordinary HIA, so the existing expanded
compiler can bootstrap it. Generated upstream-derived HIA is kept in build
storage. This directory carries adaptations and authoring additions instead
of vendoring the full upstream compiler. Source provenance remains the pinned
Hepta files named in the generated receipts; this adaptation does not assert
or replace an upstream license.

## Reproduce the native tools

Use a verified external checkout at Hepta commit
`f33d8da316de8fdf27b504371f1a5310997ff548`. First run the existing
`tools/compiler-profile` adaptation and native rebuild. Its native compiler
must hash to `c5a9c2b95a3061e080a21298f5b26000936ed9dac519690d845b0699a15744c6`.
Its generated adaptation receipt must hash to
`1926437c96c413b891b4c0b61d1555434598a40067855241e0d2a53b1bee1292`.
These files are verified again by the module recipe.

The following commands use absolute, caller-selected paths. `module_generated`
and `module_native` must be new build directories. `capacity_generated` is the
existing compiler-profile output containing `adaptation.json` and
`compiler-packer.hepta`.

```bash
python3 -I -S -B tools/modules/build_module_compiler.py \
  --source-root "$upstream" \
  --capacity-adaptation "$capacity_generated/adaptation.json" \
  --output "$module_generated"
python3 -I -S -B tools/modules/build_module_linker.py \
  --source-root "$upstream" \
  --capacity-adaptation "$capacity_generated/adaptation.json" \
  --output "$module_generated"
cp "$capacity_generated/compiler-packer.hepta" "$module_generated/compiler-packer.hepta"
tools/modules/rebuild.sh "$expanded_compiler" "$module_generated" "$module_native"
```

Python runs only the pinned source authoring stage. The scripts first verify
upstream source hashes, prove unchanged baseline generation, apply explicit
transformations, check the generated ordinary HIA with the unchanged host
parser, and bind the exact resulting source hashes. Their receipts state that
native compilation has not occurred at that point. `rebuild.sh` then performs
native compilation and HIA-native packing with an empty runtime `PATH`.

The result is:

| Native artifact | SHA-256 |
| --- | --- |
| `module-compiler` | `e250d93bc0b2a5ff878d2188c95780abf7e7ebaf186e4531b136fa2735b2af59` |
| `module-linker` | `1ba40ffe025abfcc2edd8d69f1ad689817598571138c11806af7aa60047fc7b8` |
| `M2.image` | `3382734ac6e05196bd01ef395729710d783f69fc1302830b27318d4fe8405832` |
| `module-linker.image` | `03a9ba5b2220da6c00cdac54786a5d26493af0254c465581ebb4dc2429c8fa4d` |

The native module compiler has 110 functions, 1,461 blocks and 7,600
instructions. The native linker has 27 functions, 570 blocks and 2,479
instructions. All three compiler generations and both packers must reproduce
byte for byte before the build recipe succeeds. The source and final binary
pins reject changed inputs; the adapters do not modify the upstream checkout.

## Compile the actual runtime as independent units

```bash
python3 -I -S -B tools/modules/split_runtime.py \
  --source src/heptatorch.hepta --output "$runtime_units"
tools/modules/compile_link.sh \
  "$module_native/module-compiler" "$module_native/module-linker" \
  "$runtime_native" entry \
  "$runtime_units/ht_runtime.hepta" "$runtime_units/ht_kernel.hepta"
```

`runtime_units` and `runtime_native` must be new output directories. The
partitioner records the complete function inventory and each unchanged body
hash. It constructs real `external()` declarations from the copied
signatures. It neither compiles source nor acts as the module admission gate.
The gate is in `module-linker`, and every source unit is independently checked
by `module-compiler`. The resulting application is
`$runtime_native/application`; its public packet and three-buffer ABI remains
the ordinary runtime ABI.

`compile_link.sh` accepts arbitrary one-to-64 module sources and the exact
entry symbol. It transports a `HIABLK01` frame. Each source compiles separately
to an ELF relocatable object with a `.hia.interface` section. Imported calls
are genuine `R_X86_64_PLT32` fixups with addend -4. The native linker resolves
them to the exporting definition and emits a syscall-only Linux executable.
The source splitter is specific to the current runtime's two-unit boundary;
the native compiler and linker are not tied to a single application hash.

## Native qualification gates

Every output directory below must be new. The tests are portable and run
native processes with an empty `PATH`. They use Python for source/packet
construction, observation and assertions, not production HIA admission or
numerical execution.

```bash
python3 -I -S -B tools/modules/tests/boundaries.py \
  --compiler "$module_native/module-compiler" --linker "$module_native/module-linker" \
  --legacy-compiler "$expanded_compiler" --output "$reviews/boundaries"
python3 -I -S -B tools/modules/tests/contracts.py \
  --compiler "$module_native/module-compiler" --linker "$module_native/module-linker" \
  --output "$reviews/contracts"
python3 -I -S -B tools/modules/tests/capabilities.py \
  --compiler "$module_native/module-compiler" --linker "$module_native/module-linker" \
  --output "$reviews/capabilities"
python3 -I -S -B tools/modules/tests/abi.py \
  --compiler-image "$module_native/M2.image" --linker-image "$module_native/module-linker.image" \
  --boundary-output "$reviews/boundaries" \
  --runtime-request "$runtime_native/link-request.bin" \
  --runtime-executable "$runtime_native/application" --output "$reviews/abi"
python3 -I -S -B tools/modules/tests/equivalence.py \
  --baseline-compiler "$expanded_compiler" --module-compiler "$module_native/module-compiler" \
  --output "$reviews/ordinary" src/heptatorch.hepta tools/unhex.hepta
python3 -I -S -B tools/modules/tests/fixtures.py \
  --executable "$runtime_native/application" --fixtures tests/fixtures \
  --split-manifest "$runtime_units/split-manifest.json" --report "$reviews/fixtures.json"
```

The boundaries suite has 69 assertions covering native admission and actual
linked execution, including 256 borrowed exports and a 257-definition linked
program that exceeds the per-unit function limit. The independent contracts
suite has 10 linked executions and 194 malformed-object refusals. The
capabilities suite has 19 checks covering private returned loans, sibling
read-only aliases, CFG provenance joins and refused aliases/effects. The
flat-ABI suite has 18 guarded observations with clean and dirty workspaces,
immutable input and failure output atomicity. Passing these finite suites is
not a proof of every possible program or malformed object.

## CI integration and historical comparisons

Keep the existing native runtime fixture gates, the 16 expanded-compiler
capacity checks and the real-PyTorch oracle. Run the current fixture inventory
and the same real-PyTorch oracle against both monolithic and linked runtimes.
The ordinary-mode comparison above proves that adding module support did not
change code emitted for the complete current runtime and unhex utility.

The original `87c247e4...` compiler only accepts 128 functions and 1,536 blocks.
The current runtime exceeds that profile. Do not compare that old compiler
against current full source and interpret a refusal as a regression. Preserve
the former main `31d5b1483714b32254730ba56787884c5c76e2fb` runtime as the fixed
historical witness:

| Historical witness | SHA-256 |
| --- | --- |
| HIA source | `4c1d17240dd8616798e238cdd54641ee6f2e42c02635c82b3bdeea0a530b5ca5` |
| Native executable | `ecca265b088e0688df8ad963613f6bdf460efd324d0c480e6395e4ef67372779` |

Compare old and expanded compilers on that pinned historical source; compare
expanded and module compilers on the current complete source. Preserve the
275 historical fixture identities and grow the runtime fixture suite without
describing new rows as historical coverage. Generated receipts must identify
which source and executable each suite actually exercised.
