# Native stack budget and guarded execution

This is the separate `heptatorch.hia.stack-budget-modules.v1` profile, interface ABI 3. It adds native whole-program call-path admission and an explicitly allocated, guarded stack. It preserves the existing borrowed and owned ABI 1/2 profiles and their pinned tools. It does not rename an older compiler or increase its per-unit limits.

The profile now compiles and links the complete current 151-function Tensor core, including its actual owned state entry. All three input units, all core function bodies, each object's machine text, and the final linked application text after offset 4096 remain byte-identical to the qualified ABI 2 application. Only the distinct object metadata and startup profile change. The guarded Tensor image is pinned in `profile.json`; its admitted call path is 61,488 bytes plus a 4,096-byte reserve.

## What executes natively

The HIA compiler emits the stack records and call matrix. The HIA linker validates the ELF/interface/resource contracts, checks real prologues, sequentially decodes the pinned backend's instruction subset, reconstructs actual private and external calls, compares the entire matrix, resolves cross-unit paths, and rejects an excessive budget. The emitted ELF uses Linux syscalls to allocate two guard pages around a private stack and release it after normal cleanup. Tensor arithmetic, ownership transfers, allocation, AD and optimizers execute in HIA-produced native code.

Python is a build authoring and qualification dependency. It adapts exact pinned upstream source into scratch HIA, partitions complete compiler functions, invokes native tools, and checks receipts. It supplies no numerical implementation, owner allocator, module interpreter or production stack verifier. GNU `objdump` is used only as an independent instruction-boundary reference in the adversarial qualification suite. The dirty-workspace observer uses host `mmap`/`mprotect` solely to invoke the actual native entry with guarded observation buffers.

## Rebuild

First reproduce the existing `tools/modules`, `tools/owned_modules` and `tools/owned_sessions` tools using their own unchanged recipes. The owned and session build directories must include their generated source receipts. The upstream checkout must be the exact commit in `profile.json`; generation verifies its imported source closure before and after use.

```sh
python3 -I -S -B tools/stack_budget/rebuild.py \
  --source-root /absolute/path/to/pinned-Hepta \
  --owned-build build/owned-modules \
  --session-build build/owned-sessions \
  --borrowed-tools build/modules \
  --output build/stack-budget
```

The defaults for `--owned-profile`, `--session-profile` and `--session-template` resolve to this repository's frozen packages. Explicit paths are supported for an isolated reproduction. The output directory must not already exist.

The aggregate compiler contains 8,359 instructions, above the unchanged 8,192-instruction unit limit. `bootstrap.py` splits its complete function bodies into a frontend/backend DAG and records every body hash. The old ABI 2 compiler and linker are identified only as stage 0. Three successive ABI 3 compiler products converge byte-for-byte, as do two successive generations of all nine linkers. The final compiler is reproduced again with the final guarded linker. All final tools themselves use the admitted, guarded startup.

Outputs are `native/stack-module-compiler`, `native/stack-linker-session`, `native/stack-linker-compiler`, `native/stack-linker-linker`, and `native/stack-linker-runtime`. The remaining five variants are explicitly qualification-only. See `CONTRACT.md` for their request identities and buffer capacities.

## Qualify

The current Tensor version build and ordinary canonical application are prerequisites. Build them using their original recipes before this command; they are read-only reference inputs.

```sh
python3 -I -S -B tools/stack_budget/qualify.py \
  --build build/stack-budget \
  --repo . \
  --tensor-baseline build/owned-tensor-versions \
  --canonical-binary build/heptatorch \
  --service-sources \
    tools/storage_objects/candidate-04/bytes.hepta \
    tools/storage_objects/candidate-04/records.hepta \
    tools/storage_objects/candidate-04/audit.hepta \
    tools/storage_objects/candidate-04/ops.hepta \
    tools/storage_objects/candidate-04/owners.hepta \
  --output build/stack-budget-review
```

The driver retains exact requests, compiler/linker stdout and stderr, source units, native images, observations and JSON receipts. It checks 86 native gate cases:

| Suite | Cases | Concrete checks |
|---|---:|---|
| Stack boundaries and live cleanup | 43 | Exact 64 KiB qualification boundary; fanout versus maximum path; 10,000 tail-self iterations; real guard faults; owned service execution and cleanup |
| Production capacity | 5 | All 256 function slots; all 64 object slots in both orders; a real 13,100-function call path exactly at the 8 MiB production budget; 16 bytes over rejected |
| Dirty workspaces | 6 | Actual compiler and linker entries on clean, `0x5a` and reused scratch, with source/output/workspace canaries |
| Actual call and helper binding | 32 | Single-bit edge deletion; private call targets; relocation inside an immediate; duplicate/unsorted relocations; branch boundaries; fixed helper bytes; real three-owner tuples and signed/bulk operations |

It then compiles the unchanged current Tensor units and runs the existing 11 native Tensor groups, 161 replies, 1,078,028 binary32 comparisons and 59,310,080 rollback bytes. The original peer numerical review runs 16 frames, compares 612 scalar/binary32 fields and 41,943,040 rollback bytes. The two earlier, source-backed optimizer version expectation corrections are recorded as reversible text edits; peer source changes only the target image pin. Numerical assertions and tolerances are unchanged. The strict real-PyTorch oracle is a separate CI gate and is not claimed by this driver.

## Scope and retained evidence

The trust boundary is a byte-controlled product of the pinned HIA producer. The native gate now binds stack metadata to actual calls and actual prologues; it does not authenticate arbitrary foreign machine code, verify arbitrary `rsp` writes, or re-prove every machine instruction's resource effects. The full Tensor and PyTorch compatibility work remains open outside this finite profile.

An earlier development prototype accepted objects whose call-matrix bits were cleared while the machine calls remained. That false admission is retained in the validation history and is superseded by native instruction-boundary reconstruction, exact matrix comparison and fixed-helper checks. No release claim relies on the earlier prototype.

All source, startup, native tool and application identities are explicit in `profile.json`, `source-pins.sha256`, the rebuild receipt and `validation/native-stack-budget.json`. Upstream-derived full compiler sources are generated into the build directory, not vendored here. This adaptation retains upstream provenance without asserting a license that the upstream project does not provide.
