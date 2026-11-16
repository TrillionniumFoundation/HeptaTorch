# Whole-PyTorch HIA rewrite: execution and acceptance plan

## Objective and source of truth

Audit HeptaTorch and replace the entire pinned PyTorch project with native HIA
implementations, preserving the upstream behavior and closing every acceptance
blocker. The target remains PyTorch **v2.15.0**, commit
`31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`. The 2,590 exact native declarations
in `compatibility/upstream-schemas.json` are one part of this target; they do not
replace frontend, runtime, backend, distributed, compiler or ecosystem work.

This is the single continuation plan. Read current Git commits, open pull
requests, `compatibility/full-acceptance.json`, the capability inventory and
source-bound validation receipts before resuming. A newer checked-in result
takes precedence over a previous chat or automation snapshot. Preserve active
work and use expected-head leases when publishing a concurrent branch.

**Whole-project acceptance is currently HOLD.** A green component CI run is
evidence only for the exact cases and source it executed. It cannot close an
entire subsystem or a complete upstream schema by itself.

## Current implementation and next work

The tensor-composition source is `264e9c5c4e8ea1834724cc0d8d859952016daffe4af330a8daa7745e7a5e5233`.
It contains 143 HIA functions, 1,783 blocks and 7,064 instructions. The default
native build compiles two complete source units independently and resolves 1,108
symbolic call relocations with the HIA linker. The monolithic build remains an
independently compiled comparison. Both actual build paths pass all 461 native
packet pairs, including 186 new pairs. Their source, compiler and executable
identities are recorded in `validation/`.

| Workstream | Implemented and locally qualified | Remaining next work |
| --- | --- | --- |
| Native module compilation | True external declarations, borrowed buffer signatures, effects and lifetime checks, symbolic ELF relocations, two-unit execution and 69/204/19/18 boundary suites | General owned-resource ABI, allocation and resource results; persistent storage and whole-program resource bounds |
| Tensor composition | Native `cat` and `stack`, variable-parent reverse AD, empty shapes, layout selection, version checks and graph reset | Complete dtype/device/layout variants, mutation/out forms and full upstream schema qualification |
| Indexed accumulation | Native `scatter_add` and `index_add`, exact integer index payloads, duplicate accumulation, alpha rounding and reverse AD preflight | Complete dtype/device/layout variants, mutation/out forms and full upstream schema qualification |
| Persistent execution | Two native interactive builds, complete-frame arena transactions, 477 independent live-session cases each, 22 packer / 18 flat-linker / 3 actual growth checks | Both hosted session images now pass 21 cases / 4,681 frames / 491,847 values each; general owned/resizable storage, absolute torch._version parity and cross-process durability remain open |
| Qualification | Both ordinary builds pass 461 packet pairs and 270 pinned PyTorch cases each, with 13,076 compared values and 11 provenance checks per build; all 159 prior case functions remain unchanged | Extend source-bound qualification with every new implementation; component success does not close the whole-project gates |

`validation/hosted-boundary-observations.json` contains actual pinned PyTorch
observations used to establish empty, index, alpha and layout behavior. Those
observations alone do not qualify the changed native runtime.
`validation/hosted-tensor-composition.json` records the successful actual
dual-build run 43788929713 at commit
`724cbf2eb19d2ef8d400615d7196f5397cb402f6`, binding the new source, both executables,
oracle and fixture inventory. Historical 159-case receipts keep their identities.

The borrowed module barrier is now implemented; it does not establish owned
resource transfer, general allocation or a general public tensor object ABI.
The next foundation work is explicit native storage ownership and allocation,
followed by general owned persistent state and dtype/layout services. Bounded borrowed-arena persistence is implemented separately in `tools/session`; it does not establish external heap-owner transactions. Independent
operator and AD work may continue without treating restricted packet features
as complete upstream schema coverage.

## Dependency order

1. Establish independent compilation with checked tensor/resource interfaces.
   Keep one authoritative implementation of each numerical and tensor behavior.
2. Replace fixed descriptor/arena/request lifetimes with a general storage and
   allocator model, persistent execution state and an explicit device boundary.
3. Complete dtype/layout/promotion and views/mutation foundations. Expand the
   operator and AD implementation against the frozen schema inventory.
4. Complete frontend/dispatcher/module/optimizer/RNG/serialization services.
   Qualify actual application training and restoration, including negative and
   interrupted execution cases.
5. Implement and qualify every targeted device and platform, distributed
   execution, transforms/compiler/export, data pipelines and extension APIs.
6. Run the original full acceptance matrix with independently reproducible
   evidence, retire replaced implementation paths and finish packaging.

Independent work may proceed in parallel when it does not assume an unfinished
dependency. Hardware-specific claims require execution on the corresponding
hardware or an explicitly identified emulator; CPU results do not establish GPU
or distributed correctness.

## Evidence and closure rules

- Keep every upstream schema and subsystem in the denominator. Restricted CPU
  binary32 support stays in `capabilities.json`, independently of full-schema
  qualification.
- Native HIA must execute numerical kernels, gradients, state updates and
  runtime services. Test-only Python/PyTorch references and compiler bootstrap
  hosts must remain explicit. Forwarding work to an upstream numerical runtime,
  an interpreter, a constant success result or an unexecuted stub is not a rewrite.
- Bind every result to the exact source, compiler profile, executable, input
  corpus and oracle revision. Rerun affected gates after changes; retain earlier
  FAIL, HOLD and NOT_RUN records with their original identities.
- Derive expected values independently. Freeze tolerances before measuring a
  candidate. Do not turn observed native output into its own reference.
- A subsystem may close only when its complete contract, supported platforms,
  upstream tests, adversarial failures and integration behavior have executable
  qualification. Metadata consistency is necessary but is not semantic proof.
- Publish verified increments through reviewable Git changes and read back the
  exact remote tree. Do not overwrite concurrent changes or force-push history.
- Continue implementation while any item is open. Stop only after independent
  final acceptance confirms the complete target, or the user cancels the work.

`make acceptance-status` validates that the recorded scope and blocker ledger
remain internally consistent. `make full-acceptance` uses the same checks and
returns status 2 while the whole rewrite remains unqualified. Neither target
replaces the native regression suite or the full subsystem qualification work.
