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

The single canonical HIA application is `tools/current_runtime/source/`.
Its manifest is authoritative for complete source, functions, ordered units
and native object/image identities. `make` cold-builds every application object
from this source with the fixed ABI3 HIA compiler and session linker.

`make test` and current-runtime CI share thirteen mandatory groups totaling 697
finite cases: original Bool27, view/scalar76, Integral24, integral-to-F32 Sum17,
original lifecycle35, original operator/training270, F32 Sum32, live
Tensor/RNG/backward32, logical147, inherited resource7, original AsStrided16,
Cat4 and independently referenced cat Metadata10.
The three live groups use the fixed actual
Torch oracle; all frozen groups preserve original observations. New source
identities require new execution of every affected group.

The first nine-group 660 run remains FAIL: eight groups passed, while the
current35 strict/control stages rejected cross-version AST identities after
the original 35 native/live cases passed. Read-only diagnosis confirmed seven
AST digest differences between CPython 3.12 and 3.14. Run the coordinator and
all stages under the same pinned CPython 3.12; keep every original identity and
numerical assertion. The second ten-group 667 execution actually passed all
groups, with every suite executed anew, all inputs unchanged, and complete
outer exit/EOF. Its original qualification SHA is
`aa168047fc9a0392707f3c1c08b8ab4cb88c2bc16e1732c4c37c2a3994cc14c0`.

The inherited resource7 group preserves R4 resource3, backward private cache1
and R3 preserve3. An independently cold-built current image has passed all
seven original recipes, 378 frames and 1,060 checks, including three real
RLIMIT allocation refusals and successful recovery after restoring the limits.
Whole-file source inverses preserve the original recipes and budgets. This
separate result was also rerun successfully inside the final common 667
qualification. After compact reference migration, the same seven recipes
were independently executed again and passed. Original76, Sum32 and the
strengthened live32 also passed all 140 cases in a separate post-migration
run, with every original assertion and all 32 live native exits/64 EOFs checked.
Final AsStrided16 separately passed all four original suites, 1,920 frames,
10,272 checks, complete owner archives and actual native exits/EOFs. These
executions have distinct source and result receipts. The complete 683 gate later
passed on PR40's exact published tree, and PR40 was merged as
`584b95d7f6615f27e95618bc61215df5633c4ca4`. The current cat correction changes
the image again, so the new common 697 gate requires its own complete execution.
Do not attribute new source or added cases to an earlier immutable run.

The implementation combines the prior integral-to-F32 and empty-cat fixes with
native any/all and extended Bool sum. It retains the source-proved eight-lane
sum accumulation order, converts each integral element before F32 reduction,
reclaims private temporaries, and preserves the empty-cat gradient cache's
success and ownership guards. Continued actual metadata comparison found a
remaining empty-cat Version mismatch. The current correction routes rank-one
empty cat inputs through the existing zero factory, guards saved-list allocation
failure, and removes the unnecessary empty-axis prefix scan and ordinary
metadata parent-mask lookups. The original four packets and ten independent
Torch metadata recipes pass on the new cold-built image. Logical reductions
share native planning and truth conversion. Independent error-priority probes
also require axis errors to precede the Bool-sum autograd refusal.

The latest actual results, failures and source identities belong in the
[current audit](current-audit.md) and its bound execution manifest. Source
preparation, source inverses and cold compilation are recorded separately from
runtime PASS. Earlier failures remain tied to their original inputs and image.

Continue migrating remaining historical gates onto the canonical image before
removing their live dependencies. Generator24 still requires native raw-output,
epoch/allocation contracts and real entropy; a synthesized host output is not
an equivalent implementation. The original270 numeric suite does not replace
461 legacy raw-protocol fixture pairs or the separate full resource gates.
The F64 oracle still uses the old complete143 source as a numerical-origin
witness; migrate that proof before deleting `src/heptatorch.hepta`. Native
compiler bootstrap packages and their fixed inputs are a separate dependency
family required by clean toolchain reconstruction.

Remove unreachable application functions and duplicate current partitions once
source reachability and actual regressions justify their retirement. Keep one
current full35 entry and one canonical application implementation. Necessary
original reference fixtures and licenses are part of the final test contract;
repeated process narratives and obsolete default profiles are not.

Then complete the missing dtype/promotion, operator, AD, frontend/dispatch,
serialization, device, distributed, compiler/export, data-pipeline and platform
contracts against the unchanged complete inventory. All seventeen whole-system
gates remain open or in progress until their actual full requirements pass.

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
