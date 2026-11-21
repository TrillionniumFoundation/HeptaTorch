# Whole-PyTorch HIA rewrite: execution and acceptance plan

## Objective and source of truth

Complete the entire pinned PyTorch v2.15.0 project in native HIA, preserve upstream behavior, then retire replaced implementations and redundant process versions. The upstream commit is `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`. Keep **2,590 native declarations, 688 derivative declarations and 17 full subsystem contracts** in the target.

Resume from current Git heads/open PRs, [`full-acceptance.json`](../compatibility/full-acceptance.json), the capability inventory, [current audit](current-audit.md) and actual source-bound reports. Newer checked-in results take precedence over chat snapshots. Preserve concurrent work and publish with expected-head leases.

The canonical application is `tools/current_runtime/source/`; its manifest is authoritative. The current candidate has 797 core functions / 15 units and requires **35 groups / 16,341 finite cases**, including separately executed Generator probes. Record the actual complete result, never the sum of unrelated partial successes.

## Complete active integration

1. PR48 is merged at `6a031f2d1a25e5c4ccbc1efc3dc13bd71175544e`. Its actual765 image passed24 groups/3,845 cases, all ten head checks and complete independent original artifact readback. The merged tree exactly matches the tested tree; see [the fixed review](../validation/current-runtime765/ci-review.json).
2. PR49 and PR50 are now merged at `e4d43b8fa4b4749ff2b5638cb70608959cc9633c` and `094abcd2b69c64041c879d489ef602a0ab6187c7`, after independent complete25-group/4,250-case runs, all13 exact-head checks and full original-artifact readback. Their merged trees equal their tested trees. Preserve the earlier original495-case admission failures. PR51's original26-group/5,446-case execution passed, but final artifact metadata GET returned404. Its repair `8a93be6867d6e12877b8645068decfc3b78ede71` adds bounded same-identity metadata retries with23 passing transport tests; its new26/5446 run, all13 exact-head checks, complete original Current/Native/observer archives, cold builds and separately linked Generator probes passed. PR51 is merged at `e1a2f67e894b3d9b823e9fbcc5dfc79e12275757`; merged and tested trees are both `512127e458e9778513bb1eb830304f82cbac25fc`. Current797 includes this repair. Preserve all original failed runs and artifact identities.
3. Complete the current797 successor's **35 groups / 16,341 cases** on one actual cold-built image. All29 prior groups/13851 cases remain mandatory, followed by signbit97, typed_index_select866 and indexed_f32 331; additionally require forward_nograd462, join_forward414 and spatial_forward320. Local actual native/physical observations pass, including the repaired F32 no-grad index selection, but the local oracle lacks full installation provenance. Strict CI must execute every group with all11 original provenance checks, independently pinned complete official wheel/install bytes, before/after executable mappings and complete raw frame/owner/process readback. Preserve the original joint-RECORD forgery and mapping counterexamples; a revised pure reader must reject them.
4. Preserve every failed attempt. PR52 failed public_f64_logical192 preparation; its7795 passed cases do not qualify the whole27 gate. PR53 failed Bool27/Integral24/F32-Sum17 source admission before native cases; its25 passed groups/13007 cases do not qualify the whole28 gate. Fix the shared stale source-helper pin, preserve the original strict native-executed check, and execute all three affected groups plus full35 anew. Preserve earlier same-F64 NaN failures and all setup/transport/harness failures separately from actual native observations.
5. Verify historical143 tracked-source retirement on the final tree. The candidate inherits the removal of481516 bytes of old source, the fixed restorer/manifest/16 controls, all original Native107/F64/RNG8 test steps and the repaired test-storage-objects restore prerequisite. Current CI must actually verify source absence before and after qualification; PR53 did not execute its last two guards after its failure. Legacy CI must actually restore from absent source/cache. The RNG consumer now reuses the exact current rng03 object for all185 original cases and no longer restores143; its strict new-tree CI is required. Complete semantic migration of Native/F64 and explicit historical Make consumers remains unfinished.
6. Admit a final head only after all exact-tree checks and complete original artifact readback, verify the remote merged tree and use an expected-head lease. Neither failed PR52 nor failed PR53 is a passing baseline; only a fully qualified successor may supersede them. Preserve the PR54 complete29 failures and PR55 admission repair independently. Keep one active797 qualification plan and15 active source units; preserve replaced plans/partitions at their fixed Git identities. The next F64 public-view extension remains isolated until its own exact source, oracle, alias/version/lifetime/error/OOM checks and all affected old-control successors are validated.
7. The full-contract admission mechanism is implemented and tested, but its production verifier registry is empty. Implement native contracts and independent complete verifiers before registering targets and changing closure counts. The30 denominator/false-claim controls and25 admission-fixture tests qualify only the framework, not any PyTorch declaration.

## Native implementation order

Independent work may proceed in parallel when it does not assume unfinished dependencies.

| Priority | Implementation | Required evidence |
| --- | --- | --- |
| 1 | General Tensor ABI, allocator/Storage lifetime, schema/dispatch/meta and dtype/layout iteration | Full-contract shape/dtype/aliasing/mutation/error/resource tests and declaration-level qualification |
| 2 | Remaining operators, promotion, indexing, views, out/inplace | Independent complete overload/backend/layout comparisons |
| 3 | Public frontend, Module/Parameter/buffer trees and full AD | Real application behavior; forward/higher-order AD, hooks, custom functions and transforms |
| 4 | Optimizer/RNG services and durable serialization | Training, checkpoint/restore, aliases, state_dict, failures, interruptions and retention |
| 5 | Device allocation/transfers/streams/events and native kernels | Actual required hardware or identified emulator, synchronization and errors |
| 6 | Distributed, compiler/export, data pipelines and platforms | Multi-process/device behavior, graph/export, worker lifecycle and installation |
| 7 | Final upstream/application/performance/resource acceptance and retirement | Independent full review of all 17 contracts, then migration/deletion of replaced active paths |

The current acceptance checker validates accounting and source-registered complete-contract evidence. Its production registry is empty, so all full qualification counts remain zero. Implement full-contract runners alongside the missing native functionality before closing complete gates; editing status metadata is insufficient.

## Evidence and cleanup

- HIA executes numerical kernels, gradients, state changes and runtime services. Python/Torch references and bootstrap hosts retain their explicit roles.
- Bind results to exact source, toolchain, image, inputs and oracle. Keep original failures/stopped runs unchanged and rerun affected gates on new images.
- Derive expectations independently and fix tolerances before measurement. Exact source inverses complement actual execution.
- Retain complete declarations and all required platforms/layouts/backends. Finite CPU success does not close an entire subsystem.
- Delete a source/archive only after migrating and verifying its live consumers. Retain licenses and required reference/bootstrap inputs; keep completed process narratives in fixed Git history.
- Continue substantive implementation while full requirements remain open. Status polling, repeated audits, stubs or reduced targets do not complete this work.

`make acceptance-status` checks target accounting. `make full-acceptance` remains unsuccessful until full native contracts and independent verifiers are complete.
