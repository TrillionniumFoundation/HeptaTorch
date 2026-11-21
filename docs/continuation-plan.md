# Whole-PyTorch HIA rewrite: execution and acceptance plan

## Objective and source of truth

Complete the entire pinned PyTorch v2.15.0 project in native HIA, preserve upstream behavior, then retire replaced implementations and redundant process versions. The upstream commit is `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`. Keep **2,590 native declarations, 688 derivative declarations and 17 full subsystem contracts** in the target.

Resume from current Git heads/open PRs, [`full-acceptance.json`](../compatibility/full-acceptance.json), the capability inventory, [current audit](current-audit.md) and actual source-bound reports. Newer checked-in results take precedence over chat snapshots. Preserve concurrent work and publish with expected-head leases.

The canonical application is `tools/current_runtime/source/`; its manifest is authoritative. The current candidate has 773 core functions / 15 units and requires **28 groups / 13,075 finite cases**, including separately executed Generator probes. Record the actual complete result, never the sum of unrelated partial successes.

## Complete active integration

1. PR48 is merged at `6a031f2d1a25e5c4ccbc1efc3dc13bd71175544e`. Its actual765 image passed24 groups/3,845 cases, all ten head checks and complete independent original artifact readback. The merged tree exactly matches the tested tree; see [the fixed review](../validation/current-runtime765/ci-review.json).
2. PR49 and PR50 are now merged at `e4d43b8fa4b4749ff2b5638cb70608959cc9633c` and `094abcd2b69c64041c879d489ef602a0ab6187c7`, after independent complete25-group/4,250-case runs, all13 exact-head checks and full original-artifact readback. Their merged trees equal their tested trees. Preserve the earlier original495-case admission failures. PR51's original26-group/5,446-case execution passed, but final artifact metadata GET returned404. Its repair `8a93be6867d6e12877b8645068decfc3b78ede71` adds bounded same-identity metadata retries with23 passing transport tests; its new26/5446 run, all13 exact-head checks, complete original Current/Native/observer archives, cold builds and separately linked Generator probes passed. PR51 is merged at `e1a2f67e894b3d9b823e9fbcc5dfc79e12275757`; merged and tested trees are both `512127e458e9778513bb1eb830304f82cbac25fc`. Current773 includes this repair. Preserve all original failed runs and artifact identities.
3. Complete the same-F64773 successor's **28 groups / 13,075 cases** on one actual cold-built image. Its focused same-F645088, mixed-F642541, original F6448 and public F64 logical192 runs have each passed actual execution and their original physical readers on the exact new source/image. These are separate scoped runs and do not establish the full28 gate. The new guard tests promoted result dtype4, so all original Half groups also remain mandatory.
4. Preserve every failed attempt. Same-F64 v1 failed case2224; v2b retained all3288 original successes but failed nine of1800 appended singleton cases. The current source reuses the existing dimension order and keeps all5088 recipes and original references. PR52's full27 run passed26 groups/7795 cases and failed public_f64_logical192 during preparation because its outer wrapper had stale derive736/run736 pins. The new successor fixes those pins and has actually re-executed192; never relabel PR52 as a pass. Also preserve setup/outer-supervisor failures separately from actual child results. Earlier unpinned builds, incorrect new Bool/unused-word test assumptions and unrecovered native769 claims remain distinct.
5. Verify historical143 tracked-source retirement on the final tree. The candidate removes481516 bytes of old source, preserves the exact restorer/manifest/16 controls and all old Native107/F64/RNG8 test steps, and fixes the missing test-storage-objects restore prerequisite. Current CI requires the old source absent before and after qualification; legacy CI requires actual cold recovery from absent source/cache. Fresh HTTPS restoration, cache restoration, frozen source checks and eight Make dependency plans passed, but all final-tree pipelines are still mandatory. Complete semantic migration of the remaining old-protocol consumers separately.
6. Admit a final head only after all exact-tree checks and complete original artifact readback, verify the remote merged tree and use an expected-head lease. PR52's failed head must not be merged as passing; the fully qualified successor can supersede it. Preserve the fixed historical identities for the116 earlier process-file retirements and13 replaced source units. The new candidate keeps only its active qualification plan, with the previous plan available at fixed PR52 history.
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
