# Whole-PyTorch HIA rewrite: execution and acceptance plan

## Objective and source of truth

Complete the entire pinned PyTorch v2.15.0 project in native HIA, preserve upstream behavior, then retire replaced implementations and redundant process versions. The upstream commit is `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`. Keep **2,590 native declarations, 688 derivative declarations and 17 full subsystem contracts** in the target.

Resume from current Git heads/open PRs, [`full-acceptance.json`](../compatibility/full-acceptance.json), the capability inventory, [current audit](current-audit.md) and actual source-bound reports. Newer checked-in results take precedence over chat snapshots. Preserve concurrent work and publish with expected-head leases.

The canonical application is `tools/current_runtime/source/`; its manifest is authoritative. The current candidate has 773 core functions / 15 units and requires **27 groups / 7,987 finite cases**, including separately executed Generator probes. Record the actual complete result, never the sum of unrelated partial successes.

## Complete active integration

1. PR48 is merged at `6a031f2d1a25e5c4ccbc1efc3dc13bd71175544e`. Its actual765 image passed24 groups/3,845 cases, all ten head checks and complete independent original artifact readback. The merged tree exactly matches the tested tree; see [the fixed review](../validation/current-runtime765/ci-review.json).
2. PR49 and PR50 are now merged at `e4d43b8fa4b4749ff2b5638cb70608959cc9633c` and `094abcd2b69c64041c879d489ef602a0ab6187c7`, after independent complete25-group/4,250-case runs, all13 exact-head checks and full original-artifact readback. Their merged trees equal their tested trees. Preserve the earlier original495-case admission failures. PR51's original26-group/5,446-case execution passed, but final artifact metadata GET returned404. Its repair `8a93be6867d6e12877b8645068decfc3b78ede71` adds bounded same-identity metadata retries with23 passing transport tests; finish its new complete tree gate and original readback before merge. Current773 includes this repair. Preserve all original failed runs and artifact identities.
3. Complete current773 **27 groups / 7,987 cases** on one actual cold-built image. The new mixed public F64 arithmetic2541 and affected original F6448 passed actual execution and independent physical readback, sharing the same actual2541 reference for23 arithmetic successors. The complete87 F64 requests are63 refusals,one negation,22 mixed single commands andone original FE transaction. These2,589 scoped cases do not substitute for the full27-group gate. All26 preceding groups and original Half239 controls remain mandatory.
4. Preserve every failed original attempt. The new arithmetic reference initially used the wrong Bool error type, and its first native control set incorrectly required refusal for unused command words; both failures remain immutable. Their corrected v3 test uses the same HIA source. Previous unpinned builds, admission failures and unrecovered native769 claims stay distinct. Same-F64 NaN payload parity is the next concrete native gap; proceed from actual fixed-Torch differences in a separate source candidate, keeping the real function count rather than using it as an artificial version number.
5. Finish historical143 source provenance migration phase1 on the exact new tree: original Native107 and F64/RNG8 steps are preserved, and real download/cache/existing restoration plus16 refusal controls passed. Keep tracked old source until all three old gates pass. Then remove it in a separately verified source-restoration change while continuing semantic migration of every old consumer.
6. Admit each final head only after all required checks and complete original artifact readback, verify its remote tree and merge with an expected-head lease. Retire the116 new process files and13 replaced source units only with the exact documented original Git identities and current-tree gate. Continue the implementation order below; scoped successes do not close the17 full subsystem contracts.

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

The current acceptance checker validates accounting consistency. Implement full-contract runners alongside the missing native functionality before closing complete gates; editing status metadata is insufficient.

## Evidence and cleanup

- HIA executes numerical kernels, gradients, state changes and runtime services. Python/Torch references and bootstrap hosts retain their explicit roles.
- Bind results to exact source, toolchain, image, inputs and oracle. Keep original failures/stopped runs unchanged and rerun affected gates on new images.
- Derive expectations independently and fix tolerances before measurement. Exact source inverses complement actual execution.
- Retain complete declarations and all required platforms/layouts/backends. Finite CPU success does not close an entire subsystem.
- Delete a source/archive only after migrating and verifying its live consumers. Retain licenses and required reference/bootstrap inputs; keep completed process narratives in fixed Git history.
- Continue substantive implementation while full requirements remain open. Status polling, repeated audits, stubs or reduced targets do not complete this work.

`make acceptance-status` checks target accounting. `make full-acceptance` remains unsuccessful until full native contracts and independent verifiers are complete.
