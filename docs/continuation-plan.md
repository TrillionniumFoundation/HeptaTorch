# Whole-PyTorch HIA rewrite: execution and acceptance plan

## Objective and source of truth

Complete the entire pinned PyTorch v2.15.0 project in native HIA, preserve upstream behavior, then retire replaced implementations and redundant process versions. The upstream commit is `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`. Keep **2,590 native declarations, 688 derivative declarations and 17 full subsystem contracts** in the target.

Resume from current Git heads/open PRs, [`full-acceptance.json`](../compatibility/full-acceptance.json), the capability inventory, [current audit](current-audit.md) and actual source-bound reports. Newer checked-in results take precedence over chat snapshots. Preserve concurrent work and publish with expected-head leases.

The canonical application is `tools/current_runtime/source/`; its manifest is authoritative. The current candidate has 767 core functions / 15 units and requires **25 groups / 4,250 finite cases**, including separately executed Generator probes. Record the actual complete result, never the sum of unrelated partial successes.

## Complete active integration

1. PR48 is merged at `6a031f2d1a25e5c4ccbc1efc3dc13bd71175544e`. Its actual765 image passed24 groups/3,845 cases, all ten head checks and complete independent original artifact readback. The merged tree exactly matches the tested tree; see [the fixed review](../validation/current-runtime765/ci-review.json).
2. Complete the current767 **25 groups / 4,250 cases** on one actual cold-built image. The new405 I64/F64 negation cases and original F6448 successor have passed on767, including independent physical replay. Their finite results do not substitute for the complete25 group gate. Preserve all24 predecessor groups, the239-request Half partition, and every original F64 request:86 refusals plus one exact numeric negation successor backed by the same actual405 reference.
3. Preserve the observed failures and strict admission. The previous fourteen-unit live32 rejection, missing-NumPy reference rejection, omitted FE owner transitions and preliminary unpinned767 manifest rejection remain distinct original failures. They are not rewritten as successes. Use the canonical source manifest and complete source inverses with actual execution.
4. Recover the previously documented native769 Half true-division candidate from its exact original source/build/evidence before integration. Its files are not present in the merged main tree; the older plan's separately reported1128/312/1006 results are unverified until recovered. Preserve the current767 increment. A future combined negation-plus-division image must derive its own identity and run all26 groups/5,378 cases if it retains the proposed1128 division cases; neither765 nor767 results transfer. Half AD, alpha/rounding modes, out/inplace, full layouts, architectures and devices remain open.
5. Admit each final head only after all required checks and complete original artifact readback, verify its remote tree and merge with an expected-head lease. Continue the native implementation order below. Retire old paths only after real consumers have been migrated and verified.

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
