# Whole-PyTorch HIA rewrite: execution and acceptance plan

## Objective and source of truth

Complete the entire pinned PyTorch v2.15.0 project in native HIA, preserve upstream behavior, then retire replaced implementations and redundant process versions. The upstream commit is `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`. Keep **2,590 native declarations, 688 derivative declarations and 17 full subsystem contracts** in the target.

Resume from current Git heads/open PRs, [`full-acceptance.json`](../compatibility/full-acceptance.json), the capability inventory, [current audit](current-audit.md) and actual source-bound reports. Newer checked-in results take precedence over chat snapshots. Preserve concurrent work and publish with expected-head leases.

The canonical application is `tools/current_runtime/source/`; its manifest is authoritative. The current candidate has 765 core functions / 15 units and requires **24 groups / 3,845 finite cases**, including separately executed Generator probes. Record the actual complete result, never the sum of unrelated partial successes.

## Complete active integration

1. Complete the current765 **24 groups / 3,845 cases** on one actual cold-built image. Keep all previous23 inputs and controls, with the explicitly verified four add/sub successors in multiplication312. The original Half47 recipe still executes all239 requests:173 refusals,22 multiplication,2 negation,40 single-command add/sub and2 original FE additions. The new1006 reference and every original frame/owner/provenance/EOF predicate are mandatory.
2. Keep the current765 finite results distinct: addition/subtraction1006 and physical replay passed31,468 frames/119,842 assertions; original Half47 passed3,513/17,713; multiplication312 with the same actual1006 reference passed9,664/37,168. These do not form a combined24 PASS. The fixed761 PR47 correction f762a26 independently requires its complete23 gate and original CI artifact review before merge.
3. Preserve the observed failures. The old fourteen-unit live32 admission rejected the new fifteen-unit cold build before native execution; seven exact inventory literals now match the complete build without removing predicates. Original Half47 observer failures revealed two omitted FE additions and the two ordinary HIA rounding functions in rng03. The final owner and call-graph checks preserve those cases, all complete snapshots and unknown-callee rejection. The earlier missing-NumPy CI failure remains recorded as well.
4. Integrate the separately implemented native769 true-division candidate only after verifying its own source/image identity and complete original47 division successors. Its new1128 and affected312/1006 actual runs have separate identities. Require a new shared-reference **25 groups / 4,973 cases** execution; never import a765 PASS. Half AD, arbitrary alpha/rounding modes, out/inplace variants, full layouts, other CPU profiles and devices remain open requirements.
5. Admit each final head only after all required checks and full original artifact readback pass, then verify its remote tree and merge with an expected head. Preserve superseded attempts and source identities. Continue the full implementation order below after these finite increments.

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
