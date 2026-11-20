# Whole-PyTorch HIA rewrite: execution and acceptance plan

## Objective and source of truth

Complete the entire pinned PyTorch v2.15.0 project in native HIA, preserve upstream behavior, then retire replaced implementations and redundant process versions. The upstream commit is `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`. Keep **2,590 native declarations, 688 derivative declarations and 17 full subsystem contracts** in the target.

Resume from current Git heads/open PRs, [`full-acceptance.json`](../compatibility/full-acceptance.json), the capability inventory, [current audit](current-audit.md) and actual source-bound reports. Newer checked-in results take precedence over chat snapshots. Preserve concurrent work and publish with expected-head leases.

The canonical application is `tools/current_runtime/source/`; its manifest is authoritative. The current candidate has 761 core functions / 15 units and requires **23 groups / 2,839 finite cases**, including separately executed Generator probes. Record the actual complete result, never the sum of unrelated partial successes.

## Complete active integration

1. Cold-build and run all 23 groups / 2,839 cases. Preserve all original 21 groups / 2,156 case inputs and additional controls. Exactly 22 of the original 239 Half entry requests now require successful multiplication, and two require successful negation, with actual fixed-Torch results; the other 215 refusals remain unchanged. The new 312-case multiplication and 371-case negation references and complete physical replay are mandatory.
2. Preserve the original failed 740 CI: 19 groups completed, while missing NumPy prevented numeric-predicate admission before native execution. The repair installs a SHA-pinned wheel, retains the actual pip report and keeps strict stderr checks. Use CPython 3.12 throughout.
3. Keep Half multiplication semantics explicit: 20 public ordered dtype pairs, rank-sensitive promotion, original scalar-RHS opmath, fixed AVX2/single-thread NaN payload selection, and FBGEMM conversion tails. Half negation separately preserves the vector/scalar NaN distinction and has two actual allocation-failure/recovery controls. Half AD, add/subtract/divide and direct Half/F64 views remain pending in this candidate. Keep BF16 semantics explicit: scalar/vector NaN conversion depends on paired input/output layout in the fixed AVX2/AVX512 kernel. DEFAULT and other architectures remain full-dtype/platform work. Original finite rounding and RNG support stay intact.
4. Admit each final head only after required real checks and original evidence pass; read back its remote tree. Close superseded PRs as superseded, preserving their original identities.

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
