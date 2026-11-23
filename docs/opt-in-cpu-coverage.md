# Bounded opt-in CPU coverage

This crosswalk tracks published evidence for six functional operator routes, without qualifying a full PyTorch declaration or changing which runtime is default.

Canonical [e944490](https://github.com/TrillionniumFoundation/HeptaTorch/tree/e9444902df9878060ab389a70ee957c1b7f4a887) is typed-ReLU775. Its [manifest](https://github.com/TrillionniumFoundation/HeptaTorch/tree/e9444902df9878060ab389a70ee957c1b7f4a887/tools/current_runtime/source/manifest.json) is separate from all rows below. The tracked finite ledger requires 29 groups / 13,851 cases. The [exact hosted run](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/44912529511) reports success, but this crosswalk did not independently admit the inner original reports.

Published-format outer receipts from artifact `14713706649`: qualification `a92cf94efaa4745168d4ab82239b1af60b4819942e6f89c82a794777004a9902`, byte/mode custody `0ea8a3a22f7d4c2541590f18f98b3631d51603c70aa79ef8330714bc676172a9`. They report 29 groups / 13,851 cases including Generator131; the custody reader records zero independent case admissions. Canonical execution evidence remains null in this crosswalk.

Bounded inner receipt readback (no new case admissions): `current-runtime/build/build.json` `c1b4cb1c483705d935b453f3b9ab1f82b20b59ea658048bc5750cc33bf9b0d33` (PASS_NATIVE_BUILD_ONLY; runtime_qualified=false; source and manifest hashes match); `current-runtime/build/manifest.json` `348ed8e949256f985cdca0d9c612dc646ae738beee6013c27ab405d2df3e67c8` (exact source manifest hash match); `current-runtime/review/generator131/report.json` `5ce818551600d18b45ab1af1e1c8b625844c251020238603bc9d3e24cf6e0c36` (hash only; contents and cases not independently admitted).

[Full acceptance](https://github.com/TrillionniumFoundation/HeptaTorch/tree/e9444902df9878060ab389a70ee957c1b7f4a887/compatibility/full-acceptance.json) stays **HOLD**: 0/2,590 complete native declarations, 0/688 derivative declarations and 0/17 complete subsystem contracts.

| Route | Frozen declaration | Evidence PRs | Out-of-scope overloads or derivative |
| --- | --- | --- | --- |
| 129 | `where.self(Tensor condition, Tensor self, Tensor other) -> Tensor` | [#66](https://github.com/TrillionniumFoundation/HeptaTorch/pull/66), [#71](https://github.com/TrillionniumFoundation/HeptaTorch/pull/71) | `where.self_out`, `where.ScalarSelf`, `where.ScalarOther`, `where.Scalar`, `where`; frozen derivative unqualified |
| 130 | `masked_select(Tensor self, Tensor mask) -> Tensor` | [#67](https://github.com/TrillionniumFoundation/HeptaTorch/pull/67), [#70](https://github.com/TrillionniumFoundation/HeptaTorch/pull/70), [#71](https://github.com/TrillionniumFoundation/HeptaTorch/pull/71) | `masked_select.out`, `masked_select_backward`; frozen derivative unqualified |
| 131 | `logical_and(Tensor self, Tensor other) -> Tensor` | [#68](https://github.com/TrillionniumFoundation/HeptaTorch/pull/68), [#70](https://github.com/TrillionniumFoundation/HeptaTorch/pull/70), [#71](https://github.com/TrillionniumFoundation/HeptaTorch/pull/71) | `logical_and_`, `logical_and.out` |
| 133 | `logical_or(Tensor self, Tensor other) -> Tensor` | [#68](https://github.com/TrillionniumFoundation/HeptaTorch/pull/68), [#70](https://github.com/TrillionniumFoundation/HeptaTorch/pull/70), [#71](https://github.com/TrillionniumFoundation/HeptaTorch/pull/71) | `logical_or_`, `logical_or.out` |
| 134 | `logical_xor(Tensor self, Tensor other) -> Tensor` | [#68](https://github.com/TrillionniumFoundation/HeptaTorch/pull/68), [#70](https://github.com/TrillionniumFoundation/HeptaTorch/pull/70), [#71](https://github.com/TrillionniumFoundation/HeptaTorch/pull/71) | `logical_xor_`, `logical_xor.out` |
| 135 | `nonzero(Tensor self) -> Tensor` | [#69](https://github.com/TrillionniumFoundation/HeptaTorch/pull/69), [#70](https://github.com/TrillionniumFoundation/HeptaTorch/pull/70), [#71](https://github.com/TrillionniumFoundation/HeptaTorch/pull/71) | `nonzero.out`, `nonzero_static.out`, `nonzero_static`, `nonzero_numpy`; frozen derivative unqualified |

## Exact evidence slices

Counts describe the named latest slice, not additive totals or the entire operator contract.

### #66 (7583ea5): standalone current800, stacked on #65

Ordinary owned canonical no-grad Bool condition, rank ≤4 and bounded shape; six public same-dtype values with compatible broadcasting. Mixed pairs and true rank-zero Tensor paths are enumerated by release-specific dtype/rank/CPU profile; this is not an unrestricted promotion claim. FBGEMM F16 cases keep their original profile and size branch.

**Measured:** 264 cases, 5,004 values; latest true rank-zero F64 to dimensioned F32 slice, either operand order. [Manifest](https://github.com/TrillionniumFoundation/HeptaTorch/blob/7583ea54756f6c76f722b95b2e569f417d584fc4/tools/current_runtime/opt_in_where_zero_f64_narrow/manifest.json) · [Receipt](https://github.com/TrillionniumFoundation/HeptaTorch/blob/7583ea54756f6c76f722b95b2e569f417d584fc4/tools/current_runtime/opt_in_where_zero_f64_narrow/evidence.json).

**Control boundary:** Exact predecessor status7 on those 264 fixtures; 43,339 inherited frames and 2,814 rollback checks in the cited slice. Neither is a declaration closure.

### #67 (270fde9): standalone main775 overlay

Ordinary owned contiguous CPU roots, six public value dtypes, Bool mask, rank1–4 value and ≤4 joined rank. Same-shape, one-way and two-sided compatible broadcast; input and joined lattice limits retained (262,144). Fresh rank1 output, raw value bits and empty output preserved.

**Measured:** 61 cases, 4,554 values; latest two-sided broadcasting slice. [Manifest](https://github.com/TrillionniumFoundation/HeptaTorch/blob/270fde9f8b597c1444fd711026121fef4845b36d/tools/current_runtime/opt_in_masked_select_symmetric/pinned-manifest.json) · [Receipt](https://github.com/TrillionniumFoundation/HeptaTorch/blob/270fde9f8b597c1444fd711026121fef4845b36d/tools/current_runtime/opt_in_masked_select_symmetric/pure.json).

**Control boundary:** 12 measured over-joined-cap status3 cases. Two attempted view/rank setup families do not imply actual native operator rejections when no public root was produced.

### #68 (dc4c9fd): standalone main775 overlay

Same-dtype six public types, rank0–4, ordinary canonical no-grad CPU roots, nonempty right-aligned broadcast up to 262,144 joined elements, fresh Bool; op132 deliberately not added.

**Measured:** 405 cases, 8,310 values; latest ordinary broadcast slice. [Manifest](https://github.com/TrillionniumFoundation/HeptaTorch/blob/dc4c9fd5efb9526241ea21ac83a4fa5b9196570f/tools/current_runtime/opt_in_logical_broadcast/manifest.json) · [Receipt](https://github.com/TrillionniumFoundation/HeptaTorch/blob/dc4c9fd5efb9526241ea21ac83a4fa5b9196570f/tools/current_runtime/opt_in_logical_broadcast/pure.json).

**Control boundary:** 7,765 prior wire exchanges and 567 full rollback checks, nine specifically verified archival scope changes. Only this exact published same-dtype head is included; mixed-dtype candidate held.

### #69 (b47f38e): standalone main775 overlay

Ranks 1–4 across six public input dtypes, for the bounded opt-in owned contiguous no-grad CPU path with zero offset. Output is fresh I64 [K,rank]; rank-four physical column stride is [1,K]. This latest numeric measurement uses small ordinary inputs (N≤360); it does not extend the prior Bool-only capacity evidence to the added dtypes.

**Measured:** 68 cases, 5,776 values; latest small rank-four numeric slice (65 newly scoped and three Bool controls); I64 coordinate words counted per order with all four physical columns checked. [Manifest](https://github.com/TrillionniumFoundation/HeptaTorch/blob/b47f38e09db92d425d08f865b67278ce5d3b9964/tools/current_runtime/opt_in_nonzero_r4_dtypes/pinned-manifest.json) · [Receipt](https://github.com/TrillionniumFoundation/HeptaTorch/blob/b47f38e09db92d425d08f865b67278ce5d3b9964/tools/current_runtime/opt_in_nonzero_r4_dtypes/pure.json).

**Control boundary:** 3,660 prior successful ordinary exchanges remain byte-exact. Actual earlier image kept the three Bool positives and cleanly declined all 65 ordinary numeric cases without owner mutation. No new numeric capacity or broader qualification.

### #70 (b1e1fd2): separate main775 coexistence

39 exact operator bodies; #67/#68 exact and earlier #69 2b5b2fc: numeric rank1 and Bool rank2 only. Does not contain #66 where or later #69 rank-two numeric or rank-three/rank-four scope.

**Measured:** 36 chains, 129 combined outputs, 129 isolated outputs, 925 values; ordinary operator coexistence matrix only. [Manifest](https://github.com/TrillionniumFoundation/HeptaTorch/blob/b1e1fd20de94fa12068ce1e33da2bc7c6f1f8026/tools/current_runtime/opt_in_cpu_operator_coexistence/pinned-manifest.json) · [Receipt](https://github.com/TrillionniumFoundation/HeptaTorch/blob/b1e1fd20de94fa12068ce1e33da2bc7c6f1f8026/tools/current_runtime/opt_in_cpu_operator_coexistence/ordinary-readback.json).

**Control boundary:** Existing published ordinary conformance, not pressure/fault/unsupported coverage.

### #71 (cf23e92): separate current800 coexistence

Exact #66 plus #70 additive pins and unchanged current800 owner liveness. #70’s earlier #69 scope retained; no later #69 rank-two numeric or rank-three/rank-four scope. Adds 14 where-to-selection/nonzero chains to #70’s 36 ordinary chains.

**Measured:** 50 chains, 171 combined outputs, 171 isolated outputs, 1,362 values; ordinary operator coexistence; offline verifier only rereads its sealed raw archive. [Manifest](https://github.com/TrillionniumFoundation/HeptaTorch/blob/cf23e9250234c788eb58b1a9de48d9c38fa655f8/tools/current_runtime/opt_in_current800_operator_coexistence/pinned-manifest.json) · [Receipt](https://github.com/TrillionniumFoundation/HeptaTorch/blob/cf23e9250234c788eb58b1a9de48d9c38fa655f8/tools/current_runtime/opt_in_current800_operator_coexistence/ordinary-readback.json).

**Control boundary:** Ordinary only; 768 prior live owner checks. Offline verifier does not rerun native/Torch, does not measure unsupported paths and does not add coverage.

## Separate #71 CPU application evidence

These are separate exact-image experiments, not additional operator-coexistence counts.

- **Two-layer F32 inference:** 3 cases, 15 stages, 80 exact words. One owned CPU no-grad session per fixture: matmul, bias, ReLU, matmul, bias. A fourth signed-zero-mismatch fixture was retained as Torch-only, not run in HIA. [Receipt](https://github.com/TrillionniumFoundation/HeptaTorch/blob/e7e02a54618bdd111982e447189f810d9164731c/tools/current_runtime/opt_in_current800_two_layer/pure-v2.json).
- **One owned scalar-loss SGD step:** 26 exact words, loss 8 → 2.25, weight Version +1. One owned F32 scalar-loss toy, with one backward and SGD at 1/8, same weight Tensor/Storage/Version owners, fresh recomputed loss and unchanged input/target. [Receipt](https://github.com/TrillionniumFoundation/HeptaTorch/blob/72be4108b176fa92873a726c50d2ddf5aea6ba46/tools/current_runtime/opt_in_current800_one_step_sgd/pure.json).
- **Manual demo:** Standalone, manual CPU caller for the three inference fixtures and scalar-loss SGD only; requires the exact prebuilt ELF and does not include the two-layer training fixture. [Instructions](https://github.com/TrillionniumFoundation/HeptaTorch/blob/ead53ace4f56f4b27054a61e6cf308774a4806dd/tools/current_runtime/opt_in_current800_fixed_demo/README.md) · [Live receipt](https://github.com/TrillionniumFoundation/HeptaTorch/blob/ead53ace4f56f4b27054a61e6cf308774a4806dd/tools/current_runtime/opt_in_current800_fixed_demo/live/receipt.json).
- **One two-layer F32 SGD fixture:** 54 exact words, 265 frames, loss 8 → 12625/4096. One backward, one SGD per leaf at 1/32, and fresh recomputed loss on one owned two-layer F32 fixture; both weight Version counters increment once. [Receipt](https://github.com/TrillionniumFoundation/HeptaTorch/blob/cf23e9250234c788eb58b1a9de48d9c38fa655f8/tools/current_runtime/opt_in_current800_two_layer_train/pure.json).

No application evidence here qualifies any full forward or gradient contract, default path, speed advantage or canonical acceptance.

## Pins and exclusions

#70 and #71 pin earlier #69 `2b5b2fc`; newer #69 rank-two numeric, rank-three, and rank-four evidence does not roll forward into them.
Held mixed logical and held Cat work are not credited. No crosswalk entry affects default routing, existing hosted CI, independent registered verifiers, or the complete-project gate.
