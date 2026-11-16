# Retained-446 native integration coordinator

`tests/qualify_retained_gradients.py` is integration orchestration. It preserves the immutable producer, reference reports and independent test expectations. It does not implement numerical operations, import Torch/NumPy, or claim a new independent semantic review.

The coordinator's **first actual execution passed** using the explicitly requested producer-receipt reuse mode. It verified the root's fresh 17-step cold-build receipt and its source/image/nine module pairs, copied that complete producer directory byte for byte into the new output, then executed all six additional outer runners. No new native or integration failure occurred. This result does not claim that this invocation executed the default cold-build branch; the default branch is intended for real CI execution.

## Fixed identities

| Item | SHA-256 |
| --- | --- |
| Executed coordinator | `1376074fb29bc9721018eb9fd05f90dc041ba135688e789f3ca330dab3860f51` |
| Retained HIA source | `666230f05cc6942470006d1f184135321d8cbc26a6c6ba0bf7e3cf9857fe515b` |
| Retained ELF | `7568c0a4ec40d82d3a0b46d1e8a27499dd96e8fe3fd15433523131a196da89b5` |
| Parent 428 HIA source | `1fed8ea7ba5bf755a9df9ca1e61d34cfd4712cc2fd66e0728493091c81c24dae` |
| Parent 428 ELF | `6773485896d961b6c2e62784ebb66320290df47128bd1331aecc0c8b7f6611fe` |
| Original final aggregate bytes | `e9819c1463be722ae7c1c0e022d9338a7e86e10484cb12cc5e73f69122bdba77` |
| Complete raw evidence archive | `2d6d4a19d76d421e97ca5e09b2d30dad37dff167a072162d2a77e8e34aa5414c` |

The original final aggregate is 16,503,319 bytes. This package stores it as `validation/retained-integration-review/aggregate-report.json.xz`, with exact decoding verified by `aggregate-report-custody.json`. No JSON was reserialized to compress or transfer it. The raw aggregate remains available in the original execution output; its compressed persistent copy is sufficient for recovery.

## Interface

Default CI invocation, which really performs the frozen cold rebuild:

```sh
python3 -I -S -B tests/qualify_retained_gradients.py \
  --repo "$PWD" \
  --parent-build "$PWD/build/gradient-adam-distributions" \
  --compiler "$HEPTA_STACK_BUILD/native/stack-module-compiler" \
  --linker "$HEPTA_STACK_BUILD/native/stack-linker-session" \
  --stack-build "$HEPTA_STACK_BUILD" \
  --output "$PWD/build/retained-integrated-new"
```

`--repo` defaults to the script's repository root. `--parent-package` is optional and defaults to that same repository root. If supplied, it must be the directory containing `tools/gradient_adam_distributions/` and `tests/`, not the `tools/gradient_adam_distributions` subdirectory itself.

Output must be new. Both branches produce `OUTPUT/producer/native/` and `OUTPUT/report.json`. Default mode invokes the frozen `tools/retained_gradients/qualify.py` with the actual compiler/linker, then validates the resulting complete 17-step receipt and all native files.

To reuse a specific complete, already executed producer directory, add:

```sh
  --producer-review /absolute/path/to/completed/producer
```

This branch requires the original report, every child report and stream, and the complete native directory. It checks the 17 ordered commands, exact runner/reference identities, status/classification counters, original child byte hashes, native source/ELF/unit hashes and all package inputs. It inventories the entire producer directory, copies every file with its original mode to `OUTPUT/producer`, and rechecks both source and copy. It does not rewrite child JSON absolute paths or argv. The aggregate explicitly records the old/current path mapping, `producer_receipt_reused=true`, and `producer_executed_this_invocation=false`.

## Checks and actual outcomes

The coordinator verifies the full frozen 79-file producer handoff, 22-file actual-retain review, 32-file resource review and 12-file extension package, plus the producer's 13 source inputs and the parent package's 45 source inputs. Every declared file size, SHA-256 and mode is checked. Two references retain all eleven original provenance checks, and all inherited references remain individually pinned.

The required document-only import relocations are explicit: the nonleaf README goes under `validation/retained-nonleaf-review/`; the resource README and `handoff.json` go under `validation/retained-resource-review/`; and the extension README and `handoff.json` go under `validation/retained-extension-review/`. The frozen manifest bytes remain unchanged.

The completed invocation checked **3,850 files before and after**, with the reused producer's **1,577 files** preserved as a self-contained copy. It then actually ran:

| Runner | Scope / outcome |
| --- | --- |
| Retained actual reference | Eight families, 12 variants; all qualified in the declared scope; original tolerance, exact metadata and EOF checks unchanged |
| Original optimizer case 14 additive | Two fixed original subobservations; both matched; original 24-case replay/report unchanged |
| Resource controls | Eight passed, including transient reservations, rollback and ownership |
| Real typed handoff | Two directions; 19 actually compiled/linked modules; ten complete request/response frames; old/new state rejection and recovery passed |
| Source review | All 446 bodies in exactly one actual unit; 410 parent bodies unchanged; read-only review only |
| Frozen extensions | Mathematical two, actual mixed two, Scalar-add 48 variants, six owner cross contracts, and eight extreme-stride cases all passed; 947 native frames and 344 strict extreme fields |

These counts overlap and must not be summed into unique API coverage. The inherited 17-step producer retains every original PARTIAL/OPEN classification, the original 93/94 negative-contract failure, the 18 unresolved distribution cases, four ordinary atomic side-effect differences, and old optimizer/state lifecycle observations. The original optimizer mapping remains 18 qualified + one partial + five open; the existing additive union remains 23 + partial case 14. The new case-14 and retained-12 evidence is separately recorded, without rewriting those historical receipts.

The old retained-resource archive interruption remains explicitly incomplete, with its original reports and missing tail disclosed. The new typed-handoff execution has separately verified complete raw request and response bytes. It does not reconstruct or replace the missing original history.

## Complete evidence and self-reference rule

The coordinator writes:

- `OUTPUT/raw-evidence.tar.xz`: every producer/native/child/raw/stream/adapter source byte.
- `OUTPUT/evidence-manifest.json`: exact member path, size, SHA-256 and mode, with archive verification results.
- `OUTPUT/report.json`: final aggregate binding both files after their verification.

Only the three exact top-level paths `report.json`, `evidence-manifest.json`, and `raw-evidence.tar.xz` are excluded from the archive. Child reports with the same basename are included. The final aggregate is not a RUNNING snapshot and is not included recursively. Every child absolute path and argv remains unchanged.

This invocation produced **2,051 members / 127,240,106 uncompressed file bytes**, compressed to **2,360,948 bytes**. Every member was read back and compared for size/hash/mode; the complete XZ stream/trailer was drained and verified; expanded originals were rechecked. The package carries this archive and manifest unchanged. It also carries the original outer launch command, stdout, stderr and exit code.

The main aggregate's default cold branch was not executed by this local invocation. The root had already run the frozen producer cold; this invocation intentionally reused that receipt. A future CI default run must preserve the real `producer_executed_this_invocation=true` result and its own raw archive instead of copying this result as if new.

Whole-PyTorch status remains **HOLD**, the denominator remains **2,590**, `full_schema_parity` remains **0**, and `full_rewrite_complete` remains false. Live32/frozen35 are separate root/hosted gates and are deliberately not included in this native aggregate. Bool/native-dropout/dtype work and the remaining full-project boundaries stay open.
