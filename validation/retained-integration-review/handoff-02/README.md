# Retained coordinator: preserve failed-execution evidence

This additive revision changes only the final exception-handler body in the integration coordinator. The existing successful execution and all frozen native/oracle test packages remain unchanged.

Current coordinator SHA-256:

`c71d40ea111b9c80e9d0987f7cd42729cc3f47ecf5a86b7e5d4ecf0659ec0ee2`

Original successful coordinator SHA-256:

`1376074fb29bc9721018eb9fd05f90dc041ba135688e789f3ca330dab3860f51`

## What changed

On failure, the coordinator first preserves its primary error and traceback and records `status=FAIL`. It updates the producer execution/reuse flags from the completed commands and verified copy instead of treating the requested mode as proof that execution happened. It then attempts to archive every available output byte under the same non-recursive archive rule as the success path.

Successful preservation is explicitly marked `PARTIAL_FAILED_EXECUTION_BYTES_ONLY`, with `successful_suite_completion_claimed=false`. If archiving itself fails, that secondary error and traceback are recorded separately as `evidence_archive_error` / `evidence_archive_traceback`. They never replace the primary error. A failure to write the initial or final report is similarly reported separately. The process still returns 1, and no failure branch changes a test result to PASS.

The AST comparison in `validation/retained-integration-review/failure-custody-02/source-delta.json` replaces only the bodies of the old/new final `main` exception handlers with `pass` and compares the complete remaining module ASTs. They are identical. The original and revised source bytes and a unified diff are included.

## Actual controlled failure

The control supplied a deliberately wrong, non-executed compiler-pin fixture and ran the real coordinator in its requested default fresh mode. All immutable source inputs were still checked normally. The compiler-pin check rejected the fixture before any producer, compiler, native runtime or Torch execution.

- The coordinator actually returned **1** and retained **FAIL**.
- It recorded **zero completed outer commands**, `producer_executed_this_invocation=false`, and `producer_receipt_reused=false`.
- The original `input SHA` error, fixture path and traceback remained intact.
- The partial archive contained the exact executed coordinator source. No child artifacts are invented because no child ran.
- The archive's sole member, length, SHA-256, mode, full XZ stream and trailer were independently verified. The final aggregate binds both the partial archive and manifest.
- The host control returned `PASS_EXPECTED_PREPARATION_REJECTION`. This is a test of the coordinator's failure handling; it is not a native semantic PASS or a failed native implementation.

The control report SHA-256 is `e7f2822209e420a28b7c563254c43c9f113b60fb65299aa939931c24cdd56e8b`. The coordinator's original FAIL report SHA-256 is `0930a22e85aee0d8903d08b1c91983dcbf7633d44df17f839b7cd93e9779ff70`. Both original JSON byte sequences, the fixture, command, stdout, stderr, source delta, source copies, archive and manifest are preserved.

The secondary `archive_outputs` failure path is guarded and source-reviewed; this control does not claim to have induced an archive failure.

## Prior success remains separate

The prior full reuse-mode aggregate remains exactly `e9819c1463be722ae7c1c0e022d9338a7e86e10484cb12cc5e73f69122bdba77`, produced by `137607…`. Its 2,051-member complete archive remains `2d6d4a19d76d421e97ca5e09b2d30dad37dff167a072162d2a77e8e34aa5414c`, with all 17 producer suites and six additional outer runners preserved.

The parent package is `retained-integration-extension/handoff-01`, manifest `6ea3a232cb266ef6cd4480ad5d2026b61ac19b43589f7337e92d13d03f642b75`. This revision does not relabel that run as having executed the new source, and the six successful suites were not repeated for this exception-only change. The original successful source also remains in this control's `original-qualifier.py` and in the prior complete archive.

The CLI and normal fresh/reuse workflows are unchanged. CI should run the revised source in normal fresh mode and retain its own new report and archive. The root Make/CI integration must copy/emit any available FAIL report and partial evidence even when the coordinator exits nonzero, then preserve that original nonzero exit status.

Whole-PyTorch status remains **HOLD**, `full_schema_parity=0`, and `full_rewrite_complete=false`. This coordinator control closes no additional PyTorch API row.
