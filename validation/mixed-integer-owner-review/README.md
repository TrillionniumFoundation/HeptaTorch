# Mixed I64 autograd: independent owner and resource checks, candidate 10

These checks bind runtime source `5f8ab1f7ca1593dc21b8b6a3537e210cfbc42b37ddefcb322239c5a316d88c49` and native image `69045063c00395d07fc6c0101ed1b8aba18376cb1daea54e0a01c2c5caeb49b6`. The runtime contains 556 core functions, 11 compiled units, a 640-byte descriptor, and the distinct `HTMXAD02` / `HTOMXD02` state format. The preceding I64-07 profile and its tests remain unchanged.

## Observed results and limits

| Independent group | Actual result |
|---|---|
| Full owner accounting | 4 cases; 58 live frames; 144 complete 80-word snapshots; 53 isolated transitions; 105 assertions; 3,495,120 bytes of ordinary error rollback |
| Copied new-role negative controls | 17 exact refusals; host checker only; no target memory modification |
| Existing resource controls, with explicit new AD admission | 6 cases; 536 live frames; 162 assertions, including actual `RLIMIT_AS` allocation failure and recovery |
| Existing owned-entry capacity controls | 5 cases; 46 actual probe processes; 33 live frames; 12 real compiled and linked units |
| New Saved getter capacity and error controls | 6 cases; 14 actual probe processes; 17 live frames; exact reuse of the preceding checked probe image |
| Real typed owned handoff | 2 directions; 23 actual compiled and linked units; 14 live frames; 1,398,048 bytes of complete rollback |
| Source and unit custody | All 556 core function bodies matched exact compiled unit source; 531 parent bodies unchanged, 13 changed, 12 added |

The owner checker covers the actual two-slot, 32-byte private Hold containing original Tensor and incarnation pairs. It recomputes the corresponding `600` strong Tensor and `208` graph-reference sums, including repeated `x*x` inputs. It preserves the parent storage, Version, pool partition, Generation, weak, private Hold and captured Accumulator cycle checks. A live I64 view must retain its real ultimate base. A rowless I64 metadata descriptor may still have `208` references, but must have no remaining real base edge.

The native controls demonstrate the Saved I64 view's final public release, both original input slots, `zero_grad(False)` cleanup, Saved getter absence/spent/Version errors, and history transfer. The new getter returns a 32-byte result record with no payload, so a one-command successful reply requires 64 bytes. A short output must preserve the complete owner and all 5,056 RNG state bytes. A later ordinary command error must undo any earlier publication in that frame.

Both handoff directions pass the real owned buffer through a checked HIA controller. The receiving incompatible profile rejects it, the correct profile can resume it, and the matching native close unmaps it while the process is still alive. The newer direction includes a genuine Saved-original-T private Hold and a live ultimate-base reference. No owner descriptor is copied and no native pointer is supplied by the request.

These are bounded checks of this producer's real states. They are not certification of arbitrary foreign owner memory or arbitrary machine code. The copied negative controls are not new API cases. No Torch process was run by these native checks. The fixed source-bound 54-case and 36-case references are retained as evidence where used. New candidate-10 live32/full35 execution is separate and is not claimed by this packet. The whole PyTorch gate remains **HOLD**, with denominator **2590**, closed schema count **0**, and full completion **false**.

## Preserved first failures

The original six-resource runner was first run on candidate09 unchanged. Its old `103` requires-grad refusal expected error 5 but observed successful new AD admission; the other five cases passed. The explicit successor changes only that admission into a real differentiable native add/backward control, keeps the original refusal failure, and still requires errors for mixed-dtype `.grad`, nonfloating `requires_grad`, and in-place requires-grad mutation.

The general owner runner first lacked the already-existing command40 read's zero-allocation explanation, then used opcode9 (SGD) where the intended zero-grad command is opcode8. Both original runs, scripts and exact corrections are preserved. No expected numeric result or producer source was changed.

The source auditor first imposed an invalid restriction that same-named private helpers in separate modules have identical SSA bodies. The final source audit checks every complete core body against an actual unit and requires at most one public provider for each name. The first refusal and its source are retained. This was a source-review error, not a failed native execution.

Candidate09's independent Saved-view lifetime failure is not reclassified. Candidate10's new rowless-view invariant and the native release case directly test its correction. No old execution is relabelled as candidate10 execution.

## Portable execution

Run from a repository containing this packet and the original pinned protocol/client. Set `mixed_build` to a candidate10 native build, `integer_parent` to the immutable candidate07 native build, and `mixed_stack` to the complete compiler bundle containing `native/stack-module-compiler`, `native/stack-linker-session`, and `native/stack-linker-runtime`. Every output directory must be new. The scripts accept absolute paths; none depends on the historical scratch location.

```sh
mixed_tests=tests/mixed_integer_owner_review
mixed_wire=tests/retained_resource_review/protocol.py
mixed_ref54=validation/integer-tensors/reference/torch-int64-tensor-reference.json
mixed_ref36=/absolute/path/to/torch-mixed-integer-autograd.json
mixed_out=/absolute/path/to/new-review

python3 -S -B "$mixed_tests/audit_source10.py" --build "$mixed_build" --previous-build "$integer_parent" --protocol "$mixed_wire" --output "$mixed_out/source"
python3 -S -B "$mixed_tests/test_saved_owner10.py" --build "$mixed_build" --protocol "$mixed_wire" --output "$mixed_out/owner"
python3 -S -B "$mixed_tests/test_integer_resources.py" --build "$mixed_build" --protocol "$mixed_wire" --reference "$mixed_ref54" --mixed-reference "$mixed_ref36" --expected-source-sha256 5f8ab1f7ca1593dc21b8b6a3537e210cfbc42b37ddefcb322239c5a316d88c49 --expected-native-sha256 69045063c00395d07fc6c0101ed1b8aba18376cb1daea54e0a01c2c5caeb49b6 --output "$mixed_out/resources"
python3 -S -B "$mixed_tests/test_integer_owned_entry.py" --build "$mixed_build" --stack-build "$mixed_stack" --probe-source "$mixed_tests/entry_probe.hepta" --protocol "$mixed_wire" --reference "$mixed_ref54" --expected-source-sha256 5f8ab1f7ca1593dc21b8b6a3537e210cfbc42b37ddefcb322239c5a316d88c49 --expected-native-sha256 69045063c00395d07fc6c0101ed1b8aba18376cb1daea54e0a01c2c5caeb49b6 --output "$mixed_out/entry"
python3 -S -B "$mixed_tests/test_saved_entry10.py" --build "$mixed_build" --probe-build "$mixed_out/entry" --protocol "$mixed_wire" --output "$mixed_out/Saved-entry"
python3 -S -B "$mixed_tests/test_mixed_typed_handoff.py" --build "$mixed_build" --previous-build "$integer_parent" --stack-build "$mixed_stack" --protocol "$mixed_wire" --controller "$mixed_tests/state_handoff.hepta" --output "$mixed_out/handoff"
```

The getter-entry script explicitly records that it reuses the preceding real probe compilation. It binds the exact HIA probe source, all 11 application objects, the complete native link request, compile/link statuses and the final probe image before it executes new capacity checks.

## Raw evidence

All original child JSON, stdout/stderr, request/response bytes, actual owner observations, compiler inputs and outputs are retained in three archives. Their manifests give each original relative path, size, mode and SHA-256. Original child absolute paths and argv were not rewritten. The archives also retain the original test-side and source-side failures. The restore indexes explain which verified expanded duplicates were reclaimed after full byte checks.

This read-only command verifies packet files, all archive members, exact sizes/modes/hashes, and single-XZ-stream EOF. It performs no native or Torch execution:

```sh
python3 -I -S -B validation/mixed-integer-owner-review/verify_evidence.py --repo /absolute/path/to/repository
```
