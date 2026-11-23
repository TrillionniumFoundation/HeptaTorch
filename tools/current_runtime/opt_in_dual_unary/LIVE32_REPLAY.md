# Historical Live32 on the exact dual opt-in image

The complete frozen PR39 Live32 suite passed on one physical current800/15 #63 build with actual native execution and the pinned live PyTorch oracle. The 32 original trajectories produced 2,132 frames, 17,455 exact binary32 words, 1,137,600 serialized state bytes, four original partial-backward prefixes, one original uniform refusal, and 64 native EOF observations. Both original readbacks re-decoded and rechecked the 4,264 complete before/after owner snapshots using the original structural predicates. No previous pass was transferred and the production image remains unchanged.

The scoped replay preserves the original Live32 wrapper, immutable PR39 runner, and numerical and custody predicates. Exact profile rebinding uses the full current candidate admission and reversible joint-to-canonical source proof. Retained packet census did not enter either changed opcode path. The evidence index binds the archive, original physical build, outer and inner reports, and live process streams.

## Reproduce

Use the exact current repository checkout, adjacent Resource7 proof files, and immutable validation/current-runtime32-source package restored by tests/current_runtime_fixtures.py:restore_live32. Use the candidate build/source and exact scoped receipt, plus a CPython 3.12 venv matching the repository Torch oracle lock. Start with a fresh output directory with space for the original raw evidence.

```sh
python3 -I -S -B tools/current_runtime/opt_in_dual_unary/replay_live32.py \
  --repository "$canonical_repository" --dual-build "$physical_build" \
  --dual-source "$candidate_source" --dual-scoped "$candidate_scoped_receipt" \
  --source-package "$restored_pr39_package" --torch-python "$torch_venv/bin/python" \
  --output "$new_run_directory"
```

Keep the venv launcher name: resolving it to system Python would lose venv metadata. An earlier attempt correctly failed literal interpreter-path custody because the launch token contained a parent-directory segment while Python's actual observation did not. Its original worker passed but it was never counted. The successful retry changed only the invocation spelling and retained the original validation and unmodified process receipts.

Run the separate archive checker with the original output directory and the XZ container to compare every retained regular file byte and mode and verify directory inventory and modes. It runs no native worker and does not replace the original PR39 raw readback.
