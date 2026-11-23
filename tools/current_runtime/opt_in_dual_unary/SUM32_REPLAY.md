# Historical Float32 Sum32 on the exact dual opt-in image

One physical current800/15 build passed all 32 original cases and 626 original comparison events: 64,415 native frames, 460,075 assertions, 293,342 exact words, and both original EOF observations in all 32 native sessions. The original isolated supervised worker exited 0 with both EOF observations and empty stderr. No second cold-build replay, current PyTorch execution, production activation, or adjacent suite qualification is claimed.

`sum32-evidence.json` indexes this run, including the actual candidate, original comparison report, supervisor exit and streams, native per-case packet hashes, and lossless archive hashes. The independent reader verified the original canonical `qualify_sum32` function, actual supervised streams, every paired packet and case/transport summary, and the request opcode domain. Owner blobs and owner event logs were compacted only after case closure and checked byte-for-byte with their modes before removing the raw duplicates. The named independent reader does not re-read the complete owner archive.

## Verify retained raw evidence
Use the fixed canonical source checkout, the actual run directory, and the physical build used for the run:
```sh
python3 -I -B tools/current_runtime/opt_in_dual_unary/verify_sum32.py \
  "$canonical_repository" "$run_directory/actual" "$physical_build"
```
The verifier supports original case packet streams or verified per-case lossless archives. It takes no action on the runtime and runs no worker. The original worker opened the unchanged fixed 39,045,926-byte reference, and the independent reader verified the report bound that exact reference (SHA-256 `c14e4b18f657429ad5dac5498c3abcae303b6a7e7550db7d15a3c9179bdcd728`).

`replay_sum32_original_capture.py` preserves the exact scope code invoked during the run. To reproduce its original worker, invoke it with `python3 -I -S -B`, providing `--repository`, `--dual-build`, `--dual-source`, `--dual-scoped`, `--shared` (canonical extracted original sum inputs), `--frozen` (canonical extracted frozen700 fixture), `--integral-intent` (the canonical pinned Integral24 intent), and `--output` (a new directory). The file requires the adjacent Resource7 proof scripts and uses the same exact canonical inverse. Set aside sufficient disk for the original raw capture. The archived original run's outer convenience receipt reported an error because its first optional postprocessor assumed raw case paths after those case payloads had been compacted. A separate watcher also expected a timeout field absent from a normal success receipt. Neither error came from the original worker; the independent command above was executed separately and verified the completed evidence, including actual timing, exit, both EOFs and raw packet bytes.
