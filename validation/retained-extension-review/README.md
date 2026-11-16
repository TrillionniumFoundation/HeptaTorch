# Retained 446: inherited extension regression replay

This package records a **new execution on the recovered, cold-built retained-446 native image** of the inherited 428 extension tests. All five regression groups passed. The package is an integration replay, not an independent implementation review, a new Torch installation/run, or an expansion of the PyTorch API coverage denominator.

The complete raw evidence is persistent in `validation/retained-extension-review/raw-evidence.tar.gz`. Its 297 members, 31,480,354 uncompressed file bytes, each original length/SHA-256/mode, and full gzip trailer were independently read back after compression. No raw member is missing. The original expanded copy remains available in the task's temporary execution directory, but is not needed to recover these bytes.

## Fixed native identity

| Input | SHA-256 / value |
| --- | --- |
| Full HIA source | `666230f05cc6942470006d1f184135321d8cbc26a6c6ba0bf7e3cf9857fe515b` |
| ELF | `7568c0a4ec40d82d3a0b46d1e8a27499dd96e8fe3fd15433523131a196da89b5` |
| Producer profile | `8da746dea3493b3d21714eae3d8048a38b073796d519395b96f9dd783ac80758` |
| Runtime | 446 core functions; 5,922 blocks; 21,904 instructions; nine modules |
| Internal state | `HTRET001` / `HTORET01`; 576-byte descriptors |
| Root cold-build receipt used | `51cfd96be00fd3c07de4fa45709f27c06ccd5e424240b6db55e0bb0cf213ceaf` |

This replay did not compile the image again. It read and verified the root's actual cold-built source, ELF, all nine module sources and objects against the frozen producer profile before execution, and rechecked every input after all suites. The raw archive includes those source/object/image bytes and that original build receipt.

## Actual results

| Group | Fresh native workload | Result |
| --- | --- | --- |
| Mathematical mixed-gradient counterexample | Two branch orders; 34 frames | Both exact logical gradients passed |
| Fixed actual mixed-gradient observations | Two cases; 106 frames; 542 exact assertions; 204 binary32 words; 30,336 Generator-state bytes | All recorded layouts, identity relations, Version counters, gradients and state bytes passed |
| Fixed actual Scalar-add NaN observations | 12 cases × four variants; 408 frames; 864 exact assertions; 1,848 raw words | Exact NaN sign/payload/quiet-bit and non-NaN checks passed |
| Shared-owner cross contracts | Six cases; 319 frames; 477 exact assertions; 138 raw words; 14 whole MT states; 94 committed-node audits | All passed, including 6,990,240 bytes of complete-owner rollback comparisons |
| Fixed actual extreme-stride observations | Eight cases; 80 frames; then a separate strict comparison | All 344 fields passed, zero differing fields |

Total fresh native workload: **947 frames**. Six subprocess commands were executed because the extreme-stride observation and comparison are separate commands. The comparator did not execute native code again. The two mathematical cases and the two actual mixed-gradient cases are overlapping qualifications with different evidence; they are not four distinct PyTorch API entries.

Every subprocess returned zero on this first execution. No target failure or new assertion/tolerance adjustment occurred. Original child JSON bytes, stdout, stderr, test source, raw requests/responses and recorded owner snapshots are in the archive; the top report embeds each child JSON and its original hash/length.

## Exact adapters

### Cross-contract state tags

The original frozen `test_joint_gradients_v2.py` has SHA-256 `c3bf33cbb5220469c6727b61bf612551202088c350551271833d15bd51bfd5fc`. The executed adapter has SHA-256 `cec86e7075779f2f397c86e56a137e4dd26d2041cc0c71da9ffa3da61cb46187`.

Only these two byte literals change, exactly once each:

- `HTGAD002` → `HTRET001`
- `HTOGAD02` → `HTORET01`

The inverse reproduces the original file byte for byte and its original SHA-256. The package retains both sources, the inverse, unified diff and machine proof. All numerical, identity, Version, cleanup, Generator and rollback assertions remain unchanged. The original unadapted 428-tag test was not executed on 446 here; this package does not fabricate or reclassify an unperformed failure.

### Extreme observer filename

The unmodified extreme observer accepts a build directory and looks for its historical filename `tensor-gradient-adam-distributions-session`. An isolated directory supplies a byte-identical copy of the retained ELF under that filename, alongside the exact retained source. This is a filename compatibility copy, not a compiler transformation, a second implementation, or a change to a production build directory. The observer and comparator source bytes remain frozen.

All other runners execute directly from the read-only repository paths with only their existing CLI native source/image pins pointing to 446. The protocol and owner client remain original bytes.

## Portable execution

From a repository that contains the frozen 428 extension tests and the retained producer package:

```sh
python3 -I -S -B tests/retained_extension_review/run_regressions.py \
  --repo "$PWD" \
  --build "$PWD/build/retained-gradients" \
  --output "$PWD/build/retained-extension-review-new"
```

The build directory must contain the exact retained `runtime.hepta`, `tensor-retained-session`, `build.json`, and the nine `.hepta`/`.o` module pairs. Output must not already exist. The script pins every original runner, protocol/client, observation report, prepared extreme plan and producer profile before starting a native process. It saves an input snapshot, the exact two-tag adapter proof and the commands/streams/child reports. No repository source is edited.

To archive a completed output without deleting it:

```sh
python3 -I -S -B tests/retained_extension_review/archive_evidence.py \
  --root build/retained-extension-review-new \
  --archive build/retained-extension-review-new.tar.gz \
  --manifest build/retained-extension-review-new.archive.json
```

Both archive output paths must be new. The archiver rejects symlinks and special files, streams the tar/gzip, reads every member back, verifies SHA/size/mode and gzip CRC/trailer, then rechecks the expanded originals. It removes nothing.

## Scope remains open

This package does not rerun or replace the producer's 17 inherited suites, retained actual 12 cases, optimizer case 14, resource eight, typed handoff two, or source review. Those receipts remain separate. It makes no new claim about bool masks, native dropout, unsupported dtypes, arbitrary layouts beyond these fixed observations, or the entire PyTorch project. Whole-PyTorch status remains **HOLD**, the API denominator remains **2,590**, newly closed API rows from this integration replay are **zero**, and `full_rewrite_complete` remains false.

The earlier retained-resource archive interruption and its explicit missing original raw tail are unrelated evidence and remain documented in that older package. This new extension archive does not repair, relabel or replace the older execution.
