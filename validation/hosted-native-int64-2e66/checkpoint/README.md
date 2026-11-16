# PR29 exact-source hosted admission: completed checkpoint

This checkpoint binds head `2e66ad094ca62d059a1a7ca5642a8ce27c40a949`, tree `51b2bf3b1d15e8662253668c0f6ee458885dd3de`, parents (in order) `0fd81b279c97dd969071c9efc51c8202fc16e45b` and `e0ea769c979236b0542af4b258890726cd01a99a`. The official PR checkout is `1a28a87524e7dd6ffe94ab0ba032d42c4cfb2031`, with parents main then the source head and the same tree.

## Completed actual runs

| Run | Job(s) | Actual steps | Independent checks |
| --- | --- | --- | --- |
| int64 PR 43886148312 | 133364701595 | 13 | 3,846 plus 638 historical assertions and frozen byte checks |
| int64 push 43886136338 | 133364660907 | 13 | 3,847 plus 638 historical assertions and frozen byte checks |
| quick 43886148242 | 355: 133364701492; 428: 133364701714; 446: 133364701057 | 13 each | 52,151 |
| Native PR 43886148135 | 133364700943 | 80 | 106,633 |
| Native push 43886136381 | 133364660371 | 80 | 106,633 |

All were actual attempt 1 terminal success, with 11 of 11 official source-head checks successful. Each dedicated reference retained 33 cases and 351,174 exact values. Their six-member ZIPs contain full compiler and two-generation ELF bytes, per-case requests, actual responses, framework expected bytes and complete provenance. The archived raw tar holds 433 files (207 CPython, 226 PyTorch). The frozen pure byte checker was independently rerun offline, with 2,057 CPython and 2,262 PyTorch assertions per artifact. No compiler, native application or Torch was rerun by this reviewer.

Both new Native artifacts were independently read in full: 63 original JSON reports plus the collector manifest, 33 envelopes, complete owner journals, and the fresh c71 retained-gradient producer with its 17 original and six outer suites. The 355/428/446 live32/frozen35 results include the original numerical, Version, EOF and eleven-provenance obligations. Original old50/58 reports, the 18 distribution gaps, partial scopes and raw-history caveats remain unchanged. Repeated runs and nested checker counts do not add API cases.

## Preserved history and exact corrections

`review/int64-pr-byte-admission-01/report.json` is the original local reviewer FAIL. Its frozen peer checker enumerates a set and filesystem entries, so full array-order equality was an invalid cross-process condition. `review/reader-history-01/source-correction.json` proves the separate v2 wrapper's exact inverse. It retains complete non-check equality, all row schemas/True values, complete check multiplicities, deterministic subsequence order and original CI JSON bytes. No arithmetic, expected bytes, tolerance, source pin or provenance check was changed. The frozen byte reader itself remains byte-identical.

`review/native-reader-inverse-2e66.json` proves the Native/quick reader's inverse to the original PR31 reader: only exact target identities, ordered source parents, tracked inventory (3,833 unchanged plus 28 additions) and descriptive labels change. All 98 selected source files and the retained-evidence reader are unchanged. The prior quick and preliminary README continue to say MAIN_PENDING because they are original historical snapshots; the later `review/final-admission-2e66.json` is the completed aggregate.

## Original bytes and replay

The eleven original hosted ZIPs are external to this small checkpoint. Their full artifact IDs, byte counts, SHA-256 values and original paths are in `review/original-11-zip-index-2e66.json` (185,608,714 bytes total). The immutable original API JSON and logs are included. The per-artifact tar duplicates are omitted because they are exact members of those ZIPs; the original reports and tar-expansion reclamation proofs are included. The historical PR28 first-failure ZIP is already a frozen repository file and is separately indexed in `external-inputs.json`, rather than copied again. Historical negative-control original ZIPs and correction evidence are included unchanged.

For a replay, restore the original ZIPs at paths specified by a newly relocated copy of the input manifest; never rewrite a report's bound CI path. Check out the exact published source to provide its immutable Git objects. Run the frozen wrapper with `--inputs`, `--historical-reader dependencies/review_integer_artifacts.py`, and a fresh `--output`. Native/quick replay uses `review/native-reader-2e66/review_integer_tree_native.py --bundle review/native-source-2e66 --inputs <relocated-inputs> --output <new-directory>`; `--quick-only` is only for the quick component. The repository history path can be relocated in the new input manifest. Every relocation must preserve the original byte and identity checks.

`manifest.json` binds every included ordinary file by byte count, SHA-256 and mode. The archive contains only those files plus the exact manifest. It has been fully decoded to one XZ EOF and every tar member read back with exact set, size, hash and mode checks. The Mac backup is a distinct transfer receipt and is not a prerequisite or a claimed result of this checkpoint.

The whole-project conclusion remains HOLD: zero complete schemas out of 2,590 and all 17 original gate states are unchanged. This is source-specific packed-integer and regression evidence, not general Tensor dtype parity.
