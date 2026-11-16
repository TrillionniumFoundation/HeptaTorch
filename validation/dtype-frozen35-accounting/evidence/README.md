# Complete 523 frozen35 evidence

This additive package finishes the byte-custody task for source package
`934aa2916e6cd00d634839b9b3676b73b62dfe70815a8a44b043dfb65b91d3b3`.
It does not change that source package, the native program, or the earlier
355, 428, 446 and 640 observer versions.

The actual ROG runs used the fixed 523-function native source `6ffe509a…`
and ELF `830f0b4e…`, CPython 3.12, and PyTorch 2.15.0+cpu. All eleven
original provenance checks passed in each of the four reference reports.

| Original run | Outcome retained here |
| --- | --- |
| Unmodified historical frozen35 on this 523 ELF | FAIL, 32 of 35 rows; obsolete public node ledger |
| Unchanged live32 oracle on this 523 ELF | PASS, 32 rows, 2,132 frames |
| First full640 frozen35 observer | FAIL, 4 of 35 rows; incorrect expired-Storage size check |
| Corrected full640 frozen35 observer | PASS, 35 rows, 6,344 frames, 2,590,274 compared values |

The final run kept every original EOF and post-close Version gate. It also
kept the 283 explicitly reported historical state-Version differences and
required zero direct-parameter Version mismatches. No original numerical
row was reclassified by the wrapper. The correction changes the observer:
the native Storage destructor clears offset and rounded extent but retains
the requested byte size while a weak reference keeps the tombstone alive.
The exact inverse patch, native six-frame control, four malicious copied
states, pure replay of the prior structural evidence, and independent
source-only review are all retained separately. They are not extra API cases.

## Restore and inspect original bytes

From a repository checkout containing this package:

```sh
python3 -B validation/dtype-frozen35-accounting/evidence/restore_evidence.py \
  --output /tmp/heptatorch-523-evidence-new --extract
```

The output path must not exist. `--extract` is optional: without it the tool
still joins every part, verifies both complete archives and every member,
and leaves the exact recovered XZ streams. It never executes an image,
native test, PyTorch, or a child script.

The restorer pins `evidence-index.json`, verifies each ordered part and the
whole XZ SHA, rejects trailing or concatenated XZ data, verifies every regular
tar member's name, size, SHA and mode, and rejects links or path traversal.
With `--extract` it also rechecks each extracted file and its original mode.
The included `restore-verification.json` records an actual complete local
restore and extraction of both groups.

- `remote-raw`: the byte-identical 15,587,512-byte ROG archive, split only to
  fit transport limits; 450 members and 94,644,488 expanded bytes.
- `local-supplement`: 328 original local control, peer-review, source-history,
  transport-failure and download-custody members. The prior 4/24 evidence
  archive is an unchanged input here, not a repeated native execution.

Original reports are inside those archives. Child JSON absolute paths and
argv were not rewritten. The summary gives exact member paths and hashes.
The remote archive includes the exact source, ELF and build receipt used by
the runs; all eleven individual compile-unit products belong to the separate
producer reproduction package. This archive does not assert a new build.

The malformed literal trailing `\n` in the original baseline launcher
metadata is retained byte-for-byte. Its original numerical report is valid
JSON and is independently pinned. SSH read interruption and retry records
are also retained. Reading and packaging these files did not rerun native
or PyTorch tests.

**Whole PyTorch remains HOLD, denominator 2,590, rewrite complete false.**
This component receipt does not close all dtype, autograd, or RNG coverage.
The official cold-build/CI gate is distinct from these fixed-image ROG runs.
