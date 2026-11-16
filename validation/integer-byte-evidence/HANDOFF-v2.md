# Integer original-byte evidence: provenance revision 2

This additive revision is ready for source integration. It changes only the offline reader. The new observation-only integer oracles have not run against PyTorch yet; a new actual CI run must create and qualify those bytes. Existing hosted PASS results are preserved under their original source identities.

## Repository files

Use `proposal-v2/files/`. PR29 installs the two shared helpers, `tests/i64_oracle.py`, and `.github/workflows/native-i64.yml`. PR30 installs only `tests/i8_oracle.py` and `.github/workflows/native-i8.yml` after PR29 has supplied the shared helpers. The shared files are identical for both profiles.

Only `tests/check_integer_byte_evidence.py` changed relative to proposal v1:

- SHA256: `83b79fb7fd9f7b3fec442f45f448ab9648aea72c56e690c0c0a0cf4bd8bb3750`.
- Git blob: `bcfcce292b50adbbfca69388a9f03fbda8409bae`.
- The five other repository files retain their original v1 bytes, SHA256 and Git blob identities, listed in `proposal-v2/manifest.json`.

The reader adds one pure provenance function and one call to it. Removing those two exact insertions restores the entire v1 reader and its AST, including source positions. `proposal-v2/reader-provenance-inverse-proof.json` contains the literal inverse and the earlier independent reader's provenance function. All original arithmetic/corpus checks, response-byte comparisons, source and ELF pins, file inventories, and existing reader checks remain unchanged. The oracle, byte recorder, HIA kernels, compiler, and workflow algorithms are unchanged.

## Independent provenance checks

The new function requires the complete known top-level and observed field sets. It independently compares the installed distribution version, exact wheel URL, both `archive_info.hash` and `archive_info.hashes.sha256`, the install-report path and complete download object, metadata name/version, and `requested is True` to the frozen lock. All eleven actual check names and boolean results remain mandatory. Python and platform fields and the complete release source-equivalence object remain exact. The full Python version and pip version are recorded strings; the latter must also match the captured original pip report in the unchanged full-byte verifier.

Original CI paths remain observations. No report path is rewritten to the local extraction directory. The complete original pip JSON is captured by the byte recorder and independently cross-checked by the reader.

## Actual local execution

`cpython-control-v2/report.json` records two real observed CPython oracle executions, one per profile, using the fixed native compiler. The unmodified baseline and v1 executions were reused as immutable comparisons, not rerun.

| Profile | Cases | Exact values | Result |
|---|---:|---:|---|
| int64 | 33 | 351,174 | PASS |
| int8 | 28 | 1,093,216 | PASS |

Each new report is byte-identical to its v1 observed report. Compared with the original unmodified oracle's report, only the already declared oracle self-source SHA differs. The complete v1/v2 output inventories have identical member names; all original request, response, expected, compiler, application and report bytes are identical. Only `evidence/sources/reader.py` and the manifest describing that changed input differ.

Both offline full-byte readers passed, with 2,057 int64 and 1,738 int8 checks. These numbers count overlapping custody and numerical checks, not additional upstream API coverage.

`provenance-controls-v2/report.json` records two positive checks using original real hosted provenance JSON and six negative copies, three rules for each profile. The negative copies retain all eleven `checks` values as `true` while changing exactly one of:

1. The legacy `archive_info.hash` in `observed.direct_url`.
2. `observed.distribution_version`.
3. `observed.install_report.path`.

All six copies are rejected by their exact expected rule. The tests invoke only the new pure metadata function. They execute neither native code nor Torch and do not create simulated numerical expectations. The original source and reports are verified unchanged before and after.

## Commands

The dedicated workflows still use the original source, compiler, oracle commands and numeric assertions. They additionally make a mode-preserving `.tar.xz` containing the complete two reference output directories, execute the strict readers, and upload the archive and reader reports. Native ELF executable modes survive this archive even though an outer artifact ZIP normalizes individual file modes.

The portable offline command remains:

```sh
python3 -I -S -B tests/check_integer_byte_evidence.py \
  --directory build/i64-pytorch --profile i64 --reference pytorch \
  --output build/i64-pytorch-byte-review.json
```

For int8, use `build/native-i8-pytorch` and `--profile i8`. For local CPython evidence use `--reference cpython`. The output must be a new file outside the evidence directory. The reader executes no compiler, native application or Torch operation.

## Custody and limitations

The original v1 proposal, original four CPython executions, copied hosted artifacts, and v1 raw archive remain unchanged. Revision 2 has a separate source/proof handoff and a separate complete raw archive. Each archive is verified member-by-member for SHA256, length, mode and exact membership; the full XZ stream is consumed through its trailer. The handoff manifest lists all identities and the external full raw archive.

No new actual PyTorch byte-evidence run has occurred locally. The original integer artifacts lacked full binary/request/response/expected byte files; this revision does not pretend to reconstruct their missing history. Future CI will provide those original bytes under the new observation-only oracle identities. The original numeric tests, 11-source requirements, integer profile bounds, and whole-PyTorch `HOLD` remain unchanged.
