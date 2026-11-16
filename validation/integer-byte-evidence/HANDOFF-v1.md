# Integer original-byte custody proposal

## Result and limits

The original int64 and int8 oracles and the versions with byte-observation
calls have actually run against the same pinned native HIA compiler. All four
CPython-reference runs passed. For each profile, the entire old and new JSON
reports are equal except the expected oracle-source identity field. This
includes every case, all input/result/reference hashes, exact values, refusal
results, source/ELF/compiler identities and existing success assertions.

| Profile | Original and observed result | Full new byte verification |
| --- | --- | --- |
| int64 | 33 cases, 351,174 values | 207 files, 2,057 checks, PASS |
| int8 | 28 cases, 1,093,216 values | 177 files, 1,738 checks, PASS |

The unchanged baseline report bytes also reproduce the original official
reports: int64 `25007f2b31ffac5deedd8608f98c64008895110d2b3ecdf3e72b4d887f5a41ad`
and int8 `c821ba1bade12c4182abfd464fa02529c311c424dea021f9b20e220dd480efee`.

The new PyTorch path is **NOT_RUN locally**. Its real expected bytes and full
pip record must come from the new hosted execution; no old PASS is transferred
to the new sources or combined main tree. No remote reference was changed.

## Install exactly six files, in the serial PRs

All proposed repository files are under `proposal/files/`.

For PR29, install:

1. `tests/integer_byte_evidence.py` — shared byte recorder.
2. `tests/check_integer_byte_evidence.py` — shared strict offline reader.
3. `tests/i64_oracle.py` — original oracle plus observation calls.
4. `.github/workflows/native-i64.yml` — original steps, one added archive/read
   step, expanded trigger paths and extra artifact members.

For PR30, after it is based on the resulting main, install only:

1. `tests/i8_oracle.py`.
2. `.github/workflows/native-i8.yml`.

The two shared files are byte-identical for both profiles. They must be reused
unchanged from PR29. The two HIA kernels, compiler/source pins, existing
numerical formulas, corpus, thresholds, subprocess arguments and original
assertions remain unchanged. The int8 workflow retains the already repaired
`python3 -B` Torch invocation; compiler and CPython invocations retain `-I -S -B`.

`proposal/manifest.json` gives size, SHA-256 and Git blob SHA-1 for every one of
the six files and the exact per-PR mapping.

## What the new observations retain

Each oracle saves complete original compiler stdout, stderr and return code
for both actual application compilations. It saves the real compiler ELF,
application ELF, HIA source, oracle, observer, reader, fixed source-provenance
helper, lock, and (for the actual Torch reference) the complete original
pip-install record. Each case saves its original full request, original native
stdout/stderr/return code, CPython expected bytes and the selected expected
bytes. Nonempty Torch cases additionally save bytes directly serialized from
the original real Torch operation result, before its unchanged comparison.

An empty case still follows the original oracle, which does not call a Torch
arithmetic operation. It is explicitly marked `empty-no-framework-operation`;
no synthetic Torch-result file is created. Protocol refusals retain the exact
four-byte literal and are labelled `protocol-literal`.

The recorder computes no numerical values, changes no existing report rows,
and never invokes a compiler, native application or Torch itself. Its final
manifest binds every original input before/after and every actual output file.
Only its own manifest is excluded from that file inventory.

The offline reader requires the complete nonempty fixed case plan, complete
file inventory and actual files with exact set equality. It verifies all
source and binary pins, both original compiler outputs, every request,
response and expected byte string, original case/report hashes, exact totals,
and all eleven real-wheel provenance checks. It independently reconstructs
the original fixed request/response hashes using five byte-identical pure
functions from the already executed independent integer-artifact reader.
It does not launch a native executable, a compiler, or PyTorch.

## Workflow and failure custody

Both workflows preserve every original step and its ordering. Before upload,
an `if: always()` step puts both original output directories in a `tar.xz` and
then runs the strict offline readers. The original report upload paths remain
unchanged; the tar archive and two new reader reports are added. The tar
archive preserves the original ELF file modes that direct artifact ZIP upload
would otherwise normalize. A failed run remains failed; a partial/missing
archive or reader report cannot satisfy the new complete-byte gate.

The initial prepared directory-upload draft is retained under
`proposal/prepared-workflow-draft-01/`; it was never run in CI. The final
archive version was selected to preserve file modes. YAML parsing and all new
CLI flags were checked against the actual parser shape. Each workflow now has
nine declared steps; the actual hosted job count remains to be observed.

`proposal/oracle-inverse-proof.json` records every inserted observation block.
Removing those exact blocks restores each entire original oracle byte for
byte, including its complete AST and positions. The original assertions,
cases and subprocess calls therefore survive exactly. Likewise,
`proposal/workflow-inverse-proof.json` restores the two complete original
workflows by reversing only the explicit trigger/archive/reader/upload edits.

## Executed local evidence

`cpython-control-01/report.json` records all four actual commands, raw
stdout/stderr, source/compiler before-and-after identities, original and new
report identities, and complete-report equality except the oracle's own SHA.

The full original observed bytes are frozen in
`cpython-original-bytes-01.tar.xz`:

- 10,140,076 bytes.
- SHA-256 `1c0ed412e93464c4c02c4a09016a6e9770505c21295049895179ab5915de7906`.
- 453 tar members including directories; 384 regular files.
- Exact member set, each SHA-256, length and mode, and full XZ trailer verified.

The archive was safely extracted into `archive-roundtrip-01/`, preserving
original modes, and both offline readers passed again with 2,057 and 1,738
checks. This transport check launches no additional native or Torch program.

The source/proof handoff is small and references that separately retained
complete raw archive. No raw bytes are replaced with a hash-only report.

## Run the strict offline reader

After extracting the original archive with its modes preserved:

```sh
python3 -I -S -B tests/check_integer_byte_evidence.py \
  --directory build/i64-cpython \
  --profile i64 --reference cpython \
  --output build/i64-byte-review-cpython.json
```

Use `--profile i8` and its `build/native-i8-*` directories for int8. The reader
output must be a new file outside the original evidence directory. Historical
CI absolute input paths remain in the manifest as observations; offline
verification uses their exact captured bytes and roles, not local path
equality.

These changes strengthen evidence custody. They do not expand the bounded
standalone integer operation scope or close general Tensor dispatch,
broadcasting, storage, autograd, or full PyTorch schema parity. Overall status
remains HOLD, full-schema parity 0 of 2,590.
