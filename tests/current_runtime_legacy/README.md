# Original view and scalar regressions on the current runtime

`run.py` executes all 76 original cases against the complete Bool700 HIA image:
view/reshape26, wrapped scalar32, direct view14 and input boundary4, in that order.
The source and executable identities are fixed to the current Bool700 profile.
The build-report SHA is supplied for the particular new cold build being tested.

```sh
python3 -I -S -B tests/current_runtime_legacy/run.py \
  --build-directory build/current-runtime \
  --expected-build-sha256 <SHA256-of-build/current-runtime/build.json> \
  --output build/current-runtime-legacy76
```

The output directory must be new. A source/reference/build preparation check can
be made with `--prepare-only`; it returns `PREPARED_NOT_RUN` and starts no Tensor
session. The actual command returns zero only when all four full suites pass,
all 76 cases execute, the inherited EOF checks pass and every bound input remains
unchanged. There is no case-selection option on this entry point.

## Frozen inputs and explicit profile binding

The four comparison programs are recovered byte-for-byte from
`continuation/20261117/evidence/root-sum-integral-original76-reviewed-intent-v1.json`.
The complete intent SHA, embedded source manifest, all twelve source payloads
and their original sizes are checked. The programs' original Tensor/Storage,
value, geometry, Version, Saved, gradient, Weak, RNG, rollback and EOF assertions
are retained. Their actual exit codes and reports are kept separately.

The existing read-only observer already handles Bool storage with one-byte
elements. Its only source changes are the two format tags `HTSINT01` → `HTSINT02`
and `HTOSIN01` → `HTOSIN02`. The complete reverse byte transformation is checked
and recorded. None of the numerical expectations or comparison methods change.

The new builder's original report is retained as `profile/actual-build.json`.
A separate `profile/build.json` names its own bridge schema and adds the
672-byte descriptor and current format tags for the frozen readers. It binds
the original report's SHA and every newly compiled object, source unit and
native compiler/linker process. The actual original receipt is never edited.
The runtime/source copies are exact bytes of the supplied new build.

## Reference recovery

The default reference source is the already published archive
`validation/view-reshape-r4-native76/complete.tar.xz`, SHA-256
`511dff328018a0a2d6a56a94fde4d2e0d64ff10ea66bb0cb71f89bb61ab2a3fb`.
Only the original four observation JSON files and their four probe source files
are selected by their fixed SHA-256 and size. All extracted destinations are
chosen by this runner; archive paths are recorded as provenance and never used
as filesystem extraction paths. The original observations are replayed without
running Torch or modifying reference values.

`--reference-archive PATH` selects the same exact archive at another location.
For already recovered files, `--references DIR` accepts these eight paths,
still requiring their original byte identities:

```text
view26/actual.json     view26/probe.py
scalar32/actual.json   scalar32/probe.py
view14/actual.json     view14/probe.py
input4/actual.json     input4/probe.py
```

Besides the intent and reference archive, the immutable supporting files are:

- `tests/differentiable_view_native_review/protocol.py`
- `tests/differentiable_view_native_review/owner_client.py`
- `tests/differentiable_view_native_review/replay_direct_view14.py`
- `tests/differentiable_view_shape_review/native_observer_shape02.py`
- `tests/differentiable_view_shape_review/identity_epochs.py`
- `tests/differentiable_view_shape_review/replay_shape26_r3.py`
- `tools/current_runtime/build.py`

Their hashes are checked against the frozen configuration or explicit builder
pin. These tests use Python's standard library and the native HIA application;
they need no installed numerical runtime.

## Outputs

The output tree includes the exact frozen source payloads, observer inverse
record, current build binding, four original reports, full raw per-session
observations, both child output streams and a combined `report.json`. A failure
keeps its original report and raw evidence, and the other suites still run.
The combined result describes these 76 inherited cases on one exact current
image. It does not change the full-project schema or subsystem denominator.

## Original F32 Sum32 on the same image

`sum32.py` runs the original 32 F32 sum cases and all 626 recorded events against
the same Bool700 build. It accepts the recovered original source packages and
actual reference as explicit inputs:

```sh
python3 -I -S -B tests/current_runtime_legacy/sum32.py \
  --build-directory build/current-runtime \
  --expected-build-sha256 <SHA256-of-build/current-runtime/build.json> \
  --original-source /path/to/sum-integral-original32-source-01 \
  --parent-source /path/to/sum-dims-native32-source-03 \
  --reference-source /path/to/sum-dims-f32-reference32-source-01 \
  --reference /path/to/sum-dims-f32-reference32-actual-01/actual.json \
  --output build/current-runtime-sum32
```

These inputs are independently pinned inside the runner:

| Input | SHA-256 |
| --- | --- |
| Original Sum32 source manifest | `4676870b32dd341f9e6c30d170e8c5a4abbc401668ed3cdae492d9712233e138` |
| Parent native32 source manifest | `f13a204f5071c07b34ab35ee1b4a6dc90163137b0762fb57bd8c34abb0c8b277` |
| Original reference source manifest | `7db1e860e6387a0d48b98769eb9f124211a90b92bfb62a07e55eb294ab4ef780` |
| Original actual reference, 39,045,926 bytes | `c14e4b18f657429ad5dac5498c3abcae303b6a7e7550db7d15a3c9179bdcd728` |

The complete original 27,544-byte `SumCase` and 26,961-byte reference admission
code are retained. The original 2,514-byte native constructor also remains
unchanged. The parent packages' complete forward/reverse source checks run
before any Tensor session. The strict Integral owner/observer is recovered from
the already committed native24 reviewed intent. Its derived current profile
changes only its name, function count and format tags; the F32 and I64 rules
remain unchanged.

Two source-admission predicates bind original directory names to their frozen
manifest fields, allowing the source packages to be stored at local paths.
The reference reader receives an explicit mapping from each of its 41 original
input paths to a local file with the same SHA-256, byte count and mode. The
original JSON, source input records, numerical expectations and comparison
predicates remain unchanged. The two external installation/inventory JSON
inputs are recovered from their byte-identical copies in the reference source
package. No Torch process runs during this regression.

`--prepare-only` performs the complete source, reference and cold-build
admission and returns `PREPARED_NOT_RUN`. An actual success requires all 32
cases, all 626 events, 32 successful native exits with both streams at EOF,
empty unexpected output and unchanged inputs. Every original case retains its
full report, frame journal, compressed complete owner snapshots, raw requests
and responses. `report.json` has schema
`heptatorch.current-runtime.original-sum32-comparison.v1`; its `status` must be
`PASS`, `passed` must be 32, `completed_reference_events` must be 626,
`all_native_EOF` and `all_inputs_unchanged` must both be true. A historical result
never supplies these current-image results.
