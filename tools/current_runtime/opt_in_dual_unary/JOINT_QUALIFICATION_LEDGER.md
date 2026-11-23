# Joint opt-in runtime qualification ledger

This ledger maps the [35 mandatory suites](../../../tests/qualify_current_runtime.py)
to the separately published replay evidence for the joint opt-in image
`5af793857b1606f4760a2cae1d5ce9c5e8f821e07fb499ac0349567f2a503b72`
(source `f4c43f037a3fb046425f7634215ec382aa5f8d6b0e7abe126ef64236adb5753a`).
The denominator is the original 16,365 cases and does not include extra
native sessions, controls, cold-build repetitions or prerequisite references.

**Status: 35 of 35 suites have separate passing replay evidence.**
The covered suites account for all 16,365 original roster cases. This is coverage arithmetic, not a single-run case count.
A separate replay can pass an explicitly scoped successor check without
passing the original, unchanged aggregate current-runtime gate.

| Mandatory suite | Cases | Joint replay evidence |
| --- | ---: | --- |
| `bool27` | 27 | [Historical sum](historical-sum-evidence.json), retained inner failure |
| `original76` | 76 | [Original76](original76-evidence.json) |
| `integral24` | 24 | [Historical sum](historical-sum-evidence.json), retained inner failure |
| `sum_f32_17` | 17 | [Historical sum](historical-sum-evidence.json), retained inner failure |
| `current35` | 35 | [Current35](current35-evidence.json) |
| `base270` | 270 | [Base/Cat](base-cat-evidence.json) |
| `sum32` | 32 | [Sum32](sum32-evidence.json) |
| `live32` | 32 | [Live32](live32-evidence.json) |
| `logical147` | 147 | [Logical147](logical147-evidence.json) |
| `resource7` | 7 | [Resource7](resource7-evidence.json) |
| `asstrided16` | 16 | [As-strided16](asstrided16-evidence.json) |
| `cat4` | 4 | [Base/Cat](base-cat-evidence.json) |
| `metadata10` | 10 | [Metadata10](metadata10-evidence.json) |
| `generator131` | 131 | [Generator131](generator131-joint-evidence.json), includes the separate required probe/reordered images |
| `public_f64_logical192` | 192 | [Public F64 logical](public-f64-logical192-evidence.json) |
| `f64_48` | 48 | [F64_48](f64-48-evidence.json), original independent physical readback |
| `half47` | 47 | [Half47](half47-evidence.json) |
| `half_logical264` | 264 | [Half logical](half-logical264-evidence.json) |
| `half_transactions12` | 12 | [Half transactions](half-transactions12-evidence.json) |
| `numeric_predicates` | 485 | [Numeric predicates](numeric-predicates-evidence.json) |
| `bf16_copy` | 280 | [Direct observer](direct-observer-evidence.json), BF16 pass rows only |
| `half_mul` | 312 | [Half multiplication](half-mul-evidence.json) |
| `half_neg` | 371 | [Half negation](half-neg-evidence.json) |
| `half_addsub` | 1,006 | [Half add/sub](half-addsub-evidence.json) |
| `wide_neg` | 405 | [Wide negation](wide-neg-successor-evidence.json), scoped ABS successor |
| `half_div` | 1,196 | [Half division](half-div-evidence.json) |
| `public_f64_arithmetic` | 2,541 | [Public F64 arithmetic](public-f64-arithmetic-evidence.json) |
| `same_f64_arithmetic` | 5,088 | [Same-F64 arithmetic](same-f64-arithmetic-evidence.json) |
| `typed_relu` | 776 | [Typed ReLU](typed-relu-successor-evidence.json), scoped ABS successor |
| `signbit` | 97 | [Signbit](signbit-successor-evidence.json), scoped opcode-127 successor |
| `typed_index_select` | 866 | [Indexed observers](indexed-observer-evidence.json) |
| `indexed_f32` | 331 | [Indexed observers](indexed-observer-evidence.json) |
| `forward_nograd` | 462 | [Forward/Join observers](forward-join-evidence.json) |
| `join_forward` | 414 | [Forward/Join observers](forward-join-evidence.json) |
| `spatial_forward` | 344 | [Spatial forward](spatial-forward-evidence.json) |

## What the separate replays do and do not establish

- `wide_neg` and `typed_relu` retain the original joint failure reports
  for cases 403 and 404. Their exact successor readers preserve the
  other 403 and 774 case rows, respectively, and verify both original
  requests against the observed I64/F64 ABS successes. The original
  refusal predicates were not retroactively reported as passing.
- `signbit` retains the original joint failure for its opcode-127
  unknown-opcode control. The scoped reader keeps 105 original recipes,
  still checks opcode 125 as unknown, and checks the newly admitted
  opcode 127 against the observed logical-not result.
- `bool27`, `integral24` and `sum_f32_17` keep the original inner
  dtype-refusal failures in their historical reports and count only the
  passing outer successor checks. Canonical reference runs, negative
  controls, temporary partial attempts, and duplicate cold runs are not
  counted as additional suites.
- `generator131` has one roster entry. Its main image is the exact
  joint image; the gate also binds its required separate probe and
  reordered images.

These scoped results have not been executed as one hosted aggregate run.
The complete successful raw archives for F64_48, Half47, and Generator131
were verified by byte and mode, then retained in Dot's persistent
workspace; the repository contains compact evidence and hashes, not the
archives. The unchanged canonical acceptance gate
has not been relaxed. The joint image remains opt-in and the full PyTorch
rewrite remains open.
