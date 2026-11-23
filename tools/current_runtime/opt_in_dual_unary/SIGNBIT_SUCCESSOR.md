# Retained signbit and opcode 127 successor on the exact joint image

The canonical [#61](https://github.com/TrillionniumFoundation/HeptaTorch/pull/61)
observer uses opcode 126 for `signbit`, and records that opcodes 125 and 127
are unassigned. The opt-in [#63](https://github.com/TrillionniumFoundation/HeptaTorch/pull/63)
image deliberately assigns 127 to `logical_not`. This observed contract change
is not a change to signbit numerical expected values. The original 127
refusal remains required on the canonical image.

1. Run the unchanged canonical signbit observer on exact ELF `083a92ad…`.
   It independently supervises the fixed official Torch wheel and its
   installed bytes, executes every original native recipe, and replays the
   actual physical frames and owner metadata. The baseline passed all 97
   numerical cases and nine extra recipes.
2. Re-run the old predicates on joint ELF `5af79385…`, admitted by the
   separate dual-image checker. The old observer fails, solely because
   opcode 127 returns success. Its original nonzero exit, raw input/output
   frames, and failing report are kept rather than overwritten.
3. Run the new candidate-specific successor for that exact row. It applies
   every other original signbit observation unchanged and checks actual
   fixed-version Torch `logical_not` on the same original 20-element F32
   boundary fixture. The native Bool result matches the raw output bytes,
   complete metadata and fresh Tensor/Storage/Version identities; every
   input descriptor word and the full input remain unchanged.
   The original readback rechecks all candidate frames, including that row.
   Opcode 125 still physically refuses with status 4 and no response record.

The runner pins the original source modules and preserves the original
admission denial. Candidate replay replaces only the reference-capture
block with readback of the fully verified **same original captured reference and
supervisor artifacts**; the candidate harness does not pretend to be the
original Torch supervisor. It separately verifies the candidate source,
objects and executable, and compares all 105 unmodified row results/check
counts/frame counts to baseline. The sole successor predicate uses fixed-source
reversible in-memory anchors. Exact original source files and build gates
remain unedited.

## Reproduce

From the exact #63 checkout, prepare the official pinned Torch wheel,
support install receipts and installed-member snapshot as in the original
Current HIA Runtime workflow. Then cold-build the exact canonical and dual
images as in the [dual build guide](README.md). Use fresh output directories.

```sh
R="$REPO/tools/current_runtime/opt_in_dual_unary/replay_signbit.py"
BASE_SHA="$(sha256sum "$BASE_BUILD/build.json" | cut -d' ' -f1)"
python3 -I -B "$REPO/tests/signbit/run.py" \
  --build-directory "$BASE_BUILD" --expected-build-sha256 "$BASE_SHA" \
  --torch-python "$TORCH_PYTHON" --output "$OUT/signbit-baseline"

# Exit 2 preserves the observed legacy candidate failure; the recorded
# original observer exit is 1. Any unexpected harness failure exits 1.
code=0
python3 -I -B "$R" --repository "$REPO" \
  --baseline-build "$BASE_BUILD" --baseline-run "$OUT/signbit-baseline" \
  --dual-build "$DUAL_BUILD" --dual-scoped "$REPO/tools/current_runtime/opt_in_dual_unary/qualification.json" \
  --torch-python "$TORCH_PYTHON" --mode retained --output "$OUT/signbit-retained" || code=$?
test "$code" -eq 2 || exit 1

python3 -I -B "$R" --repository "$REPO" \
  --baseline-build "$BASE_BUILD" --baseline-run "$OUT/signbit-baseline" \
  --dual-build "$DUAL_BUILD" --dual-scoped "$REPO/tools/current_runtime/opt_in_dual_unary/qualification.json" \
  --torch-python "$TORCH_PYTHON" --mode successor --retained-audit "$OUT/signbit-retained" \
  --output "$OUT/signbit-successor"
```

| Physical run | Original observer result | Frames | Checks |
| --- | --- | ---: | ---: |
| Canonical baseline | PASS | 2,347 | 8,843 |
| Joint, retained predicates | FAIL: historical 127 refusal only | 2,339 | 8,811 |
| Joint, explicit 127 successor | PASS plus full physical replay | 2,351 | 8,863 |

[Compact evidence](signbit-successor-evidence.json) binds the actual native
and reference reports and original wire bytes by hash. Exact report hashes
may vary on rerun because of paths, times and process identities. This is
signbit only, not the complete 35-suite gate or production activation.
