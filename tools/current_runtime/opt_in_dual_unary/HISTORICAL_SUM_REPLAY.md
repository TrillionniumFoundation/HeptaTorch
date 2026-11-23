# Strict historical Sum17, Bool27 and Integral24 on the joint opt-in image

These runs qualify the same bounded Sum17, Bool27 and Integral24 scopes that
the original current800 gate qualifies, but on the physically executed
joint ELF. They do not transfer the canonical outcome. The older raw
observers preserve their intentional exact legacy refusal failures:
Bool indices 2, 3, 5 and 6 and Integral indices 0, 1, 2 and 3. The same
unchanged outer validators run the original comparison and raw transport
checks, confirm the existing bounded current800 successor semantics, and
keep those original inner reports in FAIL status.

The wrapper pins the checkout files from the reviewed #63 tree and creates
a scratch-local source layout. Only two files in that layout have an
append-only, reversible identity bridge:
- The full original build reader still refuses the joint image against
  the old canonical manifest and still admits the exact canonical build.
  The candidate-specific profile checks the actual candidate with the same
  full original cold-build qualifier, using its independently hash-bound
  source/manifest, and the separate exact joint admission also checks it
  before and after observation.
- The exact candidate monolith round-trips to canonical current800, with
  only four function bodies and no definition order changes. The original
  historical lineage chain proves the independently bound canonical source.
  It does not serve as evidence of candidate execution.

No checkout files, historical predicates, old report data, or upstream
CI gates are modified.

## Reproduce

Use an exact #63 checkout, the respective canonical and joint cold builds
and the joint source directory from [README.md](README.md), and the original
mode-preserving frozen Bool27, original-sum, F32 owner reader artifacts as
prepared by the [canonical workflow](../../../.github/workflows/current-runtime.yml).
These are the original authenticated evidence inputs, not regenerated
approximations. The scratch root must be on the same filesystem as the
checkout, so the unmodified input files can be linked without duplicates.

```sh
PKG="$REPO/tools/current_runtime/opt_in_dual_unary"
for suite in sum_f32_17 bool27 integral24; do
  python3 -I -B "$PKG/replay_historical_sum.py" --suite "$suite" \
    --repository "$REPO" --pins "$PKG/historical-sum-pins.json" \
    --baseline-build "$BASE_BUILD" --dual-build "$DUAL_BUILD" \
    --dual-source "$DUAL_SOURCE" --dual-scoped "$PKG/qualification.json" \
    --sum-reference "$F32_REFERENCE" --bool-reference "$BOOL_REFERENCE" \
    --references-directory "$FROZEN_SUM_INPUTS" --owner-directory "$F32_READER" \
    --output "$OUT/$suite" || exit 1
done
```

Only these three suites are covered by the recorded evidence. Historical reference data are
replayed, not newly generated. Raw output reports and exact wire/stability
records stay in the fresh output directories. Digests that bind those
reports can vary with physical process identity, local paths and timing.

The joint run passed Bool27's 27 cases across 36 native children and 6,099
frames, and Integral24's 24 cases and all 260 original events across 29
native children and 14,504 frames. Both preserved the old classifier
failures given above, with all paired native raw frames, owner boundaries
and both EOFs actually verified. See
[compact evidence](historical-sum-evidence.json).

The Sum17 replay passed across 33 native children and 8,194 frames.
The unchanged outer observer checked all 31 original native sessions and
two independently observed Bool dtype successes on the exact joint image.
The original report remains FAIL, solely for historic dtype-refusal
assertions at indices 5 and 6; its raw frames and the two new accepted
cases were independently checked. The first attempt did not reach native
execution because the frozen Bool reference archive was missing from the
scratch inventory. The corrected inventory adds only two files already
present on #63: that frozen archive and the extension report's exact
reference source. Earlier raw failure digests are retained separately.
