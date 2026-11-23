# Exact retained indexing observers on the joint opt-in image

This wrapper runs the original typed `index_select` and indexed F32
forward/backward observers, including their existing physical readbacks.
It pins their source and provenance support files to the reviewed #63 tree
and separately authenticates the exact joint build through the opt-in
admission. The original profile must reject the noncanonical image, and
the same profile must accept the separately authenticated canonical build
and prove its own full source lineage, at both original call sites.

An additional reversible exact source check round-trips the joint monolith
with the authenticated canonical monolith. There are only four changed
function bodies, with all 800 functions in the same order. This does **not**
transfer any baseline execution result. The native observer launches the
joint executable and records its own actual frames; the unchanged
independent reader reapplies every comparison to those captured frames.
Neither suite uses the exploratory-oracle flag. Torch references are
produced afresh with the original supervisor and complete pinned-wheel
custody, including the full installed files and mapped Torch CPU code.

Reproduce the canonical and joint builds as in [README.md](README.md).
Prepare the original pinned official wheel, install reports, and shared
installed-member snapshot as in the Current HIA Runtime workflow. Use a
fresh output for each suite:

```sh
PKG="$REPO/tools/current_runtime/opt_in_dual_unary"
for suite in typed_index_select indexed_f32; do
  python3 -I -B "$PKG/replay_indexed_observers.py" --suite "$suite" \
    --repository "$REPO" --baseline-build "$BASE_BUILD" --dual-build "$DUAL_BUILD" \
    --dual-scoped "$PKG/qualification.json" --torch-python "$TORCH_PYTHON" \
    --output "$OUT/$suite" || exit 1
done
```

Actual original reports, raw per-session frames, and raw reference custody
stay in each output. Absolute paths, process IDs and timings may change
report digests across physical runs. The canonical files, numerical
predicates, protocol controls, and acceptance roster are unchanged. These
results apply only to the named mandatory suites and do not establish
the 35-suite aggregate or change the default runtime.

| Suite | Cases | Native sessions | Physical frames | Assertions |
| --- | ---: | ---: | ---: | ---: |
| Typed index_select | 866 | 869 | 17,153 | 64,271 |
| Indexed F32 | 331 | 334 | 9,938 | 37,094 |

The qualified runs preserve 78 and 56 additional protocol controls,
respectively, and verify 5 and 193 actual backward cases and 2 and 3 real
allocation refusals followed by successful identical requests. The 56
pinned checkout inputs match reviewed #63 Git blobs. The old canonical
build and an unrelated scope receipt were separately refused before starting
the native observer. See [compact evidence](indexed-observer-evidence.json).
