# Exact retained forward and join observers on the joint opt-in build

Both original suites run on the exact #63 joint executable; this does not
transfer the canonical result. The retained observer and its original
physical readback must each see the unchanged canonical profile refuse the
joint image at its original manifest gate. Each then separately runs the
full unchanged historical source proof on the authenticated canonical build.
The joint executable, objects and link request are independently admitted.
The entire joint source round trips to the exact canonical 800-function
source: only four bodies differ and their definition order is unchanged.
Only the two profile calls in each original runner are replaced reversibly
in memory. Numeric and protocol predicates, including unknown opcode 125,
are unchanged.

Use the same canonical and joint cold-build inputs as [README.md](README.md).
Prepare the original pinned official wheel, install reports and installed
member snapshot as specified by the original workflow. For each suite use a
fresh output directory:

```sh
PKG="$REPO/tools/current_runtime/opt_in_dual_unary"
for suite in forward_nograd join_forward; do
  python3 -I -B "$PKG/replay_forward_join.py" --suite "$suite" \
    --repository "$REPO" --baseline-build "$BASE_BUILD" --dual-build "$DUAL_BUILD" \
    --dual-scoped "$PKG/qualification.json" --torch-python "$TORCH_PYTHON" \
    --output "$OUT/$suite" || exit 1
done
```

The original numerical references are freshly observed from the exact
official PyTorch wheel with the full installation/source supervisor, not
the exploratory oracle. Original reports, supervised inputs, and native
frames stay in their respective output directories. Report hashes can
vary with physical process identities, paths, and timing. This is scoped
evidence for the named suites only, not the full 35-suite gate.

| Suite | Cases | Native sessions | Physical frames | Assertions |
| --- | ---: | ---: | ---: | ---: |
| Forward no-grad | 462 | 465 | 12,108 | 48,630 |
| Join forward | 414 | 416 | 14,256 | 53,638 |

The original readbacks also verified 243 and 274 actual backward cases,
respectively, and 3 and 2 real allocation refusal/recovery sequences.
Each suite's real reference receipt verifies 12,567 installed members and
mapped Torch CPU code. All 57 pinned checkout files match the reviewed
#63 Git tree; a canonical image and wrong opt-in receipt were separately
refused before native observation. See
[compact evidence](forward-join-evidence.json).
