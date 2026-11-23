# Exact retained spatial observer on the joint opt-in build

The original `spatial_forward` runs against the exact #63 joint executable,
not a copy of canonical execution results. The two profile calls (native
observation and independent raw physical readback) are replaced reversibly
in memory. Each requires that the unchanged canonical reader rejects this
joint build, then runs the unchanged spatial profile on the separately
authenticated canonical build. That old spatial profile delegates to the
original indexed source lineage proof. The joint monolith round trips to the
exact admitted current800 source with only four changed function bodies
and the same definition order. Its executable, objects and link request
are separately admitted. No numerical predicate, expected refusal, or
old observer file is edited.

Reproduce the canonical and joint cold builds as in [README.md](README.md).
Prepare the original pinned official wheel, install reports and installed
member snapshot as in the original workflow. For fresh output run:

```sh
PKG="$REPO/tools/current_runtime/opt_in_dual_unary"
python3 -I -B "$PKG/replay_spatial_forward.py" --suite spatial_forward \
  --repository "$REPO" --baseline-build "$BASE_BUILD" --dual-build "$DUAL_BUILD" \
  --dual-scoped "$PKG/qualification.json" --torch-python "$TORCH_PYTHON" \
  --output "$OUT/spatial-forward"
```

Original supervised Torch evidence and raw native frames remain within the
output. The independently checked original reference is freshly generated
from the exact pinned official wheel and verified installed member bytes
and mapped Torch CPU code. Report digests can vary with process identities,
paths, and timing. All 57 tracked input pins match the reviewed #63 Git
tree. Wrong canonical and wrong scoped receipt controls are refused before
the native observer starts. This is suite-only evidence, not a pass of the
35-suite gate or production activation.

The recorded run passed 344 cases in 347 native sessions, with 11,545
frames, 62,006 assertions, 3,477 owner and 24 mapping snapshots. It retained
28 protocol controls, 186 actual backward cases and 258 gradients, and
three real allocation refusal and identical-request recovery sequences.
See [compact evidence](spatial-forward-evidence.json).
