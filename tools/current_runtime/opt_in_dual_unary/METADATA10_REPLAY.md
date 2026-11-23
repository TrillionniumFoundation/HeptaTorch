# Retained Metadata10 on the exact joint image

This is one bounded candidate replay. The original canonical files, old
failure evidence, and all numerical and metadata expectations are unchanged.
The historical reference archive is fixed: all 15 original member bytes and
modes and the complete original 11-check live Torch provenance are rechecked.
No new Torch calculation or version offset is substituted.

The original worker, native driver and independent reader run with exact,
reversible admission-only substitutions. The wrapper uses the existing
dual-image identity checker, requires the original canonical path to refuse
the candidate, then uses a separate in-memory copy of the full original
admission verifier with only its manifest path changed to the exact joint
profile. The original 270-based owner binding, object identity, process EOF,
link input and stack checks still apply. Other driver edits remain the original
two reversible source/image literals.

The spawned worker's command and streams are physical observations. The
independent reader checks the actual launcher path rather than relabeling it
as the canonical worker. It then replays every retained raw frame and keeps
the original complete comparison and passive instrumentation. The 24 exact
source/archive blobs in the catalog were matched to the pinned #63 tree.

## Reproduce

Use a fresh exact #63 checkout containing this wrapper plus the full joint
source/build from the [dual build guide](README.md) and its scoped receipt.

```sh
P="$REPO/tools/current_runtime/opt_in_dual_unary"
python3 -I -B "$P/replay_metadata10.py" --repository "$REPO" \
  --dual-build "$SCRATCH/dual-build" --dual-source "$SCRATCH/dual-source" \
  --dual-scoped "$P/qualification.json" --output "$SCRATCH/metadata10"
```

Both cold builds of the same exact joint ELF passed independently: 10/10
original recipes, 791 physical frames, 814 original assertions, 20 native EOF
observations, 228 owner snapshots and 152 Tensor observations per replay.
The independent reader rechecked 116 original inputs. Wrong canonical image,
wrong staged manifest and wrong scoped receipt were rejected before the worker
could start. See [compact evidence](metadata10-evidence.json).

Raw original report and reference/readback artifacts remain in the output
directory. Actual report digests can differ by paths or process identity.
This does not qualify the other 34 suites, the separate 803 view work, or
activate the candidate.
