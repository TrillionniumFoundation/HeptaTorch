# Physical qualification: exact self Bool logical_not(out=)

Additive opt-in successor of [HeptaTorch #65](https://github.com/TrillionniumFoundation/HeptaTorch/pull/65), predecessor commit fac72fa, not canonical, Current803, or F32 autograd. Opcode 128 remains the same. The only behavior change is allowing its source and destination to resolve to the same public Bool Tensor, with the same node, Storage and Version. Existing CPU, contiguous, ordinary no-edge/no-grad, no-view, zero-offset, dtype and shape bounds still apply. The existing per-index read then write is safe when these owners coincide: its contiguous descriptor presents unique indexed elements; the version advances once after all writes. The unchanged trial owner rolls back aggregate failures. No general storage overlap, distinct aliases, resize, casting or autograd.

The predecessor physically refuses exact Bool self with status 7 and no records, preserving full owner bytes; its ordinary disjoint request still succeeds. This refusal is not relabeled a successor pass. In the successor, two reproducible cold builds produce the same ELF. Locked official CPU PyTorch 2.15.0 verifies six self cases (scalar, 1-D, 2-D, 4-D and two distinct empty shapes), return identity, first inversion, second restoration, stable destination and one bump per operation, including an empty in-place call. The 13 original disjoint cases still pass. Of 24 candidate rollback controls, 17 are carried over unchanged, the old exact-self rejection is replaced by refusing non-Bool self, and six are rollback-after-self-mutation tests. Distinct view aliases, other overlap and resize are out of scope, not additional physical cases.

## Reproduction

Use the exact pinned #65 opt-in disjoint source and ELF as PREDECESSOR_SOURCE and PREDECESSOR_ELF. REPO supplies the unchanged build and protocol files, PKG is this directory, and HELPERS contains the pinned original typed_logical_not comparison helpers (the observer verifies all three). Use the locked CPU Torch interpreter TORCH_PYTHON and its actual install report PIP_INSTALL_REPORT.

```sh
set -eu
STAGE="$(mktemp -d)"
BUILD="$REPO/tools/current_runtime/build.py"
PROTOCOL="$REPO/tests/integer_tensor_review/wire/protocol.py"
LOCK="$REPO/compatibility/pytorch-oracle-lock.json"
python3 -I -B "$PKG/prepare.py" "$PREDECESSOR_SOURCE" "$STAGE/source"
cp "$PKG/pinned-manifest.json" "$STAGE/source/manifest.json"
for b in cold-a cold-b; do
  python3 -I -B "$BUILD" --manifest "$STAGE/source/manifest.json" \
    --source-root "$STAGE/source" --compiler "$COMPILER" --linker "$LINKER" \
    --output "$STAGE/$b"
done
cmp "$STAGE/cold-a/heptatorch" "$STAGE/cold-b/heptatorch"
python3 -I -B "$PKG/observe.py" baseline --binary "$PREDECESSOR_ELF" \
  --output "$STAGE/previous" --protocol "$PROTOCOL" --helper-dir "$HELPERS"
"$TORCH_PYTHON" -I -B -W error "$PKG/observe.py" candidate --binary "$STAGE/cold-b/heptatorch" \
  --output "$STAGE/physical" --protocol "$PROTOCOL" --helper-dir "$HELPERS" \
  --oracle-lock "$LOCK" --install-report "$PIP_INSTALL_REPORT"
python3 -I -B "$PKG/readback.py" "$STAGE/previous" "$STAGE/physical"
```

The offline reader uses raw streams and snapshots independently of the protocol helper, and does not start the runtime. Compact evidence records their hashes; the initial physical captures were local temporary data. These tests are not whole-runtime or 35-suite qualification.
