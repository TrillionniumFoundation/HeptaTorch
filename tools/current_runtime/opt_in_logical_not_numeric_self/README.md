# Physical qualification: exact numeric-self logical_not(out=)

Additive opt-in successor of exact HeptaTorch #65 f9807c9. The reversible edit extends the exact-self route to the five public numeric tags, preserving exact resolved public handle, Tensor node, Storage and Version identity and all ordinary CPU/contiguous/no-view/no-edge/no-grad/zero-offset, dtype, extent, and shape guards. The per-element kernel reads before writing and bumps the existing Version once per invocation, including empty. Numeric self's second application restores the truth mask encoded as zero/one, not the arbitrary original raw numbers. Distinct views/aliases, overlap, resize, autograd, production activation, Current803, and canonical changes are outside scope.

Two actual cold builds reproduced the same candidate ELF. The original exact predecessor rejected numeric self for all five numeric tags with full owner equality. Candidate qualification first captured 79 positive cases using the official locked CPU PyTorch wheel. Only after its process exited, a separate native observer without Torch bound the fixture's SHA-256, verified each label and full input digest and consumed all 79 cases exactly once. The splitter uses the same immutable combined observer and preserves the matrix, real Torch assertions, and native assertion bodies. The two initial combined-process memory stops are separately retained. The completed native run has 876 frames, 15 new numeric-self cases, all 64 prior positive cases and 82 exact rollback controls. A third, independent offline process raw-decodes request/response and owner snapshots, independently computes first/second numeric truth words, and checks identities and version transitions. Raw local snapshots and frames are temporary; the compact evidence is not a replacement for all raw files.

## Reproduce

PKG is this directory, PREDECESSOR_SOURCE and PREDECESSOR_ELF are exact #65 f9807c9 numeric-out source and image, HELPERS is the prior locked typed logical_not observer helper directory, and REPO supplies the original build, protocol and wheel lock. TORCH_PYTHON and PIP_INSTALL_REPORT refer to the actual installed official CPU wheel. The compiler, linker, images and helper bytes are checked.

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
    --source-root "$STAGE/source" --compiler "$COMPILER" --linker "$LINKER" --output "$STAGE/$b"
done
cmp "$STAGE/cold-a/heptatorch" "$STAGE/cold-b/heptatorch"
python3 -I -B "$PKG/split.py" "$PKG/observe.py" "$STAGE/sequential.py"
cmp "$PKG/sequential.py" "$STAGE/sequential.py"
"$TORCH_PYTHON" -I -B -W error "$STAGE/sequential.py" fixture \
  --output "$STAGE/fixture.json" --protocol "$PROTOCOL" --helper-dir "$HELPERS" \
  --oracle-lock "$LOCK" --install-report "$PIP_INSTALL_REPORT"
cmp "$PKG/oracle.fixture.json" "$STAGE/fixture.json"
python3 -I -B "$STAGE/sequential.py" baseline --binary "$PREDECESSOR_ELF" \
  --output "$STAGE/previous" --protocol "$PROTOCOL" --helper-dir "$HELPERS"
python3 -I -B -W error "$STAGE/sequential.py" candidate --binary "$STAGE/cold-b/heptatorch" \
  --output "$STAGE/physical" --protocol "$PROTOCOL" --helper-dir "$HELPERS" \
  --fixture "$STAGE/fixture.json" --fixture-sha256 a60b42ef20214f2f7c5a8c1863e2c09d620d5441b151690ee5fe6eef586e795d
python3 -I -B "$PKG/readback.py" "$STAGE/previous" "$STAGE/physical"
```

These runs do not establish whole-runtime or 35-suite qualification.
