# Physical qualification: disjoint CPU numeric logical_not(out=)

Additive opt-in successor of [HeptaTorch #65](https://github.com/TrillionniumFoundation/HeptaTorch/pull/65) at ed155be. Opcode 128 keeps its existing guards, version and owner transaction. A disjoint preallocated destination can now also be F32, I64, F64, F16 or BF16. It encodes the two logical values as exact positive zero and one in the destination dtype and writes exactly its dtype width. The old disjoint Bool and exact same-handle Bool profiles remain active. Exact-self numeric (whether Torch permits it or not), views, overlap, resize, autograd, non-CPU and mismatched shapes are not admitted or claimed. Both tensors must be ordinary CPU, same shape, zero offset, contiguous, no-edge/no-grad and within old rank/count limits. No canonical, Current803, or F32 autograd changes.

Official locked CPU Torch preflight passed all 36 public source/output type pairs. The physically qualified candidate checked all 30 public-source/numeric-output pairs and 15 additional scalar, empty and rank-4 cases. The old 13 disjoint Bool and six exact-self Bool cases also passed. Two cold builds produced identical binaries. The exact predecessor rejected the formerly unsupported numeric destination with owner state unchanged; its exact Bool self operation still succeeded. The offline reader recomputes the numeric output words independently from the raw native source payload and checks destination/source identities and version behavior. All 68 actual candidate refusals preserved exact owner bytes, including 45 numeric writes followed by a later invalid command. Other view or overlap controls are not claimed.

The observation exceeded its requested scratch ceiling briefly because byte-identical compressed pre/post snapshots were initially stored twice while the monitor retained an older limit. The observed peak and the subsequent lossless deduplication are retained in the evidence. No further native work was run.

## Reproduction

Use the exact pinned #65 opt-in exact-self source and ELF as PREDECESSOR_SOURCE and PREDECESSOR_ELF. REPO supplies the unchanged build and protocol files, PKG is this directory, and HELPERS contains the pinned original typed_logical_not comparison helpers (the observer verifies all three). Use the locked CPU Torch interpreter TORCH_PYTHON and its actual install report PIP_INSTALL_REPORT.

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
