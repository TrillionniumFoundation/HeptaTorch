# Physical qualification: opt-in CPU Bool logical_not(out=)

This overlay derives from HeptaTorch [#63](https://github.com/TrillionniumFoundation/HeptaTorch/pull/63)'s exact joint Tensor abs and out-of-place logical_not source. It leaves the canonical runtime, owner transport and opcode 127 unchanged. It does not combine with the #64 F32 overlays. The isolated source has 803 functions, but is not the separate Current803 lineage.

## Scope and observed results

The fixed HTORCH03 record is (128, source, destination, 0,0,0,0,0). On success it returns the existing public destination handle, no payload. Both tensors must be ordinary CPU, no-edge/no-grad, zero-offset, same-shape and contiguous, with distinct Tensor, Storage and Version identities. Rank is at most 4 and count at most 262,144. Destination dtype is Bool; accepted source tags are 0/1/2/4/5/6 (F32, Bool, I64, F64, F16, BF16). Tag 3 is private and excluded. The destination Storage and Version IDs and source bytes/metadata do not change. Destination Version increments once per call, including empty and otherwise identical writes. No casting, resizing, alias acceptance, strided views, autograd, inference-mode, non-CPU or production claim.

Two cold builds produced identical ELF 568946d257cd81d00f4fb8b5a6fdd966abf1bb8de6333b59b098be08a1afc62d. The official locked PyTorch comparison passed 13 cases with 144,007 exact Boolean elements. Its test verifies one and two version increments, unchanged Torch destination pointer and returned object identity, and raw native source invariance, destination metadata and stable returned handle, including request-local result reuse. The offline reader checked 170 candidate frames. All 18 executed rejection controls retained byte-identical original owner state: old 127's six reserved words, new 128's five reserved words, exact Tensor alias, wrong output dtype, equal-count wrong shape, invalid source, invalid destination, forward reference, and a valid write followed by a refused record. The old exact physical image refused validly formatted 128 with status 4 and preserved its full owner; it refused each modified 127 envelope with status 7 and continued passing ordinary out-of-place 127. Controls for distinct stored aliases, view variants, stale generations, wrong-kind tokens, version exhaustion and reply capacity are planned, not claimed. The two initial failed drafts remain identified in the evidence.

## Reproduce

Set PKG to this directory, REPO to a checkout containing the existing fixed compiler build and tests, JOINT_SOURCE to the source derived by #63's original opt_in_dual_unary prepare_dual.py instructions, and JOINT_ELF to that original built joint image. Use the already locked official Torch interpreter and its saved pip installation report. COMPILER and LINKER are the existing tools pinned by REPO's build.py. HELPERS is a directory with the exact prior typed_logical_not copies of test_runtime_candidate.py, test_component.py and qualify_runtime_torch.py; the observer checks their bytes before importing. The original protocol and its sibling owner_client.py are also independently hash-checked by that helper.

```sh
set -eu
STAGE="$(mktemp -d)"
BUILD="$REPO/tools/current_runtime/build.py"
PROTOCOL="$REPO/tests/integer_tensor_review/wire/protocol.py"
ORACLE_LOCK="$REPO/compatibility/pytorch-oracle-lock.json"
python3 -I -B "$PKG/prepare.py" "$JOINT_SOURCE" "$STAGE/source"
cp "$PKG/pinned-manifest.json" "$STAGE/source/manifest.json"
for b in cold-a cold-b; do
  python3 -I -B "$BUILD"     --manifest "$STAGE/source/manifest.json" --source-root "$STAGE/source"     --compiler "$COMPILER" --linker "$LINKER" --output "$STAGE/$b"
done
cmp "$STAGE/cold-a/heptatorch" "$STAGE/cold-b/heptatorch"
python3 -I -B "$PKG/observe.py" baseline   --binary "$JOINT_ELF" --output "$STAGE/original"   --protocol "$PROTOCOL" --helper-dir "$HELPERS"
"$TORCH_PYTHON" -I -B -W error "$PKG/observe.py" candidate   --binary "$STAGE/cold-b/heptatorch" --output "$STAGE/candidate"   --protocol "$PROTOCOL" --helper-dir "$HELPERS"   --oracle-lock "$ORACLE_LOCK" --install-report "$PIP_INSTALL_REPORT"
python3 -I -B "$PKG/readback.py" "$STAGE/original" "$STAGE/candidate"
```

Both executables are exact-pinned by the observer. Independent readback needs only the output folders; it does not launch either executable or import observer/protocol helpers. Evidence contains the raw request/response stream hashes and the independently checked counts. Raw captures remained local and temporary; compact evidence alone is not a replacement for them or a claim about the 35 mandatory suites.
