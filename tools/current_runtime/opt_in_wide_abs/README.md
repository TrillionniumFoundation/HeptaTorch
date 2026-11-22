# Opt-in current800 I64/F64 Tensor abs

A scoped CPU Tensor integration candidate for the current800 source accepted in
[#61](https://github.com/TrillionniumFoundation/HeptaTorch/pull/61) at
`f4f3bd5b44a14868e1af532e689b07c4ff197a89`. The scratch transform modifies one
HIA body, `wide_neg_tensor`, in one partition (`runtime800_06`). It uses the
same wrapping signed-word and sign-clear binary64 operations as the isolated
[#60](https://github.com/TrillionniumFoundation/HeptaTorch/pull/60) kernels.
The existing public Tensor transaction parser and the existing no-gradient
typed allocation, layout and element traversal paths are unchanged. The
candidate adds `(3, 7, tensor)` to its already dtype-gated I64/F64 path.
Unary operation 6 retains its existing behavior.

This does not enable abs in the canonical image. The opt-in scope is
out-of-place I64/F64 CPU tensors without gradients. In-place, `out=`, autograd,
other dtypes, direct frontend registration, and the 35-suite same-candidate
gate are not qualified by this package. The canonical gate and its historical
refusal inputs are not edited.

## Reproduce

Use #61's exact `tools/current_runtime/source` directory and the pinned ABI3
compiler and four-argument linker from the current-runtime build guide.
Choose fresh scratch directories for every build. From the repository root:

```sh
BASE=tools/current_runtime/source
PKG=tools/current_runtime/opt_in_wide_abs
python3 -I -B "$PKG/prepare.py" --base-source-root "$BASE" --output "$SCRATCH/abs-source"
python3 -I -B tools/current_runtime/build.py --manifest "$BASE/manifest.json" --source-root "$BASE" \
  --compiler "$COMPILER" --linker "$LINKER" --output "$SCRATCH/abs-baseline"
python3 -I -B tools/current_runtime/build.py --manifest "$SCRATCH/abs-source/manifest.json" \
  --source-root "$SCRATCH/abs-source" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$SCRATCH/abs-candidate"
"$TORCH_PYTHON" -B "$PKG/qualify.py" --baseline "$SCRATCH/abs-baseline/heptatorch" \
  --candidate "$SCRATCH/abs-candidate/heptatorch" \
  --protocol tests/integer_tensor_review/wire/protocol.py \
  --oracle-lock compatibility/pytorch-oracle-lock.json --install-report "$PIP_INSTALL_REPORT"
```

The source preparation admits only the exact baseline manifest, monolith and
15 partitions. Both builds pin their native ELF and every object; 14 object
identities were unchanged and both observed maximum native stack paths were
133,792 bytes. The exact candidate image must pass the independent pinned
PyTorch wheel differential. The observer compares raw int64/float64 words
(including NaN payload/signaling bits), shape, strides, storage offset, fresh
output Tensor/Storage/Version ownership, input immutability, unchanged negation
and same-frame alias handling. Two real process address-space refusals restore
the exact pre-request owner bytes, then succeed on the byte-identical request
and compare the resulting 65,536 values per dtype with actual PyTorch.

The [focused receipt](strict-evidence.json) retains the historical wide-neg
I64 and F64 unary-7 probe inputs as explicit successor observations: old exact
image status 7 remains, new exact image produces the independently checked
abs outputs. Admitting the candidate into the canonical gate would require
reviewed successor handling for those frozen readers and complete candidate
requalification, not edits to old receipts.
