# Opt-in native HIA CPU F32 tensor copysign

This separate native runtime implements the equal-shape,
contiguous, zero-offset, no-grad F32 CPU slice of the original
`aten::copysign.Tensor` declaration. It adds binary command 2, operation
16, with left tensor magnitude and right tensor sign. The command is not
enabled in the canonical runtime. Unary command 3 has a different
operation namespace. The kernel combines raw magnitude/sign words in HIA
and writes into fresh Tensor, Storage, and Version records.
Python only prepares source, orchestrates verification, and queries the
independent pinned PyTorch CPU oracle.

The tested domain includes ranks 0–4, empty tensors, vector boundaries and
remainder lanes, identical operand handles, signed zeros, subnormals,
infinities, qNaNs and sNaNs with varied sign and payload. Actual refusal
controls check mixed dtype, broadcast-required shapes, non-contiguous and
positive-offset views, and grad input on either operand. Contiguous views
with zero offset are not qualified by this evidence; the guard verifies
layout and offset, not view provenance. Scalar, in-place, out, broadcast,
other dtypes, autograd and accelerator variants are outside this slice.

## Reproduce

Use commit `e9444902df9878060ab389a70ee957c1b7f4a887`'s canonical runtime
source, the original pinned stack-budget compiler/linker, and the locked
CPU wheel and installation report. From the repository root:

```sh
D=tools/current_runtime/opt_in_copysign_f32
R=tools/current_runtime
P=/absolute/path/to/fresh-copysign-scratch
python3 -I -B "$D/prepare.py" --output "$P/source"
python3 -I -B "$R/build.py" --manifest "$P/source/manifest.json" \
  --source-root "$P/source" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/build1"
python3 -I -B "$D/prepare.py" --output "$P/pinned" \
  --expected-build "$P/build1/build.json"
python3 -I -B "$R/build.py" --manifest "$P/pinned/manifest.json" \
  --source-root "$P/pinned" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/build2"
"$HEPTA_TORCH_PYTHON" -I -B "$D/qualify.py" --repository "$PWD" \
  --parent "$EXACT_MAIN_ELF" --candidate "$P/build2/heptatorch" \
  --output "$P/qualification"
```

Two cold builds emitted the same 15 objects and native ELF. The gated
differential tests all 520 cases against actual PyTorch 2.15.0+cpu
(AVX2, one thread), 9,776 output words byte-for-byte, and attests the actual
native Tensor metadata and independently decoded physical Storage bytes.
Every positive case checks input storage immutability and reads the result
again after both input handles are released. The canonical parent refused
each newly admitted input; 12 response frames for previous F32 add/sub/mul/div
are unchanged. No complete schema, subsystem, default, or coverage credit
is claimed. See [execution evidence](evidence.json).
