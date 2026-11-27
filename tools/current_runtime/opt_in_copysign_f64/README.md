# Opt-in native F64 tensor copysign

This successor of the scoped F32 projection adds same-dtype F64 to the
existing binary command 2 / operation 16. Original F32 remains on the
same native body. The new body transfers the F64 sign bit without float
arithmetic or conversion, then uses the typed native constructor and
8-byte native storage. Both arguments must be equal-shape, contiguous,
zero-offset CPU F64 and cannot track gradients. The differential covers
ordinary owned tensors at ranks 0–4. Positive-offset and non-contiguous
views, mixed dtypes, broadcast, scalars-as-a-schema, in-place, out,
autograd, and accelerators do not earn coverage. A rank-zero tensor is tested.

## Reproduce on the #84 branch stack

Use the same pinned compiler/linker, CPU-wheel oracle and pip install report
as the F32 projection. From the repository root:

```sh
R=tools/current_runtime
D=$R/opt_in_copysign_f64
P=/absolute/path/to/fresh-copysign64-scratch
python3 -I -B "$R/opt_in_copysign_f32/prepare.py" --output "$P/f32-source"
python3 -I -B "$R/build.py" --manifest "$P/f32-source/manifest.json" \
  --source-root "$P/f32-source" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/f32-build"
python3 -I -B "$D/prepare.py" --f32-source "$P/f32-source/runtime.hepta" --output "$P/f64-source"
python3 -I -B "$R/build.py" --manifest "$P/f64-source/manifest.json" \
  --source-root "$P/f64-source" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/f64-build1"
python3 -I -B "$D/prepare.py" --f32-source "$P/f32-source/runtime.hepta" --output "$P/pinned" \
  --expected-build "$P/f64-build1/build.json"
python3 -I -B "$R/build.py" --manifest "$P/pinned/manifest.json" \
  --source-root "$P/pinned" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/f64-build2"
"$HEPTA_TORCH_PYTHON" -I -B "$D/qualify.py" --repository "$PWD" \
  --parent "$P/f32-build/heptatorch" --candidate "$P/f64-build2/heptatorch" \
  --output "$P/qualification"
```

No canonical image, dispatch default, or acceptance credit changes.
[Execution evidence](evidence.json) binds the local native build and
differential receipts.

Two independent cold builds emitted the same 15 objects and linked ELF.
520 cases matched 9,776 PyTorch F64 output words bit-for-bit, including
signed zero, subnormals, infinities, qNaN/sNaN sign and payload, AVX2
vector boundaries, identical arguments and empty extents. The predecessor
refused all 520 F64 operator requests. The new native image refuses mixed
F32/F64 and i64/F64 in both argument positions and unequal shapes. The
existing F64 constructors (separate from the projected operator) refuse
the tested grad/non-contiguous/positive-offset factories on both images.
The native result has independent fresh storage and version, observes
the output after both input handles are released, and reports the same
physical and wire bytes. Previous F32 copysign and F32/F64 arithmetic
response bytes are replayed against the exact F32 predecessor.
