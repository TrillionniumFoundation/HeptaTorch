# Opt-in native F64 copysign broadcasting

This adds right-aligned Tensor-to-Tensor broadcasting to the existing native
F64 operation 2/16 in [#90](https://github.com/TrillionniumFoundation/HeptaTorch/pull/90).
The native F64 body uses the retained broadcast geometry and logical index
helpers, copies the raw sign bit, and allocates one independent contiguous
output. It does not materialize expanded intermediates. It covers rank 0–4,
same-dtype, contiguous, zero-offset, no-grad F64 CPU arguments. Rank-zero
operands are Tensor operands, not the scalar overload. F32, mixed dtypes,
strided views, in-place/out, other devices, and derivatives are not extended.
The canonical image and acceptance claim are unchanged.

## Reproduce

Use an exact checkout of this stack, the pinned runtime tools from #90,
the locked native CPU Torch wheel, and fresh scratch output paths:

```sh
R=tools/current_runtime
D=$R/opt_in_copysign_f64_broadcast
P=/absolute/path/to/fresh-copysign-broadcast-scratch
cold() {
  python3 -I -B "$R/build.py" --manifest "$1/manifest.json" --source-root "$1" \
    --compiler "$COMPILER" --linker "$LINKER" --output "$2"
}
python3 -I -B "$R/opt_in_copysign_f32/prepare.py" --output "$P/f32-source"
python3 -I -B "$R/opt_in_copysign_f64/prepare.py" --f32-source "$P/f32-source/runtime.hepta" --output "$P/parent-source"
cold "$P/parent-source" "$P/parent-build"
python3 -I -B "$D/prepare.py" --parent-source "$P/parent-source/runtime.hepta" --output "$P/candidate-source"
cold "$P/candidate-source" "$P/candidate-build"
python3 -I -B "$D/prepare.py" --parent-source "$P/parent-source/runtime.hepta" \
  --output "$P/pinned-source" --expected-build "$P/candidate-build/build.json"
cold "$P/pinned-source" "$P/pinned-build"
for image in candidate pinned; do
  "$HEPTA_TORCH_PYTHON" -I -B "$D/qualify.py" --repository "$PWD" \
    --parent "$P/parent-build/heptatorch" --candidate "$P/$image-build/heptatorch" \
    --output "$P/$image-qualification"
done
cmp "$P/candidate-qualification/qualification.json" "$P/pinned-qualification/qualification.json"
```

The tests use eager PyTorch for the exact F64 output words and shape/stride.
They also check fresh physical Storage/Version, readback after both inputs
are released, and byte-identical ownership after predecessor unequal-shape
refusals and the new explicit incompatible-shape refusals. Incompatible
broadcast geometry reports shape error 2 on the new image, rather than
the predecessor's scoped unsupported-layout status 7.
