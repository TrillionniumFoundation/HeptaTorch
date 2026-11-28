# Opt-in mixed copysign rank-zero promotion

This extends the native #104 mixed F32/F64 implementation to contiguous,
zero-offset, no-grad CPU Tensor inputs when one or both ranks are zero.
The only existing native body changed is the opt-in mixed implementation.
A dimensioned F32 Tensor with a rank-zero F64 Tensor produces F32;
otherwise the mixed result is F64. The selected dtype is the same for
both operand orders, with ordinary right-aligned broadcasting.

When the F64 magnitude must narrow to F32, the retained software
conversion performs round-to-nearest-even, while the opt-in body
preserves the high payload bits of quieted NaNs. Sign bits are taken
directly at their input width. These are rank-zero **Tensor** inputs; the
distinct scalar overload, out/in-place, other dtypes, arbitrary views,
devices, derivatives and default-runtime cutover remain out of scope.

## Reproduce

Start from the exact #104 source projection and image, the pinned native
build tools and the locked CPU Torch wheel.

```sh
R=tools/current_runtime
D=$R/opt_in_copysign_mixed_rankzero
P=/absolute/path/to/fresh-mixed-rankzero-scratch
cold() {
  python3 -I -B "$R/build.py" --manifest "$1/manifest.json" --source-root "$1" \
    --compiler "$COMPILER" --linker "$LINKER" --output "$2"
}
python3 -I -B "$D/prepare.py" --parent-source "$MIXED_SOURCE" --output "$P/source"
cold "$P/source" "$P/build"
python3 -I -B "$D/prepare.py" --parent-source "$MIXED_SOURCE" \
  --output "$P/pinned-source" --expected-build "$P/build/build.json"
cold "$P/pinned-source" "$P/pinned-build"
for img in build pinned-build; do
  "$HEPTA_TORCH_PYTHON" -I -B "$D/qualify.py" --repository "$PWD" \
    --parent "$MIXED_ELF" --candidate "$P/$img/heptatorch" --output "$P/qualification-$img"
done
cmp "$P/qualification-build/qualification.json" "$P/qualification-pinned-build/qualification.json"
```

The new differential includes directed midpoint, subnormal and overflow
boundaries, NaN payloads, signed zeros, vector boundaries and empty
outputs. It reads physical Storage/Version, requires unchanged inputs
and reads each result again after release. It separately replays exact
predecessor wire frames for dimensioned mixed and same-width controls.
