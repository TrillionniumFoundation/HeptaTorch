# Opt-in mixed F32/F64 copysign Tensor broadcasting

This adds both operand orders for ordinary contiguous, zero-offset,
no-grad CPU tensors, one F32 and one F64, both with rank 1–4.
The dimensioned tensors promote to one fresh F64 result, with
right-aligned broadcasting. The retained native widen operation converts
only the F32 magnitude; selecting a F32 sign extracts the raw sign bit
directly. The original F32 and F64 native bodies are unchanged.
Mixed rank-zero requests remain refused, since PyTorch's result-type
precedence differs between dimensioned tensors and zero-dimensional
operands. Scalar overloads, arbitrary views, grad, out/in-place, other
devices and canonical/default change are outside scope.

## Reproduce

Use the exact #99 combined same-dtype source and image with the upstream
locked CPU oracle, and build from the repository root:

```sh
R=tools/current_runtime
D=$R/opt_in_copysign_mixed_broadcast
P=/absolute/path/to/fresh-mixed-copysign-scratch
cold() {
  python3 -I -B "$R/build.py" --manifest "$1/manifest.json" --source-root "$1" \
    --compiler "$COMPILER" --linker "$LINKER" --output "$2"
}
python3 -I -B "$D/prepare.py" --parent-source "$F32_F64_SOURCE" --output "$P/source"
cold "$P/source" "$P/build"
python3 -I -B "$D/prepare.py" --parent-source "$F32_F64_SOURCE" \
  --output "$P/pinned-source" --expected-build "$P/build/build.json"
cold "$P/pinned-source" "$P/pinned-build"
for img in build pinned-build; do
  "$HEPTA_TORCH_PYTHON" -I -B "$D/qualify.py" --repository "$PWD" \
    --parent "$F32_F64_ELF" --candidate "$P/$img/heptatorch" --output "$P/qualification-$img"
done
cmp "$P/qualification-build/qualification.json" "$P/qualification-pinned-build/qualification.json"
```

The differential checks actual PyTorch output dtype, shape, strides, and
raw words, including NaN payloads and signed zeros. It verifies independent
physical output ownership, ensures both input tensors are unchanged, and
reads the result again after releasing both inputs. Predecessor mixed-dtype
and candidate incompatible-shape, rank-zero, and grad refusals must leave
the owner snapshot byte-identical.
