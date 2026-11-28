# Opt-in F32 and F64 copysign Tensor broadcasting

On the exact [#96](https://github.com/TrillionniumFoundation/HeptaTorch/pull/96)
source, extend opcode 2/16 to admit ordinary contiguous, zero-offset,
same-dtype F32 Tensor operands with compatible right-aligned shapes at
ranks 0–4. The F32 native body reuses retained shape/index helpers, writes
the output once without expanding either input, and never interprets the
raw words as arithmetic. F64 retains its exact source body.

Tests compare both widths against the pinned real CPU PyTorch build.
No default behavior or canonical coverage changes, and no scalar
overload, mixed dtypes, in-place/out, arbitrary views, other devices,
or derivative behavior is claimed.

## Reproduce

Build the exact F64-only source and image as in #96 and use its exact
same-shape #90 F64 predecessor image for the retained F64 replay.
Then produce and test two fresh cold images:

```sh
R=tools/current_runtime
D=$R/opt_in_copysign_f32_broadcast
P=/absolute/path/to/fresh-f32-f64-broadcast-scratch
cold() {
  python3 -I -B "$R/build.py" --manifest "$1/manifest.json" --source-root "$1" \
    --compiler "$COMPILER" --linker "$LINKER" --output "$2"
}
python3 -I -B "$D/prepare.py" --parent-source "$F64_BROADCAST_SOURCE" --output "$P/source"
cold "$P/source" "$P/build"
python3 -I -B "$D/prepare.py" --parent-source "$F64_BROADCAST_SOURCE" \
  --output "$P/pinned-source" --expected-build "$P/build/build.json"
cold "$P/pinned-source" "$P/pinned-build"
for img in build pinned-build; do
  "$HEPTA_TORCH_PYTHON" -I -B "$D/qualify.py" --repository "$PWD" \
    --parent "$F64_BROADCAST_ELF" --candidate "$P/$img/heptatorch" --output "$P/qualification-$img"
done
cmp "$P/qualification-build/qualification.json" "$P/qualification-pinned-build/qualification.json"
"$HEPTA_TORCH_PYTHON" -I -B "$D/replay_f64.py" --repository "$PWD" \
  --parent "$F64_SAME_SHAPE_ELF" --candidate "$P/pinned-build/heptatorch" --output "$P/qualification-f64"
```

The frozen F64 replay modifies only its candidate executable identity in
memory. The F32 suite checks operations against eager PyTorch, including
unchanged equal-shape wire control responses; shape/dtype/grad negative
tests; independent physical Storage/Version; and result readback after
input release. A separate mixed live session creates both widths at once,
executes both broadcasts in a common frame, releases all operands, and
reads back both results.
