# Opt-in F32 unsqueeze first-order reverse AD

This stacked experiment admits positive-extent, owned, contiguous graph leaf
F32 CPU inputs for native command 129, input rank 0–3. It preserves the exact
graph-free route and refuses gradients on nested views, empty inputs, and
other out-of-scope tensors. It uses the existing native first-order structural
view edge (kind 45), with the original parent descriptor and shared physical
Storage and Version. With a contiguous leaf and exactly one inserted
singleton, the adjoint is the same logical gradient stream with the singleton
axis removed. No host computation participates in the operation or adjoint.

The locked PyTorch 2.15.0+cpu reference matches 40 shape/axis cases, 80 finite
dyadic seeded accumulation rounds, 1,052 element comparisons each in forward
and reverse. Inputs are unchanged. Physical owner snapshots confirm the
alias and exact graph parent, and final release restores live object counts.
The #86 parent refuses all 40 requires-grad input requests. Three candidate
scope controls leave published owner state byte-exact; 348 graph-free frames
match the #86 parent byte-for-byte. No general higher-order, view-of-view,
saved-tensor, or wider-dtype/device claim is made.

## Reproduce

First build and pin the graph-free parent using the adjacent
[unsqueeze ABI experiment](../opt_in_unsqueeze_f32/README.md). From the
repository root, use the same environment and exact toolchain:

```sh
p=tools/current_runtime/opt_in_unsqueeze_f32_ad
base=tools/current_runtime/opt_in_unsqueeze_f32
w=$(mktemp -d)
python3 -B "$p/prepare.py" --parent "$UNSQUEEZE_PARENT_SOURCE" --output "$w/source"
for build in a b; do
  python3 -B tools/current_runtime/build.py \
    --manifest "$w/source/manifest.json" --source-root "$w/source" \
    --compiler "$HEPTA_STACK_BUILD/stack-module-compiler" \
    --linker "$HEPTA_STACK_BUILD/stack-linker-session" --output "$w/$build"
  "$HEPTA_TORCH_PYTHON" -s -E -B "$p/qualify.py" --repository . \
    --parent "$UNSQUEEZE_PARENT_ELF" --candidate "$w/$build/heptatorch" \
    --candidate-source "$w/$build/runtime.hepta" \
    --graph-free-qualifier "$base/qualify.py" --output "$w/q-$build"
done
cmp "$w/a/heptatorch" "$w/b/heptatorch"
cmp "$w/q-a/qualification.json" "$w/q-b/qualification.json"
```

See the [scoped evidence receipt](evidence.json). The default runtime and
complete contract acceptance remain untouched.
