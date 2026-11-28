# Opt-in spatial and first-order AD integration

This experiment combines exact #94 spatial routes, #89 F32 unsqueeze, and
#91 F32 square including its isolated exceptional-word adjoint. The projected
image retains explicit conv2d bias-post (125) and bias-first (126), plus
bias-free (22); neither bias opcode gains gradients or auto-dispatch.
The repository's default runtime, ATen catalog, and release gates are unchanged.

## Graph-free convolution scratch ownership

Both graph-free spatial paths allocated a temporary 128-byte geometry Storage
with a typed Hold. The predecessor persisted the geometry address on the output
but never released the scratch owner. Replaying four releases per opcode on
the exact #94 ELF leaves four extra live Storage/Hold pairs **per opcode**,
even when every public tensor is released and the ordinary invariant passes.
The graph-free route now clears the descriptor's raw geometry address after
the last reduction read, and releases the typed Hold. The existing autograd
route still records it for backward; the two are not interchangeable.

The candidate reproduces each of those 12 output buffers bit-for-bit with zero
non-page owners retained. The resident mixed qualification ends with zero live
non-page objects of all eight kinds. Five retained descriptor pages are
reported separately after verifying their actual Storage/Hold links. Three
invalid-output-extent requests and three work-cap requests leave the entire
published owner snapshot unchanged, exercising rollback after geometry
allocation, and after output allocation, respectively.

## Same-process differential

The 791-function image was cold-built twice with identical 15 object digests
and identical executable digest. A pinned single-thread PyTorch 2.15.0+cpu
AVX2 reference and each native build produced identical qualification JSON.
One persistent native process ran 3,534 tested data commands, including:

- 14 cases across two NCHW shapes and seven bias/cancellation fixtures, where
  both explicit accumulators are checked. Torch profiler traces verify the
  slow/oneDNN comparison. The tested compatible route composes directly with
  square and average pooling in the native process after releasing all inputs.
- 20 leaf unsqueeze→square AD cases across positive and negative axes and
  ranks 0–3. The input and view share physical Storage and Version, square has
  independent owners, and the public view is dropped before two native seeded
  backward calls for every case. Accumulation matches real Torch bitwise.
- 257 scalar/packed exceptional square AD fixtures, 512 words each in
  forward and backward, plus a bias-free conv2d backward with both operands'
  gradient buffers checked independently.

The scope remains bounded CPU F32 and first-order AD. No broader dtype/device,
general convolution backend selector, fused biased AD, or PyTorch replacement
credit is claimed.

## Reproduce

First produce the exact bias-first predecessor ELF and projected
`bias-first.hepta` using the adjacent
[bias-first instructions](../opt_in_conv2d_bias_first/README.md).
Then from the root of this checkout, using the same scratch `$w`,
toolchain and pinned Torch environment:

```sh
c=tools/current_runtime
q=$c/opt_in_spatial_ad
python3 -B "$q/project.py" --parent "$w/bias-first.hepta" --output "$w/spatial-ad.hepta"
python3 -B "$c/partition.py" --runtime "$w/spatial-ad.hepta" \
    --name narrow-spatial-f32-ad --prefix spatialad --output "$w/spatial-ad-source"
for n in a b; do
    python3 -B "$c/build.py" --manifest "$w/spatial-ad-source/manifest.json" \
        --source-root "$w/spatial-ad-source" \
        --compiler "$HEPTA_STACK_BUILD/stack-module-compiler" \
        --linker "$HEPTA_STACK_BUILD/stack-linker-session" --output "$w/spatial-ad-$n"
    "$HEPTA_TORCH_PYTHON" -s -E -B "$q/qualify.py" \
        --parent "$w/bias-first/heptatorch" --binary "$w/spatial-ad-$n/heptatorch" \
        --source "$w/spatial-ad-$n/runtime.hepta" --wheel "$PINNED_TORCH_WHEEL" \
        --install "$PINNED_TORCH_INSTALL_REPORT" --output "$w/spatial-ad-qual-$n"
done
cmp "$w/spatial-ad-a/heptatorch" "$w/spatial-ad-b/heptatorch"
cmp "$w/spatial-ad-qual-a/qualification.json" "$w/spatial-ad-qual-b/qualification.json"
```

The release receipts are [build-receipt.json](build-receipt.json) and
[qualification.json](qualification.json).
