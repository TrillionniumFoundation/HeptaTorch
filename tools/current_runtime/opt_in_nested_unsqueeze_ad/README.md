# Opt-in two-level F32 unsqueeze AD

The #102 image rejects a second native differentiable unsqueeze (opcode 129)
even when the first view inserts only a singleton axis. This experiment
extends native admission to one existing contiguous nonempty zero-offset
shape-view edge (kind 45, ordinary view kind 1) whose graph parent is an
ordinary dense zero-offset leaf. Runtime checks require one extra rank,
equal element count, and identical physical Storage and Version. Results
are capped at rank four, so the qualified original leaf ranks are zero
through two. The existing native structural adjoint propagates through both
edges; no gradient values are fabricated by a host wrapper.

Two cold builds match all 15 native object digests and the same ELF.
Two independent runs against pinned PyTorch 2.15.0+cpu AVX2 compare the
following bit-exact data:

- 272 double-unsqueeze cases across eight scalar/vector/matrix shapes and all
  positive and negative axis pairs. Both native edges point to the proper
  descriptor, and all three tensors share the same Storage and Version.
- 1,760 output F32 words and 3,520 accumulated leaf gradient words over 544
  reverse calls. In each case the public intermediate handle is released
  before backward, and the input remains unchanged.
- The exact #102 image rejects a representative nested request for every
  input shape with unchanged owner state. Five candidate controls preserve
  full owner state: third reverse edge, invalid axis, unrelated kind-16
  reshape, unrelated square non-leaf, and empty differentiable leaf.
- A full #102 same-process integration replay matches after normalizing only
  the new ELF and source digests; 3,534 commands retain the convolution
  scratch lifetime fix, bias-order, pooling, and first-order AD behavior.

The resident nested test finishes with no live non-page owners. The seven
remaining descriptor pages have separately verified Storage/Hold identities.
This is not arbitrary nested view AD, higher-order differentiation, device
or dtype expansion, a change to the default image, or full PyTorch coverage.

## Reproduce

Use the [#102 spatial/AD recipe](../opt_in_spatial_ad/README.md) first. It
produces the exact parent source and image, and the #94 spatial predecessor
ELF. Use the same scratch `$w`, HIA toolchain, and pinned Torch:

```sh
c=tools/current_runtime
q=$c/opt_in_nested_unsqueeze_ad
python3 -B "$q/project.py" --parent "$w/spatial-ad.hepta" --output "$w/nested.hepta"
python3 -B "$c/partition.py" --runtime "$w/nested.hepta" \
  --name narrow-spatial-f32-nested-unsqueeze-ad --prefix nestedunsqueeze --output "$w/nested-source"
for n in a b; do
  python3 -B "$c/build.py" --manifest "$w/nested-source/manifest.json" \
    --source-root "$w/nested-source" --compiler "$HEPTA_STACK_BUILD/stack-module-compiler" \
    --linker "$HEPTA_STACK_BUILD/stack-linker-session" --output "$w/nested-$n"
  "$HEPTA_TORCH_PYTHON" -s -E -B "$q/qualify.py" \
    --parent "$w/spatial-ad-a/heptatorch" --binary "$w/nested-$n/heptatorch" \
    --source "$w/nested-$n/runtime.hepta" --wheel "$PINNED_TORCH_WHEEL" \
    --install "$PINNED_TORCH_INSTALL_REPORT" --output "$w/nested-qual-$n"
  "$HEPTA_TORCH_PYTHON" -s -E -B "$q/replay.py" \
    --spatial-parent "$w/bias-first/heptatorch" --binary "$w/nested-$n/heptatorch" \
    --source "$w/nested-$n/runtime.hepta" --wheel "$PINNED_TORCH_WHEEL" \
    --install "$PINNED_TORCH_INSTALL_REPORT" --output "$w/nested-replay-$n"
done
cmp "$w/nested-a/heptatorch" "$w/nested-b/heptatorch"
cmp "$w/nested-qual-a/qualification.json" "$w/nested-qual-b/qualification.json"
cmp "$w/nested-replay-a/qualification.json" "$w/nested-replay-b/qualification.json"
```

See [qualification summary](qualification-summary.json) and
[cold-build receipt](build-receipt.json) for the complete generated report
digests and reproduced object pins.
