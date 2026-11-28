# Compose native Conv2d FMA, spatial/autograd, and scratch ownership

One opt-in image composes the exact #115 explicit CPU FMA implementation
with the nine changed and six added HIA functions of exact #102. The
projection verifies exact source digests of both images and their common
#94 source, requires mutually unchanged function-level inputs, and
refuses unexpected differences. In particular it carries over the
geometry Hold release from both `conv2d` and `conv_first_geometry`, not
only the spatial/autograd functions. The new opcode 128 remains explicit;
opcode 127 still refuses oneDNN-enabled N=2/3. Opcode 129 retains the
qualified unsqueeze slice from #102. The default runtime is unchanged.

The verifier replays the complete pinned #102 receipt and the matching
#115 oneDNN receipt on the composed image, normalizing only their verified
source and ELF identity. It also interleaves native op125/126/127/128 →
square → average pool in one live process, verifies bit-exact Torch
outputs and distinct physical Storage/Version owners, drops the inputs
before reading the dependent outputs, and checks real typed owner slots
after releasing each pipeline. Only backed descriptor-page holds are
excluded from the live-owner counts. The exact #102 replay additionally
covers bias-free differentiation, Square/Unsqueeze AD, exceptional
gradients and bad geometry/work-cap controls. Neither full backend nor
full PyTorch qualification follows.

## Reproduce

Use the [exact #115 recipe](../opt_in_conv2d_cpu_fma/README.md) and the
[exact #102 sibling](https://github.com/TrillionniumFoundation/HeptaTorch/pull/102)
checked out at `8af50d2f9f614ca21036a20d63727ba0a124c605`, including its
`tools` and `tests` directories. Build the #102 source with its recipe
and retain its generated `$SPATIAL_SOURCE`. The shared source and binary
are #94 (respectively `$w/bias-first.hepta` and `$w/bias-first/heptatorch`).

```sh
c=tools/current_runtime
q=$c/opt_in_conv2d_spatial_fma
python3 -B "$q/project.py" --parent "$w/fma.hepta" --spatial "$SPATIAL_SOURCE" --common "$w/bias-first.hepta" --output "$w/combined.hepta"
python3 -B "$c/partition.py" --runtime "$w/combined.hepta" --name narrow-conv-spatial-ad-with-fma --prefix spatialfma --output "$w/combined-source"
python3 -B "$c/build.py" --manifest "$w/combined-source/manifest.json" --source-root "$w/combined-source" --compiler "$HEPTA_STACK_BUILD/stack-module-compiler" --linker "$HEPTA_STACK_BUILD/stack-linker-session" --output "$w/combined"
replay() {
 "$HEPTA_TORCH_PYTHON" -s -E -B "$q/qualify.py" --binary "$w/combined/heptatorch" --source "$w/combined.hepta" --spatial-base "$w/bias-first/heptatorch" --slow-base "$w/slow/heptatorch" --baseline "$PINNED_MAIN_BINARY" --spatial-root "$SPATIAL_CHECKOUT" --wheel "$PINNED_TORCH_WHEEL" --install "$PINNED_TORCH_INSTALL_REPORT" --mode "$1"
}
(unset ONEDNN_MAX_CPU_ISA DNNL_MAX_CPU_ISA; replay 0)
(unset DNNL_MAX_CPU_ISA; export ONEDNN_MAX_CPU_ISA=AVX2; replay 1)
```
