# Explicit native FMA CPU Conv2d

This opt-in HIA runtime adds command **128**:
`(128, input, weight, bias, stride_pair, padding_pair, dilation_pair, order)`.
For the admitted N=2/3 input `[N,2,4,4]`, weight `[2,2,3,3]`, bias `[2]`,
stride and dilation `(1,1)`, no padding, implicit groups=1, contiguous
no-grad F32, each point traverses kernel row, kernel column, input channel
and calls the existing native F32 `fmaf` exactly once per tap. The admitted
operand magnitudes are zero or in [2^-24, 8]. The selector refuses non-finite,
subnormal and other out-of-domain operands, and does not change any
automatic path. The output is allocated fresh; the new kernel creates no
convolution geometry Hold. Legacy op22/125/126/127 paths in this branch
do not inherit #102's scratch-reclamation repair.

Explicit mode 0 starts from +0 then adds bias once; it matched the pinned
oneDNN `jit:avx512_core` implementation on the Dot host.
Explicit mode 1 starts from bias; it matched `jit:avx2` with
`ONEDNN_MAX_CPU_ISA=AVX2` on the same host. In both environments, the
actual eager profiler recorded `aten::mkldnn_convolution`; separate fresh
processes inspected the selected oneDNN v3.13.3 primitive via verbose
metadata for both batch sizes. The runtime does not infer the mode from
CPU ISA or settings; opcode 127 still rejects the N=2/3 oneDNN-enabled
routes. This is not a general backend policy, generic FMA parity, or a
default-runtime change.

Two independent 15-object builds and complete reruns produce identical
receipts. Each mode matches 90 cases and 1,800 words, including dense
non-dyadic inputs, finite domain edges, signed zero and cancellation.
The alternate mode differs in 82/90 cases per environment. Fourteen
negative controls per mode leave existing owners intact. The canonical
image and exact #108 reject 128, and an independent four-case replay on
the exact #102 spatial ELF also rejects it (129 remains its unsqueeze op).

## Reproduce

Build [exact #108](../opt_in_conv2d_cpu_route_slow_only/README.md) first.
Keep the pinned canonical runtime as `$PINNED_MAIN_BINARY`. Then:

```sh
c=tools/current_runtime
q=$c/opt_in_conv2d_cpu_fma
python3 -B "$q/project.py" --parent "$w/slow.hepta" --output "$w/fma.hepta"
python3 -B "$c/partition.py" --runtime "$w/fma.hepta" --name narrow-conv-bias-cpu-fma --prefix bfma --output "$w/fma-source"
python3 -B "$c/build.py" --manifest "$w/fma-source/manifest.json" --source-root "$w/fma-source" --compiler "$HEPTA_STACK_BUILD/stack-module-compiler" --linker "$HEPTA_STACK_BUILD/stack-linker-session" --output "$w/fma"
replay() {
  "$HEPTA_TORCH_PYTHON" -s -E -B "$q/qualify.py" --binary "$w/fma/heptatorch" --parent "$w/slow/heptatorch" --baseline "$PINNED_MAIN_BINARY" --wheel "$PINNED_TORCH_WHEEL" --install "$PINNED_TORCH_INSTALL_REPORT" --mode "$1"
}
(unset ONEDNN_MAX_CPU_ISA DNNL_MAX_CPU_ISA; replay 0)
(unset DNNL_MAX_CPU_ISA; export ONEDNN_MAX_CPU_ISA=AVX2; replay 1)
```
