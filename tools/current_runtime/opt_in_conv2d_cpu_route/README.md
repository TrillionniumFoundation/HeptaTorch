# Scoped native CPU conv2d accumulator-order selection

One new opt-in HIA wire command, **127**:
`(127, input, weight, bias, stride_pair, padding_pair, dilation_pair, mkldnn_enabled)`.
It chooses the accumulator order on the admitted geometry and does **not**
accept a caller-supplied backend enum or accumulator mode. Its last argument
is the pinned reference's Boolean oneDNN preference, not a groups field.
The scope is owned contiguous no-grad F32 with input `[N,2,4,4]`, weight
`[2,2,3,3]`, bias `[2]`, N=1 or 2, stride and dilation (1,1), no padding and
implicit groups=1. On this scope and the pinned AVX2 single-threaded reference:

- N=1: native bias-first, whether oneDNN is enabled or not.
- N=2: native post-sum if oneDNN is enabled, native bias-first otherwise.

The HIA command reads tensor shapes from validated native owners, checks the
wire parameters, and dispatches directly to the existing HIA reductions.
It does not load PyTorch or oneDNN at runtime. The two explicit opcodes (125
and 126) remain independently callable with their original semantics.
Out-of-scope metadata and unsupported settings are rejected, not silently
sent to one of the kernels. Neither the canonical source nor default
runtime manifest is changed. Broader numerical behavior of each chosen
kernel is explicitly unqualified, including arbitrary FP32 inputs, SIMD/FMA
and other layouts; this is not general backend selection, full Conv2d
parity, or a convenience wrapper claiming to replace oneDNN.

The rule is consistent with the pinned
[ATen Convolution.cpp](https://github.com/pytorch/pytorch/blob/31a77fc9/aten/src/ATen/native/Convolution.cpp)
conditions for these two shapes only. Actual `torch.nn.functional.conv2d`
profiling observes `aten::_slow_conv2d_forward` or
`aten::mkldnn_convolution` per row. Two byte-identical independent builds
and differential reports cover 40 observations and 480 bit-exact words,
including 24 observations for which using the unselected order changes the
result. Another 40 explicit route comparisons agree with the prior ELF
(including frame hashes). Seventeen negative controls verify unchanged
input owners and cleanup, and the exact #94 ELF refuses the new opcode.

## Reproduce

Rebuild the exact #94 parent using its [recipe](../opt_in_conv2d_bias_first/README.md),
retaining both `$w/bias-first.hepta` and `$w/bias-first/heptatorch`. Then:

```sh
c=tools/current_runtime
q=$c/opt_in_conv2d_cpu_route
python3 -B "$q/project.py" --parent "$w/bias-first.hepta" --output "$w/route.hepta"
python3 -B "$c/partition.py" --runtime "$w/route.hepta" --name narrow-conv-bias-cpu-route --prefix biasroute --output "$w/route-source"
python3 -B "$c/build.py" --manifest "$w/route-source/manifest.json" --source-root "$w/route-source" --compiler "$HEPTA_STACK_BUILD/stack-module-compiler" --linker "$HEPTA_STACK_BUILD/stack-linker-session" --output "$w/route"
"$HEPTA_TORCH_PYTHON" -s -E -B "$q/qualify.py" --binary "$w/route/heptatorch" --parent "$w/bias-first/heptatorch" --wheel "$PINNED_TORCH_WHEEL" --install "$PINNED_TORCH_INSTALL_REPORT"
```
