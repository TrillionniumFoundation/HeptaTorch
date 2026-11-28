# Fail closed for automatic oneDNN Conv2d selection

Automatic native command 127 now retains **only** the measured slow eager
CPU lane: F32 no-grad input `[N,2,4,4]` and weights `[2,2,3,3]`, bias
`[2]`, implicit groups=1, stride and dilation (1,1), no padding, with
N=1 for either oneDNN preference, or N=2/3 with the oneDNN preference off.
N=2/3 with oneDNN enabled returns status 10 before reaching a convolution
kernel. It does not silently substitute post-sum, slow eager, or any other
kernel. The explicit experiment opcodes 125 and 126 remain distinct and
unchanged. Neither the default runtime nor resource-lifetime behavior changes.

Two independent cold 15-object builds and two full differential runs agree.
On the actual locked Torch profiler and the admitted geometric settings,
156 observations (2,184 words) agree bit-for-bit. All 78 oneDNN observations
refuse; 32 dense cases on each N2 and N3 show the old explicit post-sum
kernel would differ from the oracle. Across 234 explicit opcode
comparisons, 125 and 126 reproduce earlier output and ownership frames.
Additional negative controls cover gradient-bearing and out-of-scope batch
inputs. This is finite scoped evidence, not arbitrary FP32 or a general
dispatch rule.

## Reproduce

Build [the exact #101 runtime](../opt_in_conv2d_cpu_route_n3/README.md), then:

```sh
c=tools/current_runtime
q=$c/opt_in_conv2d_cpu_route_slow_only
python3 -B "$q/project.py" --parent "$w/n3.hepta" --output "$w/slow.hepta"
python3 -B "$c/partition.py" --runtime "$w/slow.hepta" --name narrow-conv-bias-cpu-route-slow-only --prefix bslow --output "$w/slow-source"
python3 -B "$c/build.py" --manifest "$w/slow-source/manifest.json" --source-root "$w/slow-source" --compiler "$HEPTA_STACK_BUILD/stack-module-compiler" --linker "$HEPTA_STACK_BUILD/stack-linker-session" --output "$w/slow"
"$HEPTA_TORCH_PYTHON" -s -E -B "$q/qualify.py" --binary "$w/slow/heptatorch" --parent "$w/n3/heptatorch" --wheel "$PINNED_TORCH_WHEEL" --install "$PINNED_TORCH_INSTALL_REPORT"
```
