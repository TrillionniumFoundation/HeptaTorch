# N=3 slow CPU route and dense F32 boundary

Extend #95's experimental native HIA command 127 with one adjacent lane:
F32 NCHW input `[3,2,4,4]`, weight `[2,2,3,3]`, bias `[2]`, implicit
groups 1, stride and dilation (1,1), no padding, and **oneDNN preference
disabled**. Its native selector chooses the existing bias-first HIA
function. The N=3 oneDNN-enabled case is refused before executing either
native convolution. The N=1/2 contract remains unchanged; this is not a
claim of arbitrary F32 or general CPU Conv2d parity.

Actual locked Torch profiling and bit comparisons distinguish the
selection rule from the reduction itself. The exact eager kernel for
N=3 with oneDNN disabled is `aten::_slow_conv2d_forward`; with oneDNN
enabled it is `aten::mkldnn_convolution`. Correctly identifying the latter
does not reproduce its internal dense reduction arithmetic: the existing
explicit post-sum HIA function still yields other F32 words. The same issue
also occurs inside #95's existing N=2 oneDNN-enabled experimental lane.
A nearby N=1 width-five geometry retains the slow Torch backend but exposes
additional dense differences even with the bias-first order, so it also
remains refused. No catchall based on backend kind or batch count is added.

Two independent binaries produce identical records. N=3 slow agrees on all
39 observations and 936 output words, including 32 dense finite and seven
sparse discriminator inputs; on 38 of those observations the two native
orders differ. N=3 oneDNN-on differs on 513 words across 32 dense cases; the
command refuses all 39 on observations. N=2 oneDNN-on differs on 363 words
across all 32 dense cases while N=2 off agrees exactly. N=1 width-five slow
differs on 24 words across 19 of 32 dense cases under each preference.

The procedure replays #95's entire primary report byte-for-byte after
normalizing only the verified candidate ELF and source IDs, then records
paired backend observations on identical inputs. The receipt pins the
complete computed row stream and includes representative outputs from
positive and negative families. The prior exact #95 ELF refuses N=3
for either setting. No change here addresses convolution scratch
reclamation or any other resource-lifetime property.

## Reproduce

Build the exact #94 and #95 parents with the recipes in
[explicit bias order](../opt_in_conv2d_bias_first/README.md) and
[CPU selector](../opt_in_conv2d_cpu_route/README.md). Then:

```sh
c=tools/current_runtime
q=$c/opt_in_conv2d_cpu_route_n3
python3 -B "$q/project.py" --parent "$w/route.hepta" --output "$w/n3.hepta"
python3 -B "$c/partition.py" --runtime "$w/n3.hepta" --name narrow-conv-bias-cpu-route-n3 --prefix broute3 --output "$w/n3-source"
python3 -B "$c/build.py" --manifest "$w/n3-source/manifest.json" --source-root "$w/n3-source" --compiler "$HEPTA_STACK_BUILD/stack-module-compiler" --linker "$HEPTA_STACK_BUILD/stack-linker-session" --output "$w/n3"
"$HEPTA_TORCH_PYTHON" -s -E -B "$q/qualify.py" --binary "$w/n3/heptatorch" --parent "$w/route/heptatorch" --explicit "$w/bias-first/heptatorch" --wheel "$PINNED_TORCH_WHEEL" --install "$PINNED_TORCH_INSTALL_REPORT"
```
