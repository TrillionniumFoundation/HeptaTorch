# Opt-in native F32 conv2d with channel bias

The pinned #78 HIA runtime adds a separate experimental wire command:
`(125, input, weight, bias, stride_pair, padding_pair, dilation_pair, groups)`.
Each pair uses the same high/low 32-bit ordering as opcode 22. A separate code
is needed so the exact parent refuses the new operation instead of silently
ignoring a nonzero bias. Opcode 22 and all canonical build inputs are unchanged.

The HIA command resolves three live public F32 tensor owners, checks rank,
contiguity and matching bias length, and refuses any gradient-bearing operand.
It invokes the existing bounded graph-free HIA convolution and adds the
matching bias once per output element in the still private output buffer before
publication. The result stays graph-free and starts with a fresh storage and
version owner. No host computation enters the native request other than test
fixture construction.

## Reproduce

Use the pinned HIA tools and locked Torch wheel and interpreter described in
the parent runtime build instructions. From the checkout root:

```sh
c=tools/current_runtime
a=$c/opt_in_adaptive_avg_pool2d
p=$c/opt_in_conv2d
b=$c/opt_in_conv2d_bias
w=$(mktemp -d)
python3 -B "$a/project_pool.py" --parent "$c/source/runtime.hepta" --output "$w/avg.hepta"
python3 -B "$p/project_conv.py" --parent "$w/avg.hepta" --output "$w/conv.hepta"
python3 -B "$b/project_bias.py" --parent "$w/conv.hepta" --output "$w/bias.hepta"
python3 -B "$c/partition.py" --runtime "$w/conv.hepta" --name narrow-conv-no-grad780 --prefix conv780 --output "$w/parent-source"
python3 -B "$c/partition.py" --runtime "$w/bias.hepta" --name narrow-conv-bias-no-grad --prefix bias --output "$w/bias-source"
for name in parent bias; do
  python3 -B "$c/build.py" --manifest "$w/$name-source/manifest.json" --source-root "$w/$name-source" --compiler "$HEPTA_STACK_BUILD/stack-module-compiler" --linker "$HEPTA_STACK_BUILD/stack-linker-session" --output "$w/$name"
done
"$HEPTA_TORCH_PYTHON" -s -E -B "$b/qualify.py" --binary "$w/bias/heptatorch" --parent "$w/parent/heptatorch" --wheel "$PINNED_TORCH_WHEEL" --install "$PINNED_TORCH_INSTALL_REPORT"
```

Two independent cold builds matched all fifteen object pins and the same ELF
hash. Two pinned Torch eager 2.15.0+cpu AVX2 runs produced byte-identical
reports, with 30 biased forwards (2,628 output words) across ten contiguous,
owned NCHW/OIHW finite dyadic geometries. Bias changed output in all 30 cases.
Coverage includes groups, depthwise channel multiplier, dilation, padding and
an all-padding region. One native session composes biased convolution with
ReLU and average pooling without autograd. Exact #78 refuses opcode 125 in
all 30 cases, while 32 bias-free comparisons preserve its existing opcode 22
behavior, including 24 seeded backward cases. Seven malformed, gradient or
absent-handle controls refuse without changing existing owners.
See [qualification.json](qualification.json) and [build-receipt.json](build-receipt.json).

This is bounded forward evidence only. It does not qualify biased autograd,
arbitrary float or NaN behavior, layouts beyond the tested owned extents,
accelerators, frontend integration, or full PyTorch parity.
