# F32 conv2d bias: a backend-conditioned rounding blocker

This checks the exact #82 HIA executable and locked PyTorch 2.15.0+cpu.
It deliberately **retains observed disagreements** rather than widening
the previous dyadic acceptance or changing native behavior.

For two nonzero terms in a channel, the measured HIA operation adds the bias
after the rounded convolution sum. Locked eager does not use one universal
order on these examples. The actual profiler recorded
`aten::_slow_conv2d_forward` for N=1 with oneDNN enabled, and
`aten::mkldnn_convolution` for N=2 with oneDNN enabled. For the same N=2
arguments, toggling only `torch.backends.mkldnn.enabled` off selects the slow
kernel and changes its output. ATen reports AVX2; this is not an assertion
that oneDNN uses the same internal instruction set.

An exact F32 witness uses the bytes of 0.1 and 0.2 as weights and -0.3 as
bias. Direct eager with the slow kernel returns `000080b2` in little-endian
hex (-2^-26). Direct eager with oneDNN on the N=2 fixture returns
`00000000`, as does native opcode 125 for either fixture. The native and
eager no-bias buffers match bit for bit in every observation, isolating the
bias-order difference from convolution reduction on this test. The
separately computed pre- and post-sum comparators reproduce the reported bits.
No general statement about either eager backend's other shapes, FMA choices,
or instruction order follows from this test.

Six adversarial finite cases and one agreement control are checked under
four N/oneDNN combinations: 28 observations, 18 mismatching observations
and 192 mismatching output words. The two runs against separate byte-identical
native builds produced exactly the same [qualification report](qualification.json).

## Reproduce

Use the pinned build recipe in [the forward slice](../opt_in_conv2d_bias/README.md).
From the checkout root, run:

```sh
"$HEPTA_TORCH_PYTHON" -s -E -B tools/current_runtime/opt_in_conv2d_bias_rounding/qualify.py --binary "$w/bias/heptatorch" --wheel "$PINNED_TORCH_WHEEL" --install "$PINNED_TORCH_INSTALL_REPORT"
```

OneDNN is required and stays enabled for the ordinary-path measurements;
the off setting is a paired diagnostic control, not how parity is judged.
Successful execution means this known gap is faithfully reproduced, **not**
that arbitrary F32 bias parity passes. The constrained seeded AD adapter still
refuses values outside its dyadic grid. No runtime, default, or AD behavior changes.
