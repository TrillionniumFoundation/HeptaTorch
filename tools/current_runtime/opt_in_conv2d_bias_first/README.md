# Explicit F32 conv2d bias-first accumulation

This opt-in image adds command 126 with the same tensor, stride, padding,
dilation and groups arguments as command 125. Instead of calculating a
rounded convolution sum and then adding bias, it seeds each native HIA
accumulator with the corresponding output-channel bias. It reuses the
parent geometry, bounded iteration, F32 multiply/add, and owner validation.
No host-side floating arithmetic or automatic backend selection is involved.

Command 125 stays post-sum, and command 22 remains bias-free. Command 126
rejects any gradient-bearing operand. Native biased-autograd composition
from #88 is a separate path on the pinned #82 image and is not changed.
The new ELF is not the repository default.

On the exact #92 fixtures, 126 matches all measured slow eager CPU outputs,
including the finite non-dyadic cancellation differences. The same six
discriminators still require command 125 to match the observed oneDNN
backend: command 126 deliberately disagrees with that backend on those
fixtures. Choosing a matching opcode is the caller's responsibility; the
experiment does not establish a general PyTorch kernel-selection policy,
reduction/FMA order, NaN or subnormal handling, frontend or device parity.

## Reproduce

First build both #78 and #82 using [the #82 forward recipe](../opt_in_conv2d_bias/README.md),
which retains the exact #82 projected source as `$w/bias.hepta`.
From the checkout root:

```sh
c=tools/current_runtime
q=$c/opt_in_conv2d_bias_first
python3 -B "$q/project.py" --parent "$w/bias.hepta" --output "$w/bias-first.hepta"
python3 -B "$c/partition.py" --runtime "$w/bias-first.hepta" --name narrow-conv-bias-first --prefix biasfirst --output "$w/bias-first-source"
python3 -B "$c/build.py" --manifest "$w/bias-first-source/manifest.json" --source-root "$w/bias-first-source" --compiler "$HEPTA_STACK_BUILD/stack-module-compiler" --linker "$HEPTA_STACK_BUILD/stack-linker-session" --output "$w/bias-first"
"$HEPTA_TORCH_PYTHON" -s -E -B "$q/qualify.py" --binary "$w/bias-first/heptatorch" --parent "$w/bias/heptatorch" --old-conv "$w/parent/heptatorch" --wheel "$PINNED_TORCH_WHEEL" --install "$PINNED_TORCH_INSTALL_REPORT"
```

The test replays the complete #82 qualification and #92 default/paired
backend comparison on the new ELF, requiring all reports to match their
committed records aside from the separately verified candidate ELF pin.
The new route is then checked against all 28 backend observations, against
30 grouped/padded/dilated dyadic forward inputs (including actual per-channel
bias differences), and against six refusal controls. The parent refuses the
new opcode; it cannot silently substitute post-sum behavior.
