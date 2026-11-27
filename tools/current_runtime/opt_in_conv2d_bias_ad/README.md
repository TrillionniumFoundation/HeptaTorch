# Seeded F32 biased convolution on the native HIA graph

For the measured finite, owned contiguous NCHW/OIHW CPU domain,
`lower.conv2d_bias_seeded` builds one HIA wire graph on the exact #82 executable:

1. Opcode 22 computes native convolution.
2. Opcode 17 creates a native `[1,Cout,1,1]` view of the rank-one bias.
3. Opcode 2, subop 1, creates native broadcast addition.
4. Opcode 29 differentiates the result using an explicit F32 seed; opcode 11
   reads the required input, weight, or bias gradients.

The Python adapter only validates bounds, derives integer extents, checks
binary custody, and passes commands and bytes. It does not compute outputs or
gradients and imports no Torch. Both public intermediate owners are released
**before** native backward. The native graph retains their needed data, verified
by exact gradient reads and by releasing every remaining owner followed by the
real native invariant audit. An optional second seed exercises native accumulation
in the same graph. This is not a single fused HIA opcode or a modification of
#82's wire opcode 125, which remains graph-free and rejects gradient operands.

## Reproduce

Use the two-build recipe from [the #82 forward slice](../opt_in_conv2d_bias/README.md).
Run from the checkout root with the locked upstream Torch reference:

```sh
"$HEPTA_TORCH_PYTHON" -s -E -B tools/current_runtime/opt_in_conv2d_bias_ad/qualify.py --binary "$w/bias/heptatorch" --wheel "$PINNED_TORCH_WHEEL" --install "$PINNED_TORCH_INSTALL_REPORT"
```

The qualifier compares directly with `torch.nn.functional.conv2d(x, weight,
bias, ...)` on locked 2.15.0+cpu AVX2, not with a separately lowered Torch
composition. It covers all seven nonempty `requires_grad` masks for each of
eight matched geometries, including grouped, depthwise, dilated and batched
cases. Two geometries repeat each mask with a second distinct seed on the same
graph. Physical F32 outputs and gradients are compared bit for bit. Six invalid
requests refuse before process creation. See [qualification.json](qualification.json)
for the exact observed cases and hashes.

The adapter checks F32 finite values and seeds on the dyadic grid of multiples
of 1/16 in [-1, 1], limit 4,096 words per input or output, and positive extents
within its declared geometry bounds. The fixtures sample that domain but do
not prove all combinations. No claim is made for arbitrary F32 reduction and
rounding order, non-contiguous inputs, graph mutations between calls, higher
derivatives, accelerators, frontend equivalence, or whole-project acceptance.
