# Opt-in native graph-free F32 conv2d

This projection follows the exact #75 average-pooling runtime and changes only
the existing native `conv2d` body plus three new HIA functions. Public opcode 22
on that parent allocates a graph-free output but then reads its unset AD operand
slots. This branch instead passes the two live operand handles directly along
the forward path. It retains the existing convolution validation, work bound,
spatial/grouped indexing, scalar F32 multiplication and accumulation. Owned
output metadata remains graph-free. The original graph recording, forward,
and backward routines are used when either operand requires gradients.

The qualified scope is Linux x86-64 CPU, bias-free finite dyadic F32 fixtures,
positive contiguous owned NCHW input and OIHW weights, with the measured
stride/padding/dilation/group settings. Nothing changes the default runtime,
a full PyTorch operator or derivative admission, or whole-project acceptance.
There is no claim about arbitrary floating-point reduction order, special
values, layout, bias, frontends, devices, or higher-order differentiation.

## Reproduce

From the checkout root, use the fixed HIA toolchain and upstream-locked Torch
wheel and interpreter from the repository build instructions:

```sh
p=tools/current_runtime/opt_in_conv2d
a=tools/current_runtime/opt_in_adaptive_avg_pool2d
w=$(mktemp -d)
python3 -B "$a/project_pool.py" --parent tools/current_runtime/source/runtime.hepta --output "$w/avg.hepta"
python3 -B "$p/project_conv.py" --parent "$w/avg.hepta" --output "$w/conv.hepta"
python3 -B tools/current_runtime/partition.py --runtime "$w/conv.hepta" --name narrow-conv-no-grad780 --prefix conv780 --output "$w/source"
python3 -B tools/current_runtime/build.py --manifest "$w/source/manifest.json" --source-root "$w/source" --compiler "$HEPTA_STACK_BUILD/stack-module-compiler" --linker "$HEPTA_STACK_BUILD/stack-linker-session" --output "$w/build"
"$HEPTA_TORCH_PYTHON" -s -E -B "$p/qualify.py" --binary "$w/build/heptatorch" --parent "$OPT_IN_AVG_POOL_777_ELF" --wheel "$PINNED_TORCH_WHEEL" --install "$PINNED_TORCH_INSTALL_REPORT"
```

The two cold builds matched executable SHA-256
`37b2c11cdf50fb136dab2273063a75489ef05ef7c4fd582725f7e9550382a464`.
The two qualification runs matched byte for byte. Eight geometries were
run with neither, either, and both operands requiring grad: all 32 matched
Torch eager 2.15.0+cpu AVX2 at the F32 bit level, including the seeded
gradients where applicable. The eight graph-free requests returned status 1
on the parent and eight valid outputs on the candidate. All 24 gradient
requests matched the same parent and the independent Torch result. A channel
mismatch returned status 2 on both images. The same native session also ran
conv2d → ReLU → average pooling without a graph or detach, and matched
every eager stage. Tests check ownership, nonmutation, result survival after
inputs are released, and owner invariants. See [qualification.json](qualification.json)
and [build-receipt.json](build-receipt.json).
