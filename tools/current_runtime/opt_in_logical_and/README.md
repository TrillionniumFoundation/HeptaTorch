# Standalone logical_and for canonical CPU roots

This opt-in HIA candidate branches directly from exact main. Candidate-local operation 131 implements functional same-shape aten.logical_and.default for ordinary, canonical, nonempty CPU tensors over Bool, F32, I64, public F64, F16, and BF16, ranks 0–4. Inputs must use the same dtype. The result is a fresh Bool tensor, including when both input handles are the same. The inherited raw truth treats both signs of floating zero as false and nonzero values including NaNs as true. Default routing is unchanged.

The locked official CPU reference and a separate owner/wire reader agreed on 156 actual executions and 1,101 output bytes, including 24 true scalars, six shared-input owners, and 224 signaling-NaN input words. The exact predecessor refused all 156 requests with status 4 and no owner change. Two byte-identical cold builds, negative owner/shape/capacity controls, same-frame cursor and weak handles, and 290 exact prior-main frames were verified. Those retained frames include 28 still-unsupported op130 requests. Other branches are not part of this standalone build.

Not covered: empty outputs, broadcasting, mixed dtypes, views, autograd, in-place/out variants, non-CPU inputs or full PyTorch parity. Upstream accepts some cases intentionally outside the qualified scope.
