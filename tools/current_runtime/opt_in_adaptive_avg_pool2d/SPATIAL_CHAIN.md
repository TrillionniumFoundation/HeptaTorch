# Owned conv2d → ReLU → average pooling

This bounded application test uses the exact native ELF and source of
HeptaTorch #75, commit 951eab0503ba8e2d76f14567946894729529765a.
It runs four dyadic CPU F32 fixtures, including batch and channels,
grouped convolution, stride, padding and dilation.

- The seeded backward lane executes all three stages and first-order
  backward in one native session. Every intermediate passes by HIA
  handle; only the original input, weights and backward seed enter as
  literal arrays. Both input and weight gradients match actual eager
  Torch byte-for-byte.
- The detach lane runs conv and ReLU with gradient tracking, then
  detaches the activation without copying and dispatches the repaired
  no-grad average-pool path. It verifies detach's shared Storage and
  Version, and the final result's separate ownership. This is a mixed
  graph, **not** ordinary no-grad inference.
- Ordinary no-grad conv2d refuses with status 1 on all four fixtures.
  The test checks roots before and after, stops that session before
  ReLU/pooling and preserves the refusal as an open blocker.

Across both successful lanes all three stage tensors match actual
pinned Torch in raw F32 bytes, metadata and gradient flags. The
observer checks unchanged live owners and bytes, varied release
orders, and lifetime of the output after releasing intermediates.

## Reproduce

After [building #75](README.md) from the repository root:

    p=tools/current_runtime/opt_in_adaptive_avg_pool2d
    "$HEPTA_TORCH_PYTHON" -s -E -B "$p/qualify_spatial_chain.py" \
      --binary "$w/build/heptatorch" --source "$w/runtime.hepta" \
      --wheel "$PINNED_TORCH_WHEEL" --install "$PINNED_TORCH_INSTALL_REPORT"

The checked report is [spatial-chain-qualification.json](spatial-chain-qualification.json).
No runtime, default, Actions, or canonical acceptance entry changed.
