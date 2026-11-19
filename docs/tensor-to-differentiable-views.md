# Tensor conversion and differentiable views

Tensor conversion and differentiable views execute in the canonical
[HIA application](../tools/current_runtime/source/runtime.hepta). The
[current manifest](../tools/current_runtime/source/manifest.json) identifies its
complete source and compilation units. Use the default build and test commands
in [Current runtime](current-runtime.md); the historical reconstruction recipes
below are retained test inputs.

## Native behavior

The implemented CPU F32 paths preserve the Tensor, Storage, Version and gradient
relationships required by the current view and copy tests:

- A differentiable strided view records an edge to its immediate input. The
  ultimate storage base is retained separately, with its own lifetime.
- Version-driven history refresh moves the retained-gradient hook to the current
  Tensor identity and incarnation. Saved values remain subject to their Version
  checks before backward propagation.
- Incoming gradients for overlapping or holed views carry their actual storage
  span, effective offset and strides. The gradient layout contract selects the
  existing contiguous clone path when a fresh public gradient is required.
- Backward reserves its private preserve buffer before changing public state.
  The resource cases check complete owner and RNG rollback after allocation
  failure, then repeat the request successfully after restoring the process limit.
- Coverage normalization preserves a NaN numerator's sign and payload and sets
  its quiet bit when coverage is positive. Other inputs use the original
  arithmetic path.

The R4 view/reshape and wrapped-scalar extensions, including their resource
cases, are described in [View, reshape and scalar behavior](view-reshape-scalar-r4.md).
Supported dtype, rank, device and operation limits are recorded in the
[capability inventory](../compatibility/capabilities.json).

## Current qualification

The [current coordinator](../tests/qualify_current_runtime.py) requires these
view-related groups as part of the complete current-runtime gate:

| Group | Required coverage |
| --- | --- |
| `original76` | Original view26, scalar32, differentiable-view14 and input4 comparisons |
| `current35` | Tensor and gradient lifecycle, Saved values, Version checks and owner controls |
| `asstrided16` | Original copy4, reduction4, lanes2 and branches6 comparisons; complete frame and owner readback |
| `resource7` | R4 allocation3, backward private-cache1 and R3 preserve3 recipes, with original budgets and rollback checks |

These groups bind the actual current source, native executable and cold-build
report. Their adapters retain the original numerical expectations and verify the
source transitions used to run them on the current application. Runtime results
and the source identity they qualify are recorded in the
[current audit](current-audit.md).

## Retained reconstruction inputs

The 09b, R2 and R3 recipes remain inputs to historical reconstruction and
independent checks:

| Recipe | Role |
| --- | --- |
| [Recovered Tensor.to 09b](../tools/tensor_to_native_recovered/) | Frozen twelve-unit parent used by R2 |
| [Differentiable view R2](../tools/differentiable_view_recovered_v2/) | Rebuilds from 09b with the constructor and retained-hook correction |
| [Differentiable view R3](../tools/differentiable_view_recovered_r3/) | Rebuilds from R2 with incoming-gradient geometry and coverage handling |

The unused R1 live recipe has been retired. Its 32 installed files are indexed
in [the R1 retirement record](../validation/differentiable-view-r1-build-history/live-source-retirement.json)
with exact Git blobs, bytes, hashes and archive-member mappings. Its 26 source,
generated and build records also remain byte-identical in the original
[R1 build archive](../validation/differentiable-view-r1-build-history/); the six
installation-only records remain available in the fixed Git snapshot. Original
archive modes and later Git installation modes are recorded separately.

The original outcomes remain distinct: R1's strict14 replay had 13 passes and
one failure; R2 passed strict14 but passed only one of four input-boundary cases;
R3 passed strict14 and input4 and separately passed the scoped 72-case replay.
The original [R1 runtime evidence](../validation/differentiable-view-r1-native14-history/),
[R3 build history](../validation/differentiable-view-r3-build-history/) and all
other reference records are retained unchanged. The complete prior development
account is available in the
[fixed historical document](https://github.com/TrillionniumFoundation/HeptaTorch/blob/07eb8fd19539add61458d70b1f0076c9660c8ad3/docs/tensor-to-differentiable-views.md).

## Full-system scope

The complete target remains 2,590 native declarations, 688 derivative
declarations and 17 system gates in the
[acceptance ledger](../compatibility/full-acceptance.json). General dtype,
layout and device behavior, higher-order and forward automatic differentiation,
public frontend integration and the remaining system contracts are still part
of that target. Current finite view qualification and retirement of an unused
recipe do not close those full-system requirements.
