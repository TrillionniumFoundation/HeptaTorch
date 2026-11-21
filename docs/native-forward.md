# Native forward, normalization layout and linear output versions

The current 797-function source adds three finite F32 groups to the original 32-group gate. HIA performs all forward calculations, layout copies, version changes and AD; Python records actual commands and invokes an independent fixed Torch reference. The full current gate has 35 groups and 16,365 cases. Full PyTorch declaration and subsystem qualification remains HOLD.

## Explicit inputs for no-gradient forward paths

The original `record` clears AD operation and parent slots for tensors that need no gradient. Matmul, softmax, convolution, pooling, axis reductions and joins previously recovered some forward inputs from those cleared slots. The new helpers receive the live input capabilities and operation/axis explicitly. For outputs requiring gradients, they invoke the complete original numeric helper. No temporary AD edge is fabricated. Every original backward body and record helper is retained byte for byte.

`matmul` now handles the actual rank/broadcast geometry and output version for vector×matrix. `softmax` and `log_softmax` preserve the original finite numerical tolerances. The join path derives output layout from the complete list and copies every logical element to its proper position, including transposed, gapped, broadcast and channels-last inputs. Original protocol and resource boundaries remain explicit.

## Normalization layout and linear versions

Layer normalization must return the actual reference contiguous layout. Batch normalization preserves its recognized channels-last path; otherwise a noncontiguous composite result is copied through the existing native contiguous-copy and AD17 functions. This corrected 24 actual metadata failures without changing the numerical normalization expressions or backward helpers.

Singleton dimensions require a further distinction: a tensor can satisfy the contiguous predicate while retaining noncanonical strides on dimensions of size one. The current helper canonicalizes the fresh contiguous output's own stride fields through the existing `incoming_contiguous_layout`. Independent native/Torch comparison reproduced and repaired the four batch-normalization failures for shapes `[2,3,1,1]` and `[2,1,2,2]`, with and without gradients. This change adds no allocation and changes no input, numerical kernel, backward body or Version behavior. The other 796 functions remain byte-identical to the frozen797 predecessor.

For linear with bias, a rank-2 input follows the reference fused output-version behavior. Other ranks also use version 0 when both input and bias are contiguous. The nonfused fallback produces version 2 for a vector input and version 1 for higher ranks. A bias-free result retains matmul's native version. The rule was checked against the exact fixed upstream Linear.cpp and 324 actual Torch observations covering ranks, strides, empty extents, bias layouts and gradient combinations. The new output's own saved creation-version is recorded consistently; later mutation checks remain mandatory.

## Fixed finite acceptance

| Group | Cases | Actual AD cases | Native sessions | Protocol controls | Real OOM recoveries |
| --- | ---: | ---: | ---: | ---: | ---: |
| forward_nograd | 462 | 243 | 465 | 37 | 3 |
| join_forward | 414 | 274 | 416 | 58 | 2 |
| spatial_forward | 344 | 186 | 347 | 28 | 3 |

Join additionally compares 194 truly absent gradients; spatial compares 258 actual gradient tensors. The original320 spatial case records are unchanged, with24 added singleton/channels-last regression cases. Cotangents are nonuniform. All original request/response bytes, metadata, full owners, map snapshots, inputs, RNG, FE commit/rollback and public release behavior are replayed by a pure reader. The separate212 layout/version probes retain their own exploratory scope outside the complete35 denominator. No numeric tolerances or original control predicates are relaxed.

Each runner has the same strict interface:

```sh
python3 -I -S -B tests/forward_nograd/run.py \
  --build-directory /absolute/current-build \
  --expected-build-sha256 ACTUAL_BUILD_JSON_SHA256 \
  --torch-python /absolute/pinned-env/bin/python \
  --output /new/forward-review
```

Replace `forward_nograd` with `join_forward` or `spatial_forward` for the other groups. Prepare the original official wheel/install snapshot as described in [current-runtime.md](current-runtime.md). Strict qualification recomputes the eleven original provenance expressions, verifies every immutable installed wheel member against an independent fixed anchor, and replays before/after actual computation-library mappings. Default strict readers reject the locally observed environment when its original pip/PEP610 records are absent.

The source inverse reconstructs the exact 781-function source and then the exact 775-function source, preserving 9 changed, 16 appended and 772 unchanged functions relative to 781, and 15 changed, 22 appended and 760 unchanged functions relative to 775. Full source round trips, two fresh compile/link executions, actual numerical observations and pure physical readback retain separate meanings. See [source review](../validation/current-runtime797/source-review.json), [native review](../validation/current-runtime797/native-review.json) and the [current800 plan](../validation/current-runtime800/qualification-plan.json).

The797 source and native reviews above retain their own execution scope. Candidate800 additionally requires the [public F64 view gate](native-public-f64-views.md) and a fresh complete35-group/16,365-case run; these predecessor results do not qualify800.
