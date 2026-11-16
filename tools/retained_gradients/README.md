# Native retained gradients

This profile adds actual `retain_grad` behavior to the immutable combined
gradient/Adam/distribution parent. It contains 446 core HIA functions in nine
native modules including the unchanged RNG and owned-entry units. Runtime
work remains native HIA. The host builder checks and composes HIA bytes,
invokes the pinned native compiler and linker, and verifies every resulting
object and executable hash.

| Artifact | SHA256 |
| --- | --- |
| Parent source | `1fed8ea7ba5bf755a9df9ca1e61d34cfd4712cc2fd66e0728493091c81c24dae` |
| Parent executable | `6773485896d961b6c2e62784ebb66320290df47128bd1331aecc0c8b7f6611fe` |
| Retained source | `666230f05cc6942470006d1f184135321d8cbc26a6c6ba0bf7e3cf9857fe515b` |
| Retained executable | `7568c0a4ec40d82d3a0b46d1e8a27499dd96e8fe3fd15433523131a196da89b5` |
| Native compiler | `b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f` |
| Native linker | `aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16` |

## Build and qualification

The parent argument is the completed native output directory of the frozen
`tools/gradient_adam_distributions/build.py`. It includes the parent runtime,
RNG/owner sources, build receipt and executable. All paths below are supplied
by the caller; no development workspace path is built into the producer.

```sh
python -I -S -B tools/retained_gradients/build.py \
  --parent-build /absolute/parent-native \
  --compiler /absolute/native-compiler \
  --linker /absolute/native-session-linker \
  --output /absolute/fresh-retained-native
```

The output is `tensor-retained-session`, `runtime.hepta`, all nine actual HIA
units/objects and a complete build receipt. `functions.json` binds every body
and its origin. Exactly 410 of the 428 parent core bodies are unchanged; the
36-body overlay contains 18 replacements and 18 additions. The original
leaf gradient transfer, optimizer arithmetic, all 25 distribution functions,
RNG functions and owner wrapper remain byte-identical.

```sh
python -I -S -B tools/retained_gradients/qualify.py \
  --parent-build /absolute/parent-native \
  --parent-package /absolute/repository-containing-frozen-combined05 \
  --compiler /absolute/native-compiler \
  --linker /absolute/native-session-linker \
  --output /absolute/fresh-retained-qualification
```

This performs a cold native build plus the sixteen inherited, separately
counted execution suites. It checks the frozen parent package manifest
`3e8c80053365f7e5de1d24725c511f49f256e8437f5a6d3556f09ff07682dc69`.
All original values, tolerances, source checks and reference files stay
unchanged. Three explicit host adapters are reversible byte for byte: private
owner tags, and the former unknown-command 93/94 negative tests now checking
nonzero reserved fields. The original unknown-command test's observed FAIL
is preserved as a protocol-contract change, not a repaired runtime failure.

The fixed retained reference comprises **8 families / 12 variants**. It is
separately replayed by `tests/retained_nonleaf_review`, with independent
resource and legal owned-entry checks supplied by the separate resource
review package. These are not added to the inherited suites as a count of
unique PyTorch APIs or tests. The original Optimizer24 report remains frozen;
its retained-nonleaf admission subobservation is addressed by a separately
identified additive mapping of the same original case 14.

## Behavior and source basis

Commands 93/94 enable/query a real per-Tensor flag; `PROTOCOL.md` specifies
fields, errors and the new `HTRET001` / `HTORET01` private format. The flag
does not retain the Tensor. A live retained nonleaf's first gradient is a
fresh contiguous clone of its incoming gradient; later updates create a new
out-of-place sum. Held old gradients keep their actual Tensor, Storage,
Version and values. Optimizer admission allows leaf or retained parameters
while preserving the nonleaf graph and existing private ownership.

The hook executes before the node's Saved/version checks. Structured command
58 therefore commits a retained gradient update even when that node fails
with error 6 or 10. Its completed-node count does not count a hook alone.
Fresh results are reserved during admission, unused results are released on
structured exits, and ordinary command/admission failures retain the existing
whole-frame rollback contract. Legacy 7/29/57 remain atomic.

This follows fixed upstream
[`VariableHooks::retain_grad`](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/variable.cpp#L563-L612),
the engine's
[`prehook ordering`](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/engine.cpp#L906-L914)
and
[`node application`](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/engine.cpp#L1114-L1164),
and Optimizer
[`parameter admission`](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/optim/optimizer.py#L1188-L1197).
The hosted reference is pinned to
`ca8b5d3a7c679dcc5ad9a3d7b2bbd38f664c90e6474cf3e3c28496d5b716d1b3`.

## Boundaries

The implemented surface remains dense CPU binary32, rank 0–4, first-order,
single-output autograd under the existing bounded native protocol. Sparse
gradients, custom hooks, create_graph, functional input capture and general
in-place graph rebasing are separate open requirements. The legacy atomic
backward protocol and the structured partial-error protocol keep distinct
contracts. Existing unrepresented distribution dtypes and public-gradient
setter cases are not reclassified by this change.

Full PyTorch closure remains false, with the original denominator 2590.
