# Owned gradients, Optimizers and CPU distributions in one native state

This component combines the previously qualified 355-function real gradient-object runtime and the 378-function owned Adam/AdamW plus CPU-distribution runtime. Gradients, parameter and optimizer-state Tensors, Generator state, graph edges, Saved records and Storage/Version ownership now operate in the same checked HIA owner. The final candidate contains 428 complete core functions in seven core modules, plus the unchanged RNG and owner modules.

The final runtime source is `1fed8ea7ba5bf755a9df9ca1e61d34cfd4712cc2fd66e0728493091c81c24dae`. Its native executable is `6773485896d961b6c2e62784ebb66320290df47128bd1331aecc0c8b7f6611fe`. There are 5,738 blocks and 21,348 instructions across the core modules. The existing compiler and linker limits and checks were not changed.

Python composes exact HIA bodies, invokes the pinned native compiler/linker, transports requests and checks the native process. It does not execute Tensor arithmetic, gradient propagation, optimizer updates, random sampling, allocation or object lifetime on behalf of the target.

## Source composition

The common ancestor is the frozen SGD-owned-object runtime. The direct parents are:

| Parent | Runtime source SHA-256 | Native SHA-256 |
|---|---|---|
| Real gradient objects and incoming ownership | `1b18da2cf9791e1b20982f9706434c1359ea194dd983dc6c66eb005f152e77b5` | `be18ab877f38a4027a527783db80a1418a137c7d6da91e5bc39e2af8c044cb6b` |
| Owned Adam/AdamW and CPU distributions | `5d3a3c047d024a3880d257a317a04c2166a2e87bbc6b248bef62649a88546873` | `3fc59ab7ed9c32dd75ff440ef9bfbec31756f50edd77715d665bdac098cc3f22` |

`functions.json` records the exact origin and hash of every final body. There are 24 overlay bodies; 340 of the 355 gradient-parent bodies and 333 of the 378 Adam/distribution-parent bodies remain byte-identical. All 25 distribution bodies remain byte-identical. The 17-function RNG source and four-function owner source also retain their existing pins.

The real-gradient `tgrad` and leaf accumulation remain authoritative. Adam reads through `tgrad`. Its new moments preserve the parameter layout, whereas SGD's first momentum buffer follows the actual gradient layout. The Bernoulli backward dispatcher wraps the real-gradient dispatcher; it does not restore the older private-array gradient implementation.

The builder accepts only the two exact parent sources, their explicit receipt schemas, the pinned compiler/linker and the checked overlay/map. It reproduces every unit source, object and final executable hash. The build does not import the original Python HIA authoring environment or any numerical package.

## Incoming layout and allocation

Bernoulli's derivative is a fresh `zeros_like(grad)` result. Its incoming layout therefore derives from the actual incoming gradient layout. A contiguous derivative work array and a physical Preserve-layout Storage cannot be treated as interchangeable.

Each affected parent can reserve a separate real physical Storage, Hold and Version before backward starts. The numerical work array remains in logical element order. Construction and subsequent merges use the incoming layout to write the physical Storage, while updating the logical work values consistently. This fixes the observed case where a transposed zero contribution arrives before a nonuniform contribution. Both branch creation orders now yield the same mathematically required logical gradient while preserving their actual layout and Version behavior.

The reserve is admitted before partial execution. A structured backward exception releases unused reserves and transient incoming references while retaining completed leaf gradients and Saved-release effects. Ordinary admission and frame failures continue to roll back the complete owner. Repeated clear/rebuild and real allocation-failure controls check that no transient Hold or Version survives a committed frame.

## Pointwise layout and scalar addition

Binary pointwise outputs now follow the qualified CPU TensorIterator layout rules: contiguous and channels-last fast paths, exact common dense strides, and the ordered input-stride fallback. Stores use the Tensor's actual address calculation after layout selection. The fallback compares binary32 **byte strides as signed 64-bit values**, including the pinned implementation's wrapped values on a size-one axis. Its zero test also uses the wrapped byte stride. Admission was not narrowed to remove extreme strides.

The signed-byte correction changes only `pointwise_compare_axes` relative to candidate04; the other 427 core functions are byte-identical. The eight fixed Torch observations include strides around `2^61`, `2^62` and `2^63-1`, with both operand orders. The old candidate04 output-layout failures and the final eight-case exact replay are retained separately.

Command95's no-grad scalar addition has a narrowly scoped binary32 word helper. It quiets a NaN while preserving the observed sign/payload, gives an admitted scalar NaN priority when both operands are NaNs, and returns the observed negative indefinite NaN for opposite infinities. Its finite path calls the existing arithmetic helper. The original global `addf` and other consumers are unchanged. This qualifies the frozen CPU observations; it does not assert bitwise equivalence across all PyTorch operators or ISAs.

## Executed qualification

The portable qualifier completed all 17 steps after a cold native rebuild. It checks every packaged input before and after execution. Existing source and observation bytes, numerical assertions and tolerances are retained. Two private-layout observers have only reversible internal-tag substitutions from the gradient parent to this profile; their inverse is byte-identical to the original scripts.

| Separate suite | Final native result |
|---|---|
| Original public-gradient 10 recipes | 9 qualified, 1 partially represented, 0 failures; 1,036 frames, 5,975 exact comparisons |
| Incoming-gradient 25 variants | 22 qualified, 3 unrepresented, 0 failures; 895 frames, 4,758 exact comparisons |
| Gradient alias/edge 12 variants | 12 qualified; 1,334 frames, 7,268 exact comparisons |
| Bernoulli gradient layout/alias 8 recipes | 8 qualified; 2,112 frames, 15,604 exact comparisons |
| New physical incoming reserve controls | 4 passed; 214 frames, 65,676 checks, 14,155,536 rollback bytes |
| Inherited private lifetime controls | 5 passed; 1,110 frames, 131,836 checks |
| Typed public-gradient references | 26 frames; 12 checked refusal cases |
| Signed-zero 13 / partial-backward 16 | All qualified; exact bits and strict gradient None/presence checks |
| Legacy gradient-presence 8 | Parameter/gradient/skip scope qualified; original 8 state-Version and 65 state-creation differences retained |
| Legacy gradient-clear 10 | All represented clear/rebuild behavior qualified; original 11 state-Version, 9 state-creation, 30 missing-step-Tensor observations and 6 Adam raw-word differences within the unchanged tolerance retained |
| Distribution 183, atomic contract | 161 passed, 4 transaction differences, 18 unrepresented |
| Same distribution 183, explicit operation-error contract | 165 passed, 18 unrepresented |
| Existing Optimizer 24 reference | Original mapping 18 qualified + 1 partial; its Adam/AdamW cases21–24 add 4 qualified; its formerly unmapped public-gradient-alias case20 adds 1 qualified |

The resulting mapping of the existing Optimizer reference is **23 qualified cases and one partially represented case**. Case20 is an additive mapping of an existing reference recipe, not a new upstream test. It checks a real held gradient through Optimizer `zero_grad`, real scalar mutation, a later distinct gradient, and final Storage Weak expiry.

Preserved replay reports retain their historical scope wording. A legacy explicit-state report does not measure the separate owned-state interface, and the unchanged Adam-case mapper does not by itself claim the subsequently mapped public-gradient case20. Those results are connected by the explicit reference-case mapping rather than by rewriting old report fields.

Independent peer packages separately qualify exact compilation/body coverage, owned entry capacities, four-direction typed handoff to the two parents, 48 scalar-NaN variants, two fully observed mixed-layout graphs, six cross-feature cases, two mathematical controls, and eight extreme-stride observations. Their final05 receipts remain distinct from their candidate04 receipts.

These suites overlap. Their case, frame, value and byte counts must not be summed into a count of unique upstream tests. Hosted observations were executed separately on pinned Torch with the existing 11 provenance checks. The portable replay executes native HIA against those immutable observations; it does not execute Torch locally. A new 35-case live-Torch run on this final combined executable is not claimed by this package.

## Preserved failures

The validation history retains candidate01's actual transposed zero-first merge failure; candidate01/02 NaN-payload failures; candidate02's binary forward-stride failures; candidate04's four extreme-stride failures; and the corresponding final source-bound replays. None is relabeled as a later native image.

Development-only failures are also retained: the first portable assembly used the wrong parent's receipt-field name; the first 17-step run reached a private observer still checking the old gradient-only owner tag. The receipt adapter now names each parent's schema and fields explicitly. The observer correction changes exactly two tag literals and proves the inverse returns the frozen script. No runtime arithmetic, expected values, identity assertions, numerical tolerance or reference observation was changed for those host corrections.

The parent gradient package retains its earlier public-gradient, incoming ownership, GC and host-observer failures. The independent review packages retain their own first count expectation error and original native failures. History records explain whether a result is native behavior, a host correction or a scope boundary.

## Build and run

All CLI paths are explicit and must be absolute. The two parent build directories must contain their original generated runtime, RNG/owner sources and successful receipts.

```sh
python3 -I -S -B tools/gradient_adam_distributions/build.py \
  --gradient-build /absolute/path/to/gradient-parent-build \
  --adam-distribution-build /absolute/path/to/adam-distribution-parent-build \
  --compiler /absolute/path/to/pinned-abi3-compiler \
  --linker /absolute/path/to/pinned-owned-session-linker \
  --output /absolute/path/to/new-combined-build
```

```sh
python3 -I -S -B tools/gradient_adam_distributions/qualify.py \
  --gradient-build /absolute/path/to/gradient-parent-build \
  --adam-distribution-build /absolute/path/to/adam-distribution-parent-build \
  --compiler /absolute/path/to/pinned-abi3-compiler \
  --linker /absolute/path/to/pinned-owned-session-linker \
  --output /absolute/path/to/new-qualification
```

The compiler pin is `b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f`; the linker pin is `aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16`. Any source, object, native-image or expected suite-result mismatch fails the gate. Outputs include complete raw request/response records and separate JSON reports.

## Remaining boundaries

The represented runtime is CPU binary32, rank0–4, with the inherited dimension, element, descriptor, packet, allocation and work-budget limits. It does not complete hooks, higher-order autograd, retained nonleaf gradients, arbitrary autograd identity splitting, every incoming use-count/layout path, sparse/complex/device/dtype gradient assignment, arbitrary real-valued Adam step exponents, all random distributions, or the rest of PyTorch. Overlapping assigned-gradient behavior is qualified only where explicitly observed and executed.

Legacy atomic operations retain their stated transaction behavior. The explicit partial-error commands provide a separate contract. Legacy explicit-state optimizer lifecycle differences remain in their original reports even though real owned Optimizer objects are also available. Commands 93/94 remain reserved. `full_rewrite_complete` remains false and the original API denominator remains 2590.
