# Native typed dropout profile

The production source is generated from the exact retained446 parent and the checked HIA authoring files in this directory. All numeric work, random generation, object ownership, gradient history and allocation execute in native HIA. Python emits/checks HIA source and runs qualification; it does not provide a Tensor runtime or numerical fallback.

The frozen candidate has these identities:

| Artifact | SHA-256 |
| --- | --- |
| Full HIA source | `6ffe509afce28ff15705b4c7b269eaef8f8f89f9d8b2486d4d9fa45b6be66ef3` |
| Native executable | `830f0b4ef8d6ad8c6a105be56d383628b45c84e3e43905fee6812c7c52d35a51` |
| Retained446 parent source | `666230f05cc6942470006d1f184135321d8cbc26a6c6ba0bf7e3cf9857fe515b` |
| Native module compiler | `b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f` |
| Owned-session linker | `aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16` |

There are 523 core functions, 6881 blocks and 25112 instructions across nine real core modules, plus the exact RNG and owned-state modules. No compiler capacity, loan checker, effect checker, stack admission or owner ABI gate is weakened. The descriptor is 640 bytes and the internal formats are `HTDTYP03` / `HTODTY03`.

## Build

Use the repository's existing pinned compiler bootstrap. `--parent` is the real retained446 native build directory containing `runtime.hepta`, `rng03.hepta` and `owners.hepta`. `--upstream` is the existing pinned HIA source checkout used for authoring verification. No upstream compiler is vendored here.

```sh
python tools/native_dropout/build.py \
  --parent /absolute/retained446/native \
  --repository /absolute/HeptaTorch \
  --upstream /absolute/pinned-hia \
  --compiler /absolute/stack-module-compiler \
  --linker /absolute/stack-linker-session \
  --output /absolute/new-typed-native-build
```

The supplied `compact_functions.py` has the original pinned hash. It removes dead block parameters while preserving the existing compiler's limits; the original authoring is re-emitted byte-for-byte before applying explicit changes. The wrapper then requires the whole source, native executable and all eleven object/source unit hashes to reproduce `profile.json` exactly. Build success alone does not claim runtime qualification.

## Execute the portable qualification

```sh
python tools/native_dropout/qualify.py \
  --parent /absolute/retained446/native \
  --repository /absolute/HeptaTorch \
  --upstream /absolute/pinned-hia \
  --compiler /absolute/stack-module-compiler \
  --linker /absolute/stack-linker-session \
  --flat-compiler /absolute/frozen-c5a-native-compiler \
  --output /absolute/new-typed-qualification
```

The default dependency root is the supplied repository. When validating an isolated handoff, `--inherited-package`, `--retained-package` and `--retained-review-package` can name the original frozen handoff roots instead. Their tests and reference manifests are verified before and after execution. The inherited 16-suite coordinator keeps all historical OPEN counts and exact tolerances. The retained twelve-variant runner is admitted only under its exact `QUALIFIED_EXPOSED_SCOPE` status, twelve executed/twelve qualified/zero failures, and the original eleven provenance checks.

Each qualification step writes complete stdout/stderr, its original report, and raw interactive request/response files. The qualification output is create-only. `report.json` records exact source/native/package hashes and individual suite receipts. A native source change requires a new output directory and new evidence; prior PASS receipts are not relabeled.

## Separately counted evidence

| Suite | What it establishes |
| --- | --- |
| Original native-dropout 8 | Additive replay of the exact eight previously unrepresented rows in the immutable 183-row distribution observation |
| Native-dropout contract 28 | True bool mask, optional training, empty F32 mask, actual Saved mask, Version checks and complete RNG behavior |
| Empty identity 9, independently supplied | Same Tensor history, captured AccumulateGrad, nested/old edges, successful empty nonleaf AD, real cycle and detach release |
| Detach 26, independently supplied | Actual F32/bool view rejection, retained/gradient identities, and all original mutation witnesses including true bool logical-not |
| Existing Optimizer 6 | Continued legal member ownership, actual zero/step behavior, real lazy states, layout, versions and None skipping |
| SGD effective-layout 6 | Plain/maximize/maximize-with-decay first momentum layout and two-step behavior for empty and nonempty transposed G |
| Retained 12 | Original retained gradient object, prehook, Saved failure and Optimizer lifetime behavior |
| Native controls | Typed Storage/alias/Saved migration, F32 rejection matrix, entire-owner/RNG rollback, real OS allocation failure and recovery |
| Inherited 16 named suites | Original gradient, AD, raw/owned optimizer and distribution behavior, with the original explicit remaining gaps |

The independent identity, detach, output-capacity, cross-profile owner and complete-accounting reviews are separate packages with their own manifests. Their source/native identities must match this profile when an aggregate gate admits them. They are not implicitly qualified by this producer package.

## Preserved failures and corrections

The validation directory retains original reports and source identity for the early bool Hold/F32-reader failure, the empty-AD unrepresented parent, the real Optimizer-after-history audit failure, and both actual SGD layout failures. Candidate10's first actual existing-member replay had five passes and one true momentum-stride failure; the later maximize observation reproduced two additional failures on that same immutable candidate. Candidate11 passes the unchanged comparisons.

Host-side issues are kept separately: initial authoring-name refusals before a native build, a negative-test prefix that unintentionally changed its own Saved Version, a cross-dtype error-priority expectation, an allocation-test setup limit, an old Sub-versus-Mul literal in resource controls, and a coordinator that did not recognize the frozen retained runner's exact success label. Corrections include their precise scope and original reports; none changes numerical reference values, tolerance or the production runtime merely to hide an assertion.

See `PROTOCOL.md` for the exact wire and lifecycle contract. Full PyTorch rewriting remains incomplete, with denominator 2590 unchanged. Bool Storage is real; this package does not claim that F64/I64 kernels, generic type promotion or the entire Tensor API are complete.
