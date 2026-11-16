# Native owned SGD Optimizer and captured gradient edges

This package adds real Optimizer ownership, ordered parameter groups, lazy SGD momentum Tensor objects, and leaf `requires_grad_` to the frozen RNG plus nullable-gradient runtime. Runtime data, state, Tensor identity, gradient routing, allocation, ownership, arithmetic, and transactions are native HIA. Python is restricted to source authoring, build transport, and independent verification.

The exact source is `7c68a96b08aa9a21a52e2fd12be912c5f47dbde889e462f9a0dbb38cb3c09edf`; the linked ELF is `b6634158e079512e797d8237cca12d65052db18df23b12b8414adeb8d0b4b0e1`. The runtime has 315 core functions, 4,072 blocks and 15,237 instructions, compiled as seven complete native modules. Compiler `b1c851a0...` and linker `aae650e8...` are unchanged. The single authoritative numerical lineage remains the original 143-function HIA core and its explicit, hash-bound overlays; this package does not introduce a separately maintained copy of that numerical implementation.

## Implemented behavior

Commands 71–79 manage an actual kind-8 Optimizer object. Each parameter occurrence owns the real Tensor and gradient binding, independently of whether its public handle remains open. A realized state dictionary key and momentum Tensor have their own ownership obligations. Reset, descriptor-page preservation, table growth and owner migration preserve these bindings. Reopening a parameter or state issues a fresh public generation when necessary, while retaining its native incarnation, Storage, Version and gradient. State is keyed by Tensor identity, so separate Tensors that share Storage do not accidentally share optimizer state.

SGD skips None gradients without changing parameter data, Versions, or state. A defined empty or zero gradient participates. First momentum is a fresh real Tensor with independent Storage and Version 0; later steps mutate that same Tensor with multiply and add and advance its Version by 2. Dense transposed layout is preserved in the qualified first-momentum path. Duplicate occurrences are planned before the group's updates, which preserves the upstream first-step behavior when every occurrence initially sees no momentum buffer. Defaults and per-group overrides use their distinct upstream validation rules.

Command 90 changes a leaf's requires-grad flag without changing its Tensor/Storage/Version identity or deleting an existing gradient. Graph edges are captured during forward. Turning a formerly non-requiring input on later cannot manufacture an edge in an older graph. Turning a requiring leaf off later suppresses its current accumulation while preserving Saved checks, release behavior and actual AccumulateGrad retention. Nongrad operation outputs are true leaves, including noncontiguous transpose results. Their Storage and strides are preserved when subsequently enabling gradients.

The internal state tags are `HTOPT002` / `HTOOPT02`; the external packets remain `HTORCH03` / `HTORSP03` inside interactive `HTOWNR01` / `HTOWNS01` frames. The predecessor seven-kind state and initial `HTOPT001` are rejected as nonempty inputs. The checked owner itself is returned unchanged after rejection. Runtime requests never contain owner pointers.

See [PROTOCOL.md](PROTOCOL.md) for exact payloads, public generations, strong-reference accounting, captured-edge fields and refusal codes. Native reference counts are counts of this implementation's ownership obligations; they do not claim numerical equality with PyTorch `intrusive_ptr::use_count`.

## Executed evidence

Pinned upstream sources are PyTorch commit `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`. The source-custody receipts preserve Git blob and SHA256 verification. The actual Torch observations were produced in hosted CI with all 11 source/build provenance checks; Torch was not installed or simulated locally. Numerical expectations and error outcomes were taken from those frozen observations, or from separately derived exact test arithmetic.

| Qualification bound to candidate 08 | Actual result |
| --- | --- |
| Original 24 Optimizer-object observations | 18 recipes fully qualified in the exposed SGD scope, 1 retained-nonleaf recipe partially covered, 5 APIs explicitly open; 2,332 native frames and 9,535 exact comparisons; no unexpected failures |
| Six leaf flag and captured-edge observations | 6 passed; 600 frames and 2,489 exact comparisons |
| Three AccumulateGrad lifetime observations | 3 passed; 106 frames and 102 observations, distinguishing public weak handles from read-only private-gradient allocation observations |
| Private Optimizer roots across public closure/reset/table/page/owner growth | 72 native frames passed; preserved incarnations, Storage/Version, gradients, lazy state and last-owner release |
| Frozen signed-zero gradient replay | 13 passed; 166 frames, 131 exact metadata fields and 70 raw binary32 words |
| Frozen partial-backward replay | 16 passed; 606 frames and 1,587 exact fields/values |
| Frozen gradient-presence replay | All 8 parameter/presence/skip recipes qualified; 880 frames, 2,088 exact comparisons plus 180 existing Adam-tolerance comparisons; no bit differences in those 180 values |
| Frozen gradient-clear replay | All 10 gradient-clear recipes qualified; 780 frames, 2,669 exact comparisons plus 288 existing Adam-tolerance comparisons; six preexisting bit differences within the unchanged tolerance |
| Original common RNG/partial-backward transactions | 3 passed; 155 frames, 14 complete Generator-state comparisons and five whole-owner rollbacks |
| Original nullable/RNG/training transactions | 4 passed; 56 frames, 393 exact words, 34 complete Generator states and 2,517,280 bytes of whole-owner rollback checks |
| Independent native SGD and ownership arithmetic | 4 passed; 100 frames, 29 exact dyadic values, 1,834,848 rollback bytes |
| Typed payloads, exact limits, result-reference kinds and real constrained allocation | 4 groups passed; 77 frames, 37 precise refusals and 13,280,976 whole-owner rollback bytes |
| Legal predecessor/new owner transfers | Four directions passed across two actual 15-module linked images; 91 frames and 1,219,136 unchanged owner bytes, plus cross-format close and live-process unmapping |

The legacy raw-state optimizer replays still report their original explicit-state lifecycle differences. These are not silently relabeled as passing owned Optimizer tests: the eight-case replay retains 8 state-Version and 65 state-creation observations, while the ten-case replay retains 11 state-Version, 9 state-creation and 30 missing-step-Tensor observations. The new owned SGD recipes directly address first momentum Version 0, later +2, lazy creation and true retained Tensor identity. Adam/AdamW owned state remains a separate next component.

Historical candidate 06 is preserved with its exact source and ELF identity. Actual observations exposed two native defects: nongrad transpose was incorrectly treated as a nonleaf for command 90, and enabling gradients after forward incorrectly added a missing edge. A separate first-run host replay schema error was fixed in a new version of the replay script and is recorded distinctly. Candidate 08 passes the same corrected replay recipes and numerical assertions without relaxing them. The aggregate authoring refusal and the two native per-block-parameter refusals also remain in history.

## Rebuilding

The build consumes the actual frozen predecessor artifacts. It does not substitute empty callee bodies or change the language checker limits. Supply the complete outputs of the existing object, backward, materialization, RNG-composition and nullable builds:

```sh
python3 -I -S -B tools/owned_optimizers/rebuild.py \
  --baseline build/tensor-rng-nullable \
  --objects build/tensor-objects-abi3 \
  --common-authoring build/tensor-rng-composition/native/overlay-authoring.py \
  --nullable-authoring build/gradient-none/overlay-authoring.py \
  --partial-authoring build/backward-partial/overlay-authoring.py \
  --materialization-authoring build/gradient-materialization/overlay-authoring.py \
  --repository . \
  --source-root /path/to/pinned/hia-toolchain \
  --compiler build/stack/native/stack-module-compiler \
  --linker build/stack/native/stack-linker-session \
  --output build/owned-optimizers
```

The directory names are CLI inputs; they need not match this example. Their source bytes must match the fixed pins in `profile.json` and the builder. Every input, actual emitted unit, object and selected compiler/linker is snapshotted and checked before/after use. The resulting source, every native object and final image must reproduce the frozen bytes. `source-pins.sha256` binds the portable package itself.

The combined authoring document exceeded the existing 262,144-node host Core limit. That refusal is preserved. `compact_functions.py` checks and compacts each actual function against the real immutable interfaces, recording before/after structured hashes and block-parameter substitutions. Selected old bodies are independently regenerated through both the original whole-document path and this bounded path and must match the frozen HIA bytes. This is explicitly not claimed as whole-aggregate host admission. Production modules, containing all actual bodies and real import/export signatures, each pass the unchanged full native checker; the unchanged linker checks the actual linked call graph and stack budget. It still rejected 17- and 18-parameter blocks until the source was split into real helpers.

## Portable test entry points

`test_native.py` accepts `--binary --source --expected-sha256 --expected-source-sha256 --protocol --output`. `test_guards.py` accepts `--binary --source --profile --protocol --output`. The protocol helper is the existing byte-bound object-profile wire codec; it performs no numerical computation.

`adapt_regressions.py` binds the frozen grad-clear and two common-RNG scripts to the new private tags, eight-kind observer offset and target image identity. Its exact inverse must recover each original file byte for byte. It does not edit requests, expected numeric results, error outcomes, commit expectations or tolerances. Signed-zero, partial-backward and presence scripts require no source edits.

`build_state_handoff.py` supports `--predecessor nullable` and `--predecessor optimizer001`, together with `--old --new --compiler --linker --output`. It reversibly namespaces every actual function/import/export in both full runtimes, compiles all complete modules, and adds the checked `state_handoff.hepta` controller. `test_state_handoff.py` accepts `--executable --build-receipt --protocol --output`. The controller only moves legitimate HIA owners returned by actual `step` functions. Tests observe `/proc` read-only, reject the wrong format, resume the proper step, and verify mapped storage was released while the process remains alive.

The independent replay and accumulator handoffs are separate source-pinned packages provided alongside this one. Their historical reports remain bound to their original candidates. This package does not overwrite them.

## Remaining scope

The current public limits remain explicit: CPU binary32, rank at most 4, dimension at most 262,144, at most 16,777,216 elements, at most 65,535 descriptor ordinals, 1 MiB request/response frames, and bounded native owned allocation. Optimizers admit at most 1,024 groups and 4,096 parameter occurrences; full object audit retains its existing 4,096-slot work bound.

Public `.grad` Tensor aliases and their Versions, nonleaf `retain_grad` optimization, Adam/AdamW owned step and moment Tensors, hooks, state dictionaries, serialization, warning objects, arbitrary optimizer failure partial effects, sparse/complex/mixed-dtype/capturable/differentiable/fused/foreach behavior, and all remaining unsupported schemas remain open. This is a qualified native component in the continuing rewrite. The 2,590-entry upstream denominator and `full_rewrite_complete=false` are unchanged.
