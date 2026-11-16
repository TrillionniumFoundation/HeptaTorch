# Native owned Adam and AdamW

This package adds real, lazily created Adam and AdamW state Tensors to the
frozen owned Optimizer08 runtime. Parameters, `step`, `exp_avg` and
`exp_avg_sq` have native Tensor identities and independent Storage/Version
objects. Public aliases and Saved objects observe their actual mutations and
lifetime. No host numerical runtime is used by the executable.

The frozen implementation is candidate02. Its four newly represented recipes
from the existing 24-case pinned PyTorch observation report passed an
independent replay: 908 interactive frames, 3,469 exact comparisons and
1,223,232 bytes of failure rollback checks. All 210 compared Adam parameter
and moment values were bit-identical; the previously declared tolerances were
unchanged. The existing 19 executable recipes retain 18 cases qualified
within the exposed scope plus one partial retained-nonleaf case. Combining the two reports gives
**22 qualified cases, one partial case and one public-gradient-object case
still open**, not 24 complete passes. See the separately frozen
`tests/adam_object_review` and `validation/adam-object-review` handoff for the
independent inputs, observations and source review.

This is a bounded CPU binary32 component, not complete Optimizer or PyTorch
coverage. `profile.json` preserves `full_rewrite_complete: false` and the
original 2,590-item denominator.

## Reproduce the executable

Supply an already reproduced, exact Optimizer08 native build directory. It
contains `runtime.hepta`, `overlay-authoring.py`, `rng03.hepta`,
`owners.hepta` and its original build receipt. The producer checks the exact
bytes of those inputs. The repository argument supplies the unchanged,
pinned HIA authoring helper and function mapping; the upstream source root
supplies the pinned HIA toolchain authoring sources.

```sh
python tools/owned_adam/rebuild.py \
  --baseline build/owned-optimizers \
  --repository . \
  --source-root /path/to/pinned-hia-toolchain \
  --compiler build/stack-module-compiler \
  --linker build/stack-linker-session \
  --output build/owned-adam
```

The output directory must not already exist. Paths are explicit; no workspace
path or existing candidate executable is used by the producer. `rebuild.py`
checks every package source pin before executing the producer and verifies
that the same source bytes remain afterward. The producer similarly binds
external inputs, compiler, linker, generated units and final output before
and after their use. It emits a native build receipt and `reproduction.json`.

| Artifact | SHA-256 |
| --- | --- |
| Frozen Optimizer08 input | `7c68a96b08aa9a21a52e2fd12be912c5f47dbde889e462f9a0dbb38cb3c09edf` |
| Generated Adam source | `253438648a437a75ef2dbc77138eca74cb5fe6a6e2ea322c7c38e5a535e5d2a6` |
| Native Adam executable | `fa92356954dbb9e0c454f2a58c2f78aa33950103d377c57e7b659ec9904183af` |
| Native ABI3 compiler | `b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f` |
| Native ABI3 linker | `aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16` |

The executable has 1,843,200 bytes. The assembled numerical document contains
352 functions, 4,598 blocks and 17,185 instructions. Eight real native units
are compiled and linked. This is not a claim that the aggregate authoring
document passes a larger single-document budget: selected real function
bodies are compacted against their actual callee interfaces, and every
complete production unit passes the unchanged native checker. The native
linker checks the real whole-program call graph and stack budget. No empty
callee stubs or raised upstream limits are used.

## Source continuity

`build.py` consumes the authoritative frozen predecessor instead of keeping a
second manually maintained numerical core. 308 of its 315 functions remain
byte-identical. The seven changed names are `entry`, `state_header`,
`init_state`, `so_state_init`, `so_state_header`, `owned_optimizer_command`
and `optimizer_audit_record`. The prior SGD dispatcher and auditor are
preserved under explicit aliases; algorithm 0 retains its old representation
and operations.

The owned Adam numerical loop is a renamed copy of the qualified
`adam_element`, with precisely four access changes: read the actual gradient
through `tgrad`, apply maximize before coupled decay, and address both moment
Tensors by their real logical strides. The inverse patch restores the old
function bytes. Floating-point lerp/FMA, second-moment ordering, bias
correction, division and update arithmetic remain unchanged.

Candidate01 was already executable and qualified. Candidate02 isolates the
moment allocator as `owned_adam_new_moment`, an exact name-only copy of the
frozen parameter-layout allocator, and changes two calls to it. This prevents
a later public-gradient implementation from accidentally giving Adam moments
SGD's gradient-clone layout. No previous numerical failure is attributed to
that change. `source-map.json`, the focused source diff and the numerical
inverse receipt record all function identities. Original candidate01 source,
image, receipts and the one test-authoring syntax failure before native
execution remain in the verified history archive.

The upstream source evidence uses PyTorch commit
`31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`. The exact `adam.py`, `adamw.py`,
`optimizer.py` and `MemoryOverlap.cpp` blobs and their Git/SHA-256 identities
are retained under `upstream/`. Actual Torch observations were obtained on the
hosted pinned CPU profile with all 11 provenance checks; local replay runs
only the native executable and does not claim to execute Torch again.

## Protocol and ownership

See [PROTOCOL.md](PROTOCOL.md) for complete word layouts and status codes.
The existing outer interactive frame protocol and `HTORCH03` / `HTORSP03`
packet protocol are unchanged. Nonempty private states use new tags
`HTADAM01` and `HTOADAM1`. A real state from an older profile cannot be silently
interpreted as this state.

Commands 71–79 select algorithm 1 for Adam and 2 for AdamW. Creation and
read-only inventory do not create state. A participating parameter's first
step creates three actual Tensors; None gradients skip all updates and state
creation. Defined zero and empty gradients participate. The step Tensor is a
real rank-zero float32 Tensor, read and incremented on every participating
occurrence. It is not an integer metadata substitute. Repeated occurrences
share one state immediately, and each group initializes participating entries
before its functional loop.

Commands 75 and 77 publish the retained parameter/state Tensor. Closing a
public token does not destroy an Optimizer's private reference, and publishing
again returns a fresh public token for the same stable incarnation and S/V
identities. Optimizer drop releases its obligations; public aliases and Saved
references can independently keep objects alive. The native object auditor
recomputes these logical obligations. These internal counts are not claimed
to equal C++ `intrusive_ptr::use_count()`.

Command 95 is a native, explicit no-grad scalar addition. It writes through
the Tensor's actual strided addresses and bumps shared Version exactly once,
including empty tensors. It is used to express actual external state
mutation in the fixed reference recipes. It does not claim full public
`Tensor.add_` autograd-history semantics.

Normal admission, allocation and command errors preserve the whole-frame
transaction contract, including rollback of earlier successful mutations in
that frame. Existing command 58 retains its explicit partial-backward
structured-error commit behavior. General optimizer late-exception side
effects remain outside this component's compatibility claim.

## Native qualification commands

The wire adapter is the existing frozen
`tools/tensor_objects/protocol.py`, with its pinned sibling
`tools/owned_sessions/client.py`. It serializes packets and observes responses;
it does not execute tensor arithmetic. Run the following from the repository
after the build. Each output directory must be new.

```sh
python tools/owned_adam/test_native.py \
  --binary build/owned-adam/tensor-adam-session \
  --source build/owned-adam/runtime.hepta \
  --expected-sha256 fa92356954dbb9e0c454f2a58c2f78aa33950103d377c57e7b659ec9904183af \
  --expected-source-sha256 253438648a437a75ef2dbc77138eca74cb5fe6a6e2ea322c7c38e5a535e5d2a6 \
  --protocol tools/tensor_objects/protocol.py \
  --output build/adam-native-controls

python tools/owned_adam/test_guards.py \
  --binary build/owned-adam/tensor-adam-session \
  --source build/owned-adam/runtime.hepta \
  --build-report build/owned-adam/build.json \
  --protocol tools/tensor_objects/protocol.py \
  --output build/adam-native-guards

python tools/owned_adam/test_step_counter.py \
  --binary build/owned-adam/tensor-adam-session \
  --source build/owned-adam/runtime.hepta \
  --build-report build/owned-adam/build.json \
  --protocol tools/tensor_objects/protocol.py \
  --output build/adam-step-counter
```

These suites passed 3 ownership/numerical controls, 5 admission/capacity
controls and 11 actual float32 counter boundary cases. The capacity suite
includes 900 real state Tensors, table/page/owner migration, private-root reset
and fresh publication. Allocation failure is a real native `try_new` failure
under a 64 MiB address-space limit, with whole-owner rollback checks.

For legal owner transfer, compile complete old and new units with a checked,
reversible symbol prefix and one HIA controller:

```sh
python tools/owned_adam/build_state_handoff.py \
  --predecessor optimizer008 \
  --old build/owned-optimizers --new build/owned-adam \
  --compiler build/stack-module-compiler \
  --linker build/stack-linker-session \
  --output build/adam-owner-handoff

python tools/owned_adam/test_state_handoff.py \
  --executable build/adam-owner-handoff/state-handoff-session \
  --build-receipt build/adam-owner-handoff/report.json \
  --protocol tools/tensor_objects/protocol.py \
  --output build/adam-owner-handoff-review
```

Both directions passed across 49 interactive frames. Wrong-format step kept
the same owner pointer, length and all 611,616 compared bytes. The correct
step then resumed actual state updates. Cross-profile close unmapped the
allocation while the native process remained alive; a subsequent legitimate
empty owner initialized the other profile. No user-supplied pointer or
fabricated owner is used in this test.

The required integration regressions are listed in `qualification-plan.json`.
They retain the original 13 signed-zero, 16 partial-backward, 8 presence and
10 gradient-clear recipes, the legacy SGD controls, and both existing
Tensor/RNG joint suites. `adapt_regressions.py` performs only exact, reversible
private-observer tag/layout and image-pin edits for the three scripts that
need them. Inputs, numerical values, tolerances, error classes and commit
assertions are unchanged. The old explicit-state reports retain their stated
lifecycle differences; they are not relabeled as object-API coverage.

## Explicit boundaries

The frozen limits are 1,024 groups and 4,096 ordered occurrences per
Optimizer, 65,535 descriptor ordinals, 16,777,216 values per Tensor and a
1 MiB request frame. Increasing storage is real owned allocation with bounded
admission; these are not unbounded claims.

Bias correction currently requires the **actual float32 incremented step**
to be a positive integer below 2^32. This preserves normal float32 counting,
including stagnation at 2^24. Externally assigning +0 or -0 legitimately
resumes at one. An incremented fractional, nonpositive, nonfinite or
out-of-range value causes status 7 and whole-frame rollback. Arbitrary
real-exponent bias correction is unfinished and is not hidden by truncation.

AMSGrad, other dtypes, sparse/complex, differentiable/capturable updates,
foreach/fused backends, hooks, serialization, general state-map replacement,
full in-place autograd semantics and optimizer exception partial effects
remain open. The separately developing public `.grad` Tensor branch has not
been composed into this frozen profile. None of these gaps is removed from
the whole-PyTorch goal or its acceptance denominator.
