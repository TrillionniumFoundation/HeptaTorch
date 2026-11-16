# Backward execution, partial failures and gradient presence

This native HIA successor adds commands 58/59 to the separately frozen
`backward_release` component. Command 58 executes ready nodes and commits the
effects of completed nodes when a later backward node raises. Command 59 exposes
actual per-Tensor gradient presence, independently of gradient values. The
existing atomic command 57 remains available.

The fixed 16-case actual PyTorch branch report was replayed on this executable:
all 16 cases passed, with 606 interactive frames and 1,587 exact comparisons. Each
leaf's `None` versus defined gradient is checked through command 59, and the
numerical gradients, failing operator, completed-node count and completed-leaf
count match the recorded PyTorch execution. This resolves the observed
downstream-release/upstream-version failure for this explicit interface. It does
not establish full PyTorch API or arbitrary-hook compatibility.

## Interfaces

Outer framing remains `HTOWNR01` / `HTOWNS01`; request/response packets remain
`HTORCH03` / `HTORSP03`. The maximum request and response body is 1 MiB. Command
records still contain eight u64 words and use the existing typed Tensor IDs and
same-frame result references.

| Command | Word1 | Word2 | Word3 | Result |
| --- | --- | --- | --- | --- |
| 58 | Output Tensor | Explicit seed Tensor or0 | `retain_graph` flag 0/1 | Runtime error code, plus32-byte execution payload |
| 59 | Tensor | Unused | Unused | Gradient presence 0/1, no payload |

Command 58's 32-byte payload is four u64 words:

1. Runtime error code, equal to the record's scalar result value.
2. Number of successfully applied graph nodes, including leaf accumulators.
3. Number of successfully applied leaf accumulators.
4. Failing internal AD opcode, or0 on success.

An internal AD opcode identifies the derivative operation, not a public command
number or Tensor handle. A successful58 returns value 0. Neither58 nor59 returns
a Tensor result; using those scalar results as Tensor references is rejected by
the ordinary typed-handle checks.

### Admission errors and runtime errors

Flags, public handles, required gradients, exact seed shape and result capacity
are validated before node execution. Work/queue allocation also occurs before
execution. These admission errors still reject the entire frame and return the
canonical 32-byte error response. A real allocation failure is not fabricated as
a Python exception or silently retried after partial execution.

A Saved-version mismatch, already-released Saved role, or invalid gradient shape
encountered at a ready node is returned in the 58 record. The response envelope
status and record status remain 0, so the owning step commits the trial. The
record's scalar result and payload carry the backward error. A frontend should
first accept the committed state/result records, then raise the corresponding
backward exception.

When 58 returns a runtime error, the packet stops at that record. Later input
commands are not executed, and response count is the number actually completed,
including 58. Commands before 58 and all already completed backward-node effects
are committed. A client must not require the response record count to equal the
input count in this case.

If 58 succeeds and a subsequent ordinary command fails, the existing whole-frame
transaction rejects that frame and rolls everything back. Command57 always
retains its whole-frame failure contract. These distinctions are intentional
and have independent native controls.

## Node scheduling and Saved lifetime

The implementation first marks topology and counts every required next-edge,
including repeated parents. It uses an actual dependency queue rather than
checking all Saved values before applying any node. Ready nonleaf nodes are
ordered by construction sequence. Ready leaf accumulators receive priority over
ordinary nodes, matching the `AccumulateGrad` priority used by the pinned CPU
engine. A shared leaf is not ready until all incoming dependencies have completed.

For each ready node, the runtime checks released Saved roles, Saved versions and
the existing gradient-shape requirements; applies the existing numeric kernel;
releases that node's Saved roles when not retaining; and only then advances its
parent dependencies. A failing node does not release its Saved values, while
already completed nodes remain released. A leaf writes its accumulated gradient
when that leaf becomes ready, so a different branch can retain a committed
gradient even when another branch later fails.

The recorded original failure now produces error 6 followed by 10 on retry:
downstream nodes completed and released their Saved values before an upstream
version check failed. The original atomic57's6→6 behavior is preserved in its
own frozen component and remains available in this profile.

The readiness heap and dependency array are a temporary private Hold in the
existing owned pool. They are released before returning. Work and unobservable
nonleaf gradient buffers are also released on either runtime success or runtime
error. Unvisited Saved obligations, graph topology, public Saved objects and leaf
accumulation references remain intact.

The source basis is the pinned PyTorch`31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`
`torch/csrc/autograd/engine.cpp`, `engine.h`, and
`functions/accumulate_grad.cpp`. The actual reference wheel's distinct build
commit, exact wheel hash and verified release source equivalence remain in the
observation report; they are not represented as the same Git commit.

## Gradient presence and optimizer behavior

Node field 360 is a persistent presence flag. Construction initializes it to 0.
A successful leaf accumulator sets it to 1, including an empty gradient or a
gradient containing only zeroes. All four existing/new backward entries
(7,29,57,58) maintain this invariant. A nonleaf or a Tensor that does not require
gradients reports 0 through 59. Presence survives owner relocation and graph reset
for a preserved leaf; a newly reused node starts absent.

The legacy command 8 zeros the existing gradient values. It does not create a
gradient where presence was0 and does not remove an existing gradient. It is not
`zero_grad(set_to_none=True)`. Existing raw gradient reads through 11/40 retain
their old wire behavior; a frontend uses59 to decide whether those bytes represent
a defined gradient or whether it should return `None`.

Simple SGD, momentum SGD, Adam and AdamW now skip absent-gradient parameters after
their existing argument/configuration validation. They leave parameter bytes,
supplied optimizer state bytes and all associated versions unchanged. A defined
zero gradient still follows the original numeric update and weight-decay path.
The present-gradient arithmetic bodies and the verified Adam FMA behavior are
unchanged. The native tests distinguish absent gradients from defined zeroes,
including AdamW's extra parameter version increment when decay applies.

Optimizer state is still explicitly supplied by the caller, and step numbers are
still caller-controlled. This component does not yet create a real Optimizer
object with lazy per-parameter state, does not reset an existing Version identity
to imitate a newly created state, and does not claim that those lifecycles equal
PyTorch's internally created momentum/state Tensors.

## State format and source custody

The numerical owned state is tagged `HTSTATE5`. The frozen atomic component's
`HTSTATE4` does not maintain gradient presence and therefore cannot accept the
new state. Both versions accept a live empty buffer; nonempty cross-format calls
are rejected with status 7 before mutation.

An actual checked HIA qualification links both complete implementations under
reversible symbol prefixes and moves the unique owner between their real step
and close exports. Both directions preserve the exact owner address, length and
all 595,232 compared bytes on rejection; the original step resumes correctly.
Cross-format close unmaps the allocation while the process is still alive, and
the returned empty owner can initialize the other version. No serialized pointer
or debugger write is used to construct those owners.

| Identity | SHA256 |
| --- | --- |
| Frozen atomic-release core | `e9e325374153f64744fbddfd2ccee00e570585cc42e4ad239361ce035ad4d2c2` |
| This profile's core | `302677af52d3e04520e6f132aaf17b98cbe5d2cf404aec040bf5144e324d2d4d` |
| Native session | `8f1fae7cc6a9181d788ef80c82a43e24c3ba526e77d01f8a761ea5d9913be4ae` |

The core has 253 functions, 3,039 blocks and 11,770 instructions, plus three unchanged
owned entry functions. Five native modules are independently compiled and linked
by the existing pinned ABI3 tools. No compiler or stack limit is enlarged.
Of the 242 frozen atomic-core functions,234 remain byte-identical. The eight
changes are entry dispatch, object dispatch, legacy accumulation presence,
three optimizer gates, and two internal state-tag constants. Eleven new
functions implement the queue, execution and presence query.

The builder reads the frozen atomic core and the original candidate 09 authoring
snapshot. It verifies baseline re-emission, applies checked overlays and runs the
unchanged Core verifier/CFG compactor. Exact real function documents are parsed
individually before being combined for verification; no fake callee body or
relaxed native admission replaces a source limit. Every actual compiled unit and
all compiler/linker inputs are bound before and after the build. The generated
`complete-overlay.hepta`, `source-map.json` and `atomic-to-partial.diff` expose
the exact changes without another hand-maintained numeric implementation.

## Build and tests

Build the frozen Tensor object candidate 09 and frozen atomic-release component
first. Then:

```sh
python3 -I -S -B tools/backward_partial/build.py \
  --repository . --source-root "$HIA_SOURCE_ROOT" \
  --tensor-objects build/tensor-objects-stack \
  --baseline build/backward-release \
  --compiler "$HEPTA_STACK_MODULE_COMPILER" \
  --linker "$HEPTA_STACK_SESSION_LINKER" \
  --output build/backward-partial
python3 -I -S -B tools/backward_partial/test_partial.py \
  --executable build/backward-partial/tensor-partial-session \
  --protocol tools/tensor_objects/protocol.py \
  --observations tools/backward_release/evidence/actual-torch-36-observations.json \
  --output build/partial-native
python3 -I -S -B tools/backward_partial/test_presence_bounds.py \
  --executable build/backward-partial/tensor-partial-session \
  --protocol tools/tensor_objects/protocol.py --output build/presence-native
```

The two suites passed 19 groups / 352 frames. They include 300 simultaneous ready
leaves across a descriptor-page boundary, repeated parents, retained failures,
strict output capacity, full-frame rollback, an actual 32 MiB process address-space
allocation failure and subsequent recovery, and 262,144 exact large-gradient values.
The frozen57 independent controls passed 14 groups / 207 frames on this new image.
Four additional independent58 controls passed 55 frames / 105 exact comparisons.

Replay the actual 16-case reference with the independent script:

```sh
python3 -I -S -B tools/backward_partial/replay_partial_observations.py \
  --binary build/backward-partial/tensor-partial-session \
  --expected-sha256 8f1fae7cc6a9181d788ef80c82a43e24c3ba526e77d01f8a761ea5d9913be4ae \
  --source build/backward-partial/runtime.hepta \
  --expected-source-sha256 302677af52d3e04520e6f132aaf17b98cbe5d2cf404aec040bf5144e324d2d4d \
  --controls tools/backward_partial/native_release_controls.py \
  --transport-observer tools/owned_sessions/client.py \
  --observations tools/backward_partial/evidence/actual-torch-16-observations.json \
  --output build/partial-hosted-replay
```

This replay uses actual reference values and recorded execution traces, never
native output as a golden reference. The source report separately checks traced
and hook-free runs for equivalent gradients, presence and Saved outcomes. It
does not claim that the native interface implements user hooks.

The cross-format builder and test use the same CLI as their atomic counterparts,
with `--old build/backward-release`, `--new build/backward-partial`, and the same
pinned compiler/linker. They create a separate qualification-only executable.
Python is only build orchestration, wire transport and independent testing; all
Tensor execution, scheduling, mutation and ownership operations are native HIA.

`evidence/` retains both original authoring/native refusals, the exact executed
producers, the earlier command 58 candidate's receipts, the final raw native
qualification, actual hosted source reports and independent replays. Earlier
identities are not relabelled as this image. The separate eight-case actual
gradient-presence/optimizer reference was also replayed: 880 frames, 2,088 exact
comparisons and 180 Adam comparisons at the recorded tolerance (the observed
native/reference bit difference was zero). All parameter versions and gradient
presence matched. The report retains eight momentum-state version differences
and 65 explicitly supplied versus lazily created state observations. It does
not turn those state lifecycle differences into a full Optimizer compatibility
claim.

Rank 0–4, binary32, per-dimension, total-element, pool, slot and graph bounds remain
those explicitly documented by the underlying Tensor object profile. No
higher-order differentiation, `create_graph`, sparse/complex gradients, arbitrary
hooks, `set_to_none=True`, or general Optimizer object is added here. The actual
signed-zero reference confirms a remaining difference: an initially absent leaf
gradient must preserve a negative-zero seed, while this candidate adds it to a
positive-zero buffer. Other gradient paths also require separating direct
assignment from reductions such as gather's zero-initialized scatter-add. That
correction is a separate successor; this candidate's original identity and
receipts remain intact. The successful finite cases do not prove all-bit gradient
equivalence. Full PyTorch rewrite closure remains false, with the same
2,590-schema inventory denominator.

## Preserved signed-zero failure

The unchanged 13-case raw-binary32 replay of the actual pinned PyTorch report
produced 4 passes and 9 failures on this frozen image: 166 interactive frames,
131 metadata comparisons and 70 gradient-word observations. Fifteen observed
words were `+0` where the reference was `-0`. The original report, native replay
and exact replay source are included. The passing gather case requires its
zero-initialized scatter reduction; changing every gradient accumulation to
first-write copying would not preserve that behavior. A successor must fix
initial gradient materialization and input-gradient accumulation separately.
This package intentionally preserves the failed image and does not claim that
signed-zero gradient parity is complete.
