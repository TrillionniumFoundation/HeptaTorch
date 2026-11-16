# Nullable leaf-gradient storage

This component adds native command70, the per-parameter operation underlying
`zero_grad(set_to_none=True)`. It releases an existing leaf gradient's private
Storage/Hold and clears actual presence. A later backward reserves a new
buffer and uses the previously qualified first-gradient bit-copy behavior.
Commands8,11,40 safely handle the absent buffer instead of reading or writing
address0. The detailed wire contract is in [PROTOCOL.md](PROTOCOL.md).

This is an isolated successor to materialization05. It does not change any
frozen predecessor, compiler, linker, RNG profile or reference script. It
adds no Python numerical runtime, public gradient-Tensor handle, synthetic
Version reset, or client-supplied replacement optimizer state.

## Behavior and lifetime

| Operation/state | Result |
| --- | --- |
| Construct a differentiable leaf | Gradient is absent; no persistent gradient buffer is allocated |
| Command8 on None | Success; no gradient allocation or memory write |
| Command70 on None | Idempotent success |
| Command70 on a defined gradient | Release its private Hold/Storage; clear presence and gradient address |
| Command8 on a defined gradient | Keep presence and buffer; set its values to `+0` |
| Backward reaches an absent leaf | First materialization copies incoming binary32 words, including `-0` |
| Backward reaches a defined leaf | Accumulate into its existing gradient |
| Command58 fails before a leaf becomes ready | Reclaim that leaf's reserved buffer; leave it None |
| Command58 fails after a leaf completes | Keep its committed gradient and presence |
| Ordinary frame rejection | Restore the complete previous owner, including any cleared gradient |

Tensor construction still reserves backward **work** for differentiable nodes.
That scratch policy is separate from the persistent leaf gradient. This
component does not claim that constructing a Tensor has no work allocation.
The gradient reserve is made before node application so an admission or
allocation failure cannot erase already committed state. Every successful
frame, including command58's structured runtime-error result, satisfies:

- An absent leaf has presence0, gradient address0 and gradient Hold0.
- A defined leaf has presence1 and a valid private gradient Hold/address.
  A defined empty gradient has a real zero-length Storage and Hold.
- A nonleaf has no persistent gradient buffer. Backward work remains separate.
- Temporary staging fields336,368,376 are zero.

The complete owner retains the existing fixed profile limits: binary32 CPU
Tensors, rank0..4, dimensions at most262,144, at most16,777,216 elements per
Tensor,65,535 descriptor ordinals between resets and a bounded growable owned
pool. Freed gradient extents become available in that pool; this does not
promise a separate operating-system unmap for each gradient.

## State-version barrier

Nullable gradient addresses require **HTSTATE6**. Old HTSTATE5 code assumes
allocated leaf gradient storage, so it must never operate on this state.
The opposite `step` rejects a nonempty owner with status7 before mutation.
The frame wire remains HTORCH03 / HTORSP03; commands60..69 are still reserved
and rejected by this standalone component.

The qualification controller consists of real checked HIA calls into complete
old and new runtimes. It never constructs an owner from a serialized address.
Two executed cases,31 frames and595,232 compared owner bytes demonstrate that
both wrong-profile calls return the identical pointer, length and contents.
The new-to-old case first clears a real gradient to a null address. The owner
then resumes successfully in its original implementation. Cross-profile
`close` unmaps the allocation while the process remains alive; a returned
live-empty owner can initialize the other version.

## Source and build

| Item | SHA256 |
| --- | --- |
| Materialization05 baseline core | `9feab47afea6723284e15a2ad6c9071dc8c5bdc27b572d2eef4ab4628ed1aea9` |
| Nullable-gradient core | `cebd8b474c2604c635bdc39a8a4e75721ff04a0544c5cbff9b706d5e5c85b126` |
| Native session ELF | `e84156898c42a69ca92ecc58d7e086098d356cbb9ddf6fe3e8c69b09287cfcad` |
| Unchanged owned entry source | `993651c88e69be93dea49ef473b8409b9f609e5b226c6fcec10de230ef4af99e` |
| ABI3 module compiler | `b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f` |
| ABI3 session linker | `aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16` |
| Qualification-only cross-state ELF | `b8731ce847b0f3ccabc129aef57ce3604f6ee2a6c3601d765858a47a05c10a5e` |

The core contains262 functions,3,154 blocks and12,168 instructions. It compiles
as five core modules plus the unchanged three-function owned entry. All
per-unit native compiler caps and linker admission checks remain unchanged.
Of261 baseline functions,253 remain byte-identical. `zero_grad_none` is the
only added function. Eight functions change: descriptor, tgrad, zero_grad,
ensure_backward_work, release_backward_scratch, object_command, init_state
and state_header. The original numerical derivative and scalar-arithmetic
bodies remain unchanged. Per-function custody and the focused source diff are
included; the canonical numerical source remains the underlying authority.

```sh
python3 -I -S -B tools/gradient_none/build.py \
  --repository REPO \
  --source-root PINNED_HIA \
  --baseline MATERIALIZATION05_BUILD \
  --partial-baseline PARTIAL04_BUILD \
  --tensor-objects TENSOR_OBJECTS09_BUILD \
  --compiler ABI3_MODULE_COMPILER \
  --linker ABI3_SESSION_LINKER \
  --output FRESH_NULLABLE_BUILD
```

The producer checks the selected source, authoring, actual compiled units,
object bytes and tool identities before and after the build. The final core
and executable must equal the profile hashes. Python only generates checked
HIA, transports requests and evaluates independent test assertions.

## Executed native qualification

`test_nullable.py` passes six groups,179 interactive frames,131,101 exact
gradient-word comparisons,10,638,464 bytes of full-owner rollback comparison,
and261 committed-node invariant checks. These include all four backward
entrypoints, actual gradient Hold replacement, signed-zero rematerialization,
zero-length Storage ownership, reserved/typed/leaf admission failures,
clear-plus-later-error rollback, partial execution cleanup and real allocation
failure under RLIMIT_AS32MiB followed by successful recovery.

The unchanged actual-reference replayers also pass all13 signed-zero and16
partial-backward cases on this image. The existing8 optimizer/presence cases
retain their qualified parameter values, absolute parameter counters and None
behavior, while still recording the existing8 momentum-state counter and65
explicit-versus-lazy state creation observations as open. These predecessor
references are regressions; they do not replace the new command70 reference.

The separate actual10-case gradient-clear report has now been independently
replayed on this exact image: all10 cases qualified, with0 unexpected failures
over780 interactive frames. There were2,669 exact integer, metadata and gradient
word comparisons, and288 comparisons under the unchanged Adam tolerance policy.
Six Adam words differed in raw bits while remaining within that pre-existing
tolerance; they are recorded rather than called bit-exact. Gradient words,
including signed zero, and parameter Version/presence checks are exact.
The new report separately retains11 momentum-state Version observations,
9 explicit-state-creation observations and30 missing native step-Tensor
observations as open. These are repeated observations in this specific suite,
not additional API counts or replacements for older reports.

`replay_grad_clear.py` is the independent executed source. Its actual reference
is `evidence/actual-torch-grad-clear-observations.json`, SHA256
`ddd585f50034005d8333b24e68c4e3aef675f45bb3c273d973e3525195ac77cf`.
The actual pinned wheel run passed all eleven provenance checks. The native
replay imports only the explicitly hash-bound wire/observer helpers; it does
not rerun Torch locally or generate expected values from the native result.

```sh
python3 -I -S -B tools/gradient_none/test_nullable.py \
  --binary FRESH_NULLABLE_BUILD/tensor-nullable-session \
  --expected-sha256 e84156898c42a69ca92ecc58d7e086098d356cbb9ddf6fe3e8c69b09287cfcad \
  --source FRESH_NULLABLE_BUILD/runtime.hepta \
  --expected-source-sha256 cebd8b474c2604c635bdc39a8a4e75721ff04a0544c5cbff9b706d5e5c85b126 \
  --protocol tools/tensor_objects/protocol.py \
  --output FRESH_NATIVE_TEST_OUTPUT
```

Build and execute the legal-owner state barrier separately:

```sh
python3 -I -S -B tools/gradient_none/build_state_handoff.py \
  --old MATERIALIZATION05_BUILD --new FRESH_NULLABLE_BUILD \
  --compiler ABI3_MODULE_COMPILER --linker ABI3_SESSION_LINKER \
  --output FRESH_HANDOFF_BUILD

python3 -I -S -B tools/gradient_none/test_state_handoff.py \
  --executable FRESH_HANDOFF_BUILD/state-handoff-session \
  --build-receipt FRESH_HANDOFF_BUILD/report.json \
  --protocol tools/tensor_objects/protocol.py \
  --output FRESH_HANDOFF_TEST_OUTPUT
```

## Upstream reference and remaining object scope

The pinned [`Optimizer.zero_grad`](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/optim/optimizer.py#L1066-L1126)
distinguishes removing an existing gradient reference from zeroing its data,
and does not create a gradient when it is absent. The
[`AccumulateGrad` implementation](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/functions/accumulate_grad.h#L187-L252)
distinguishes first materialization from later accumulation. Source custody
and the separate actual10-case observation report are preserved with their
original identities; no local Torch execution is fabricated.

The wire exposes gradient values and presence, not a public `.grad` Tensor
identity or Version. Independently held external gradient aliases therefore
remain a separate API gap. True Optimizer-owned state, fresh momentum Version0
and later two-operation momentum mutation are also separate work. No Version
of a caller-owned state Tensor is reset here. Command70 inherits the existing
per-parameter guard: it accepts differentiable leaves, not every Tensor that
a broader Python frontend might accept. Full PyTorch coverage, higher-order
autograd, additional dtypes/backends and the original2,590-schema target remain
open beyond this component.
