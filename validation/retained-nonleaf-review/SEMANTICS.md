# Fixed actual retained-gradient semantics

This is an independent reading of the **executed** pinned Torch reference,
not a model-generated oracle. The complete original observation is
`actual-01/torch-retained-nonleaf.json` (722,720 bytes, SHA-256
`ca8b5d3a7c679dcc5ad9a3d7b2bbd38f664c90e6474cf3e3c28496d5b716d1b3`).
Its unchanged probe is `7cdb53f7a95c48acd1b85d93d90c693d4caa97ada72a54a4021bb616d098fcba`.
The file records eight families, twelve actual variants, no probe errors,
eleven successful provenance checks, and the Boolean
`source_files_unchanged=true`. The original reference-preparation handoff
remains a historical `NOT_RUN` artifact; this observed record is additive.

The reader validates those fields before extracting
`actual-01/semantic-observations.json`. It did not execute Torch. Native
execution, if present, has a separate source/image-bound replay report.

## Twelve observed variants

The pointers below address the unchanged original JSON. The listed facts are
supported by the complete before/after snapshots, including logical binary32
words, layout, Storage size, Tensor/Storage identity and absolute Version.

| Variant | Original JSON pointer | Observed behavior |
| --- | --- | --- |
| Admission, leaf no-op, idempotence | `/rows/0/observation` | `retain_grad()` on a requiring leaf succeeds with `retains_grad=false`. A nonrequiring Tensor raises. A requiring nonleaf is rejected by ordinary SGD before retain, then accepted after retain. Repeated retain preserves the same Tensor, Storage, values and Version. |
| Replacement, clear, old Saved alias | `/rows/1/observation` | The first retained G has words `[-0,2]` and Version 0. The next accumulation produces a different Tensor and Storage with `[1,+0]`, also Version 0. The held first G remains unchanged, so a Saved-data graph using it succeeds. Setting `.grad=None` clears the binding; a subsequent first materialization preserves both negative-zero words in a new G. |
| Transpose and assigned G | `/rows/2/observation` | A `(3,2)` strided incoming seed with stride `(1,3)` yields first retained G stride `(2,1)`: the first copy is contiguous. Assigning a real strided G preserves that Tensor and its `(1,3)` layout. Later accumulation creates a different Tensor/Storage with `(1,3)` stride and Version 0; it does not mutate the assigned alias. |
| Empty G replacement | `/rows/3/observation` | Both backward calls define an empty `(3,0,2)` G with stride `(2,2,1)` and Version 0. The second G is a new Tensor and Storage. Empty incoming stride `(100,17,1)` is not adopted as the first retained G layout. |
| SGD with retained parameter | `/rows/4/observation/variants/0` | None skips state construction and all mutation. Defined G creates a real momentum Tensor at Version 0. A zeroed defined G still participates in decay/state updates; later momentum Versions are 2 and 4. Clearing G to None preserves existing state during the skipped step. Three participating steps advance parameter Version to 1, 2 and 3. |
| Adam with retained parameter | `/rows/4/observation/variants/1` | Lazy `step`, `exp_avg` and `exp_avg_sq` are real Tensors. Across the three participating updates their Versions are respectively `1,2,3`, `1,2,3` and `2,4,6`. None does not create or advance them. Parameter Version is 1, 2 and 3. |
| AdamW with retained parameter | `/rows/4/observation/variants/2` | The same lazy-state and None rules apply. With nonzero decay the parameter Versions after participating steps are 2, 4 and 6; state Versions match the Adam sequence above. |
| Hook before own Saved-version error | `/rows/5/observation` | After changing the scalar coefficient once, the first backward raises a version error but has already committed retained G `[1,1]`, Version 0. Retry reaches the hook again, replaces G by a new Tensor/Storage containing `[2,2]`, Version 0, then raises the same error. Leaf `x.grad` remains None throughout. |
| Hook before own released-Saved error | `/rows/6/observation` | Default backward through `x*x` produces retained G `[2,-.5]` and leaf G `[4,2]`. Retry first replaces retained G by `[4,-1]`, Version 0, then raises released-Saved. The previously accumulated leaf gradient is unchanged by that failed retry. |
| Weak retain, never materialized | `/rows/7/observation/variants/0` | Dropping the public nonleaf releases its Storage immediately even though a child graph remains. Child backward succeeds without recreating the dead nonleaf or its retained G. Leaf accumulation still follows the live graph. |
| Weak retain, held old G | `/rows/7/observation/variants/1` | Dropping the public nonleaf again releases its Storage immediately. The explicitly held old G keeps its own Storage alive and stays unchanged while child backward succeeds. |
| Optimizer-only strong parameter root | `/rows/7/observation/variants/2` | Dropping external `x` and `q` leaves their Storages and the existing leaf G alive because the Optimizer strongly holds the actual retained nonleaf and its graph. An optimizer step still updates it. Dropping the Optimizer releases `x`, `q` and the leaf-G Storage; explicitly held nonleaf G remains alive until its last public alias is dropped. |

For all three optimizer variants, zeroing with `set_to_none=False` mutates the
same G and advances its Version from 0 to 1. `set_to_none=True` removes the
binding while an explicitly held old G stays valid. After rebuilding G, the
new G starts at Version 0. The parameter, state and gradient objects have
independent identities; equal numerical Version counts do not imply shared
Version objects.

## Native replay contract

The new interface enables retain with command 93 and reads the flag with 94.
Requiring leaves are successful no-ops. Nonrequiring tensors are refused with
status 5; invalid/stale Tensor IDs use 1 and nonzero reserved fields use 7.
The strict replay uses the existing explicit partial-backward command 58 to
compare effects committed before version error 6 or released-Saved error 10.
The existing frame-atomic 7/29/57 behavior is a distinct interface contract.

The replay compares logical binary32 words, None, layouts, Storage sizes,
absolute Versions, leaf/gradient-function presence, retain flags, and opaque
identity equality relations. All SGD and gradient words are exact. Only Adam
parameter and moment values retain the already declared tolerance
`rtol=4e-5, atol=5e-6`; the raw words and any differences are still reported.
Step Tensor values, Versions, identities and all metadata remain exact.

The observer reads the native child's memory only to identify actual Tensor
incarnations, leaf role and whether a getter had to publish a temporary public
token. It closes those temporary G/state/parameter tokens immediately after
primitive snapshots. Explicitly held aliases from the fixed recipe remain
held. No owner memory is modified, and no expected value is inferred from a
native result. Dead Torch Tensor addresses may be recycled, so the two weak
nonleaf recipes explicitly retire their known-dead Tensor identity before
observing later allocations.

Python warning objects, wrapper weak-reference lifetimes, exact Python
`grad_fn` class names and exception text are retained as diagnostics. The
native API does not claim those Python object interfaces. The reference did
not install node hooks, so the replay does not invent expected native
completed-node/leaf counts. Full PyTorch compatibility, higher-order
derivatives, custom hooks, sparse/non-CPU/non-binary32 tensors, and in-place
history rebasing remain outside these twelve observations.
