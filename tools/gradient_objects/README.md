# Native public gradient Tensor objects and incoming ownership

This component makes a defined gradient an actual Tensor with its own incarnation, layout, Storage and Version. The parameter holds that Tensor through a checked private reference. Reading `.grad` publishes the existing object; assigning `.grad` changes the binding; clearing the binding releases only that reference. An externally held gradient, detach alias, Storage weak reference or Saved record continues to obey its own lifetime.

All Tensor values, object management, reference counts, gradient copies and merges, and Version updates execute in native HIA. Python only composes and compiles HIA, serializes requests, and independently observes the native child. Runtime payloads never supply owner pointers.

The frozen source is `1b18da2cf9791e1b20982f9706434c1359ea194dd983dc6c66eb005f152e77b5`; its ELF is `be18ab877f38a4027a527783db80a1418a137c7d6da91e5bc39e2af8c044cb6b`. There are 355 core functions, 4,675 blocks and 17,450 instructions. Six complete core units, the unchanged 17-function RNG unit and the unchanged four-function owner unit pass the existing ABI3 compiler and linker. No checker limits were raised.

## Source lineage

The sole numerical lineage remains the existing HIA runtime. The published SGD08 source `7c68a96b...` is the common ancestor. An 82-function overlay replaces 42 of its 315 bodies and adds 40 functions; 273 original bodies remain byte-identical. `functions.json` identifies each final function and its exact baseline and final hashes. The current incoming layer separately leaves 305 of the 330 public-gradient-foundation functions unchanged. Scalar numerical kernel arithmetic is preserved; six exact call-target substitutions direct individual parent edges into distinct staging slots.

The source-bound authoring path first re-emits selected predecessor bodies and requires their bytes to match the predecessor, then emits the new HIA. Its original authoring scripts and checked input closure are included for reproducibility. The production build needs the actual SGD08 HIA baseline and this HIA overlay, and compiles every complete target body with the native tools. It does not maintain a second implementation of the original scalar kernels.

The internal formats are `HTGRAD02` and `HTOGRAD2`, with 576-byte descriptors. External packet formats remain `HTORCH03` / `HTORSP03` inside `HTOWNR01` / `HTOWNS01`. The predecessor SGD08 and intermediate public-gradient owner formats are deliberately rejected as nonempty state. Legal checked HIA calls demonstrate both rejection directions, unchanged original owners, resumption through the correct entry, and actual unmapping through cross-format close while the process remains alive.

## What changed

Commands 91 and 92 expose a gradient getter and setter. The getter returns None as zero or a public token for the existing gradient Tensor. Repeated getters preserve its Tensor incarnation, Storage, Version and layout. Republishing a privately retained Tensor after public closure issues a fresh public generation; it does not create a replacement Tensor. Setter validation is atomic, and the assigned Tensor is retained through its real identity rather than copied.

Commands 70 and 73 with `set_to_none=True` remove the binding. Commands 8 and 73 with `set_to_none=False` zero the existing gradient in place and advance its real Version. Qualified detached-gradient behavior follows the observed leaf/nonleaf distinctions. A retained nonleaf view is refused for in-place detach; general splitting of a nonleaf Tensor's autograd identity while downstream consumers remain live is still open.

Backward now represents incoming gradient values with real Storage and Version ownership. A derivative that produces a fresh result owns a fresh Version; a view or transpose derivative can share its seed's Storage and Version. A qualifying first accumulation transfers that ownership into the gradient Tensor. A required layout copy writes a new Storage. Versions therefore follow actual allocation, copy and mutation operations; no Version is reset to force a match.

Every returned parent edge has a separate staging value. This preserves InputBuffer addition order when multiple edges target the same Tensor. The independently observed `2^24 + 1 + 1` example distinguishes branch creation order: the qualified results are respectively `2^24` and `2^24 + 2`, with the corresponding observed Versions. Grouping the two small edges before adding would change the result and is not used.

New Storage/Version reserves are allocated before partial backward begins. Completed leaves and Saved release remain committed after a structured command-58 runtime exception; unused incoming owners and placeholders are released. Admission failures and ordinary frame failures still roll back the complete owner. Legacy backward entries keep their existing graph-retention contract while releasing temporary incoming caches after execution.

Private gradient references are separate from Optimizer roots and captured AccumulateGrad edges. Their GC queue is iterative. A qualified chain of 300 gradient bindings is reclaimed without recursive host or native traversal, and a cycle is explicitly refused. Public handles, Tensor incarnations, Storage, Version and physical allocation identity remain distinct.

## Executed evidence

The genuine hosted references use the pinned Torch 2.15 CPU build and all original eleven provenance checks. The source target is PyTorch commit `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`. No pinned Torch execution was simulated locally. The raw observations, original probe bytes, replay sources and complete request/response archives are retained.

| Separate suite bound to incoming05 | Result |
| --- | --- |
| Original 10 public-gradient observations | 9 exposed recipes qualified; one setter recipe partially represented; zero unexpected failures; 1,036 frames and 5,975 exact comparisons |
| Incoming materialization: 16 families / 25 variants | 22 represented variants qualified; two Python-hook variants and one create_graph variant remain unrepresented; 895 frames and 4,758 exact comparisons |
| Alias mutation and edge order: 6 families / 12 variants | 12 qualified; 1,334 frames, 7,268 exact comparisons and 20,970,720 bytes of whole-owner rollback checks |
| Native lifetime, growth and allocation controls | Five groups passed; 1,110 frames, 131,836 checks and 44,023,024 rollback bytes |
| Typed and same-frame result references | Passed; 26 frames and 12 precise atomic refusals |
| Frozen signed-zero replay | 13 passed; 166 frames, 131 metadata fields and 70 raw binary32 words |
| Frozen partial-backward replay | 16 passed; 606 frames and 1,587 exact comparisons |
| Frozen gradient-presence replay | Eight parameter/presence/skip recipes qualified; 880 frames, 2,088 exact and 180 unchanged-tolerance comparisons |
| Frozen gradient-clear replay | Ten recipes qualified; 780 frames, 2,669 exact and 288 unchanged-tolerance comparisons |
| Complete checked owner handoffs | Four directions across two linked images passed; 100 frames and 2,009,664 bytes of unchanged owner checks, followed by live-process unmapping and reinitialization |

These suite sizes and value counts overlap and are not a count of unique upstream tests. The public-gradient ten-case adapter changes only read-only descriptor addressing. Its 418 fewer counted checks than the earlier report are kind checks performed as assertions by that observer; all original case methods and numerical, identity, Version and lifetime comparisons are inherited unchanged.

The alias-mutation replay deliberately uses fixed literal low-level mutation fixtures: a temporary `.data` alias receives the fixed writes, a real write through the source advances its actual Version once, and the temporary alias is dropped. Each fixture checks that temporary live Storage/Version counts and free bytes return to their prior values. This qualifies sharing and lifetime under actual native writes, not a `Tensor.add_` implementation. Literal arrays are fixed in the replay source rather than copied from expected snapshots. The actual Torch probe itself executes real `add_` calls. Command 95 and its numerical implementation belong to the separate Adam branch and are not claimed here.

The legacy explicit-state optimizer replays retain their preexisting lifecycle differences. The eight-case suite still records 8 state-Version and 65 state-creation observations. The ten-case suite still records 11 state-Version, 9 state-creation and 30 absent-step-Tensor observations, plus six Adam word differences within its unchanged original tolerance. These categories are kept separate, not summed or relabeled as successful owned-state tests.

The native allocation control uses a real 32 MiB address-space limit. The failed backward leaves all 8,563,344 committed owner bytes unchanged. Releasing an unrelated extent permits the same request to succeed, and 131,072 gradient words are then checked exactly. A separate test grows real Tensor data past 1 MiB and descriptors past 255 while preserving private Optimizer and gradient roots across reset.

## Preserved failures and corrections

The first public-gradient candidate's actual command-58 GC failure remains archived, along with its exact source and image identity, requests and responses. A distinct first host replay accidentally selected command 5 instead of the view command; the original report and source correction are separately retained. The native fix drains the real pending release queue after a successful command, including a structured runtime exception that commits prior work.

The intermediate public-gradient candidate passes the original ten recipes but fails seven of the 22 represented incoming variants. It copies every first gradient, producing incorrect Storage relations and Versions. Its immutable 895-frame report is retained. Replaying the later alias/edge reference against that old candidate also exposes nine failing variants and 251 fields; that is explicitly a retrospective old-image replay, not a report relabeled as the new image.

Two authoring-stage refusals in this incoming layer are retained: a non-unique audit insertion anchor and an incorrect assumption about the concat helper call chain. Neither reached native compilation. The final source routes the real `concat_copy -> concat_part` call, including the unchanged forward branch.

The first lifetime-control script incorrectly required a constructor's pre-backward work cache to be empty. The inherited constructor actually owns a valid private Hold. The original failure is archived. The corrected observer verifies the real Hold and pointer pair, all new incoming fields zero at commit, and released work caches on nodes that have executed backward. Numeric inputs and native bytes were unchanged. A separate host diagnostic SyntaxError in the reference-control script is also preserved before its syntax-only correction.

## Build and qualification

Build the actual HIA overlay against a reproduced SGD08 baseline:

```sh
python3 -I -S -B tools/gradient_objects/build.py \
  --baseline build/owned-optimizers \
  --compiler build/stack/native/stack-module-compiler \
  --linker build/stack/native/stack-linker-session \
  --output build/gradient-objects
```

The compiler and linker must match the full pins in `profile.json`. Every generated module and object, the complete source and the final executable must match their frozen hashes. Runtime allocation is the checked native owner/allocator implementation; the build does not inject foreign callbacks or dummy import bodies.

The complete portable qualifier also re-emits the intermediate public-gradient foundation and incoming authoring layers from the pinned HIA source checkout, requires their final image to equal the independent overlay build, runs each separate fixed suite, and builds and executes both complete owner-handoff images:

```sh
python3 -I -S -B tools/gradient_objects/qualify.py \
  --repository . \
  --baseline build/owned-optimizers \
  --source-root /path/to/pinned/hia-toolchain \
  --compiler build/stack/native/stack-module-compiler \
  --linker build/stack/native/stack-linker-session \
  --output build/gradient-objects-qualification
```

Historical absolute paths inside receipts identify development inputs only. Rebuild commands use explicit arguments and repository-relative package paths. Complete temporary build/test evidence is archived with a member-by-member size, SHA256 and byte-equality verification; the final handoff does not depend on the continued existence of a temporary filesystem.

## Remaining scope

This is the SGD08-based public-gradient component. It has not yet been composed or qualified in the same owner with the newer real Adam/AdamW object implementation and the 25 distribution functions. That combination is a subsequent native implementation and qualification task, not inferred from separate passing branches.

The represented runtime remains CPU binary32 with rank at most four and its explicit dimensional, element, descriptor, packet, allocator and work-budget bounds. General hooks, create_graph/higher-order gradients, retain_grad for nonleaf parameters, arbitrary autograd identity splitting, ownership cycles, sparse/complex/device/dtype setter behavior, and unqualified incoming allocation/layout/use-count paths remain open. In particular, arbitrary overlapping assigned-gradient layouts are not claimed qualified by these dense and explicitly strided examples.

The original 2,590-entry upstream denominator and `full_rewrite_complete=false` remain unchanged.
