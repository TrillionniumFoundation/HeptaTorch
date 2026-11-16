# Retained nonleaf gradients: source-bound next component

## Finding and current evidence

The original fixed Optimizer reference, SHA256
`7ad0a2e718300a33c91a9212f14ada3c0c7997be7fc77f64bb557c2ecd858160`,
observes in case 14 that an ordinary nonleaf parameter is refused and the same
Tensor is accepted after `retain_grad()`. That case does not execute backward
or a step on the retained nonleaf. It cannot establish its gradient layout,
Version, alias replacement, exception side effects or lifetime.

The current public-gradient runtime provides real `.grad` getter/setter
bindings. It does not implement nonleaf gradient retention. In particular,
being able to assign and read a nonleaf `.grad` is not evidence that backward
will retain its incoming gradient. The former 24-case reports remain frozen;
the newer additive qualifications are accounted for separately.

The new fixed reference-only probe is
`pytorch_retained_nonleaf_probe.py`. It has **8 families / 12 variants** and is
currently **PREPARED_NOT_RUN**. No local Torch results or native compatibility
are claimed. It imports only the already pinned oracle support when actually
executed and preserves all eleven environment/source checks.

## Exact upstream contract

All source references below use commit
`31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`; copied files are bound in
`source-custody.json` with both SHA256 and verified Git blob SHA1.

| Relationship | Required behavior | Primary source |
| --- | --- | --- |
| Admission | `requires_grad=False` rejects before the leaf check. A requiring leaf is a no-op. An already retained nonleaf is a no-op. Enabling retention changes neither Tensor/Storage identity nor its data Version. | [`VariableHooks::retain_grad`, lines 563–604](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/variable.cpp#L563-L604) |
| Registration lifetime | The installed retain hook captures a **weak** TensorImpl reference. It performs no retention if that Tensor has expired or the incoming gradient is undefined. This flag is not a new strong Tensor root. | [`variable.cpp`, lines 581–600](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/variable.cpp#L581-L600) |
| First dense gradient | With no previous `.grad`, it clones the incoming gradient using **Contiguous** memory format. The leaf AccumulateGrad steal/layout-contract path does not apply. | [`variable.cpp`, lines 587–593](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/variable.cpp#L587-L593) |
| Later gradient | With an existing `.grad`, it assigns the result of an **out-of-place** old-gradient plus incoming-gradient operation. It does not mutate the old gradient Tensor with `add_`. External aliases and Saved references to the old gradient therefore require independent lifetime handling. | [`variable.cpp`, lines 594–595](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/variable.cpp#L594-L595) |
| Ordering | Tensor hooks run first, then retain hooks, before the node's `apply`. Consequently, a retain update can happen before that node raises while unpacking Saved tensors. Successful-node counters must still count only completed node applications. | [`engine.cpp`, lines 906–914](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/engine.cpp#L906-L914), [lines 1114–1164](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/engine.cpp#L1114-L1164) |
| Saved release | Default backward releases a node's Saved values only after its apply/validation/post-hook phase returns successfully. Retained `.grad` ownership is separate from Saved ownership. | [`engine.cpp`, lines 1244–1248](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/engine.cpp#L1244-L1248) |
| Optimizer membership | Ordinary non-differentiable Optimizer admission permits a leaf **or** a Tensor whose `retains_grad` property is true. It does not convert the retained Tensor into a leaf or detach its graph. | [`Optimizer.add_param_group`, lines 1188–1197](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/optim/optimizer.py#L1188-L1197) |
| Optimizer gradient clearing | `zero_grad(True)` drops the gradient reference; `False` changes the existing gradient in place. Neither operation is specified as leaf-only. State and skip behavior depend on actual gradient presence. | [`Optimizer.zero_grad`, lines 1066–1114](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/optim/optimizer.py#L1066-L1114) |

Fresh-result Version values and actual layouts for the first clone, later add,
empty tensors and assigned strided gradients will be taken from the new real
reference. They must not be approximated by resetting an existing public
Version counter. Likewise, Python wrapper Weak lifetime is recorded separately
from Storage Weak lifetime; neither is a claim about matching internal C++
reference counts.

## Minimal actual-observation matrix

| Family | Variants | Concrete distinction |
| --- | ---: | --- |
| Admission and idempotence | 1 | Requiring leaf no-op; nonrequiring reject; nonleaf enable twice; Optimizer before/after acceptance; identity/Version unchanged |
| Replacement, clearing and a Saved old-gradient alias | 1 | First negative-zero-containing G; later out-of-place replacement; the held old G remains valid for its independent Saved graph; clearing then fresh materialization |
| Transposed incoming and assigned gradient | 1 | First retained G must be Contiguous even for a dense transposed seed; subsequent assigned-G addition records the real binary layout and fresh identity |
| Empty retained gradient | 1 | A defined zero-element G is distinct from None; repeated retain still records actual object/Storage/Version and layout |
| Real owned optimizer state | 3 | SGD / Adam / AdamW: None skip, first materialization, lazy state, zero False, zero True, skip with existing state, resume |
| Own Saved-version error | 1 | Retain prehook updates q.grad before q's derivative fails; retry records another retained replacement while leaf grad remains uncomputed |
| Own released-Saved error | 1 | A successful default backward is followed by a failing retry; retained G is observed before/after that failure |
| Weak versus strong roots | 3 | Public nonleaf gone before backward, with/without a held prior G; separately an Optimizer privately keeps the real nonleaf, its graph and G until drop |

No custom Python autograd hook is registered. Only primitive snapshots are
retained by the observer. Held seed/G aliases and the Saved old-gradient guard
are explicit fixed inputs. No native result is used to create expected data.

## Proposed native interface and implementation obligations

These are design proposals, not implemented operations or reserved command IDs.
Two commands would suffice: `retain_grad(T)` returning a scalar/void result and
`retains_grad(T)` returning a boolean. A requiring leaf remains false after the
no-op. The enable operation takes no disable flag because the upstream API has
no corresponding `retain_grad(False)` operation. Existing `.grad` getter,
setter, presence and optimizer interfaces should work with the same real
Tensor identity.

The implementation must preserve the following distinctions:

1. **Flag versus ownership.** Store an explicit per-Tensor retention flag in a
   new, versioned private state format. Do not reuse an occupied transient or
   ownership field. Enabling the flag adds no strong reference. When metadata
   survives after its Tensor row is collected, the hook must behave like an
   expired weak reference. A privately held Optimizer parameter has a live
   Tensor row even after its public token closes, so it must still retain G.
2. **Retain hook versus leaf accumulation.** Use the incoming gradient's logical
   values to create a fresh contiguous first G. For an existing G, produce a
   fresh no-grad sum using the old G and actual incoming layouts, then transfer
   the binding. Do not change the old G's Storage, Version, values or identity.
   The current leaf accumulation path and its steal/layout policy stay separate.
3. **Order before per-node Saved checking.** In structured partial backward,
   the retain update belongs before the existing `partial_check_node` call,
   not merely inside `gradient_apply`, which is currently reached after that
   check. If apply fails, the retained update can commit even though the
   successful-node count does not advance. Ordinary frame admission/rollback
   remains its separately documented protocol contract.
4. **Reserve before execution.** Extend bounded backward admission to reserve
   any fresh retained-G node/Storage/Version needed by each reachable live
   retained Tensor. Reserve capacity without publishing a G or replacing an
   existing binding. The prehook transfers the reservation on execution;
   scratch cleanup releases unexecuted reservations on both success and
   structured error. This avoids accidentally treating an allocation failure
   after completed node effects as an ordinary whole-frame retry.
5. **Real Optimizer roots.** Extend admission from leaf-only to leaf-or-retained
   while preserving the nonleaf opcode, parent edges, Saved records and
   incarnation. The actual `.grad` binding is the numerical source. Existing
   SGD momentum allocation must continue following G's layout; Adam moments
   must continue following the parameter layout. No state Version reset is
   permitted. No numerical kernel change is required merely to admit nonleafs.
6. **Release and reset.** Keep the current strong Tensor references from
   Optimizer and gradient bindings. The public-gradient foundation already
   removed the old reset restriction that allowed only privately retained
   leaves, so future changes must use the final 576-byte body rather than
   copying that obsolete restriction from the earlier 384-byte authoring.
   Still qualify reset/growth/drop with a real nonleaf Optimizer root and its
   complete parent/Saved graph. A retained G alias can outlive both the
   parameter and Optimizer.
7. **Fresh object flags.** Clone, detach and newly constructed views must not
   inherit another Tensor object's retention flag merely because they share
   Storage or Version. Re-publication of the same private Tensor must retain
   its flag. New state tags require legal typed-owner handoff refusal tests
   against the predecessor, not reinterpretation of its bytes.

The initial implementation target is the current dense CPU binary32,
first-order, single-output numerical surface. Sparse gradients, custom hooks,
create_graph, functional autograd input capture and nonleaf inplace graph
rebasing remain separate requirements. The full PyTorch goal is unchanged.
