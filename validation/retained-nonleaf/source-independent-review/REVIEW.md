# Source-only review of retained nonleaf gradients

This review binds the frozen combined candidate05 source `1fed8ea7ba5bf755a9df9ca1e61d34cfd4712cc2fd66e0728493091c81c24dae`, its576-byte descriptor profile, and the proposed `retained-nonleaf-review/DESIGN.md`. It does not implement retain_grad, execute Torch, or assign new commands or state fields. The new12-variant reference is still awaiting actual hosted observations in this review.

## Supported design choices

The upstream retain hook is weak with respect to its TensorImpl. It runs before the node's derivative and Saved checks, clones the first dense incoming gradient contiguously, and uses an out-of-place addition for a subsequent gradient. These are materially different from the current leaf accumulation path. The proposed separation is correct at the source level.

The original24-case Optimizer observation establishes admission after retain_grad, not the retained numerical or lifetime behavior. Keeping the new probe additive and preserving the original case14 partial scope is appropriate until a real native replay exists.

## Concrete current-runtime obligations

### Nonleaf admission must reach every numerical consumer

The current `tgrad`, `zero_grad` and `zero_grad_none` still explicitly reject any nonzero AD opcode. Allowing retained nonleafs only in `optimizer_fill_group` would therefore admit an object that fails on step or zero_grad. The same issue applies to legacy gradient-range reads if they delegate to `tgrad`. Getter91 and `gradient_binding` already support a real assigned nonleaf gradient; they do not implement automatic retention.

Both SGD and Adam must continue reading the real gradient binding. Their layout policies remain distinct: SGD's first momentum follows the gradient; Adam's moments follow the parameter. No old private-array gradient reader should be restored to bypass the guard.

### A live graph descriptor is not a live Tensor object

Current descriptor304 says that graph metadata is live. Descriptor240 and the checked kind1 row establish that the Tensor object itself remains live. `tensor_row_release` can remove a nonleaf's Tensor row and gradient binding while downstream graph metadata still survives. A retain hook must skip this expired object; checking304 alone would resurrect a gradient for a dead Tensor.

Conversely, Optimizer private references at312 or incoming gradient-binding references at400 can keep a real Tensor row alive after public token closure. Its retained gradient must still update. The flag must not add a new strong Tensor reference. Existing captured AccumulateGrad references at88 remain a separate role.

### There is a possible flag location without changing descriptor size

The kind1 Tensor record's last word at offset56 is currently reserved zero, explicitly checked by `so_audit_records`. It is separate from descriptor offset56. A future versioned object format could use that reserved record word for a retention boolean, provided every constructor, audit, publication and copy path is checked.

This is a candidate layout, not an allocated field. Current `publish_tensor` closes and re-creates the public kind1 row, copying Storage, Version, extent and node mapping but not the reserved word. A flag placed there must be copied when re-publishing the same private Tensor. Fresh clone/detach/view objects must start with their own flag rather than inherit it from shared Storage. Row release must discard the flag with the expired object. Table growth already copies whole records, but the actual future producer must revalidate this property.

The existing descriptor fields cannot be silently reused:408 is a view-related flag;416/424/432/440 are audit/GC/DFS scratch;448–536 own incoming values and cache state;544–568 own the separate physical Preserve-layout reserve.312/320/88 and384/392/400 must retain their current ownership and identity roles.

### Retention must consume the incoming layout, not forward Tensor data

The current node descriptor's ordinary data pointer and strides describe the forward Tensor. Incoming backward values have their own Storage/Version, byte offset and strides at448–520, while160 is a logical-order cache. Calling the ordinary `binary(old_grad, forward_tensor)` would use the wrong data and possibly the wrong layout.

The retain implementation needs an explicit first-order no-grad clone/add path over the actual incoming value. The first result is fresh contiguous Storage with its own Version; a later result uses the actual old-gradient/incoming pointwise layout and fresh Tensor/Storage/Version. It must not reuse leaf `gradient_accumulate`, which may adopt input Storage or mutate an existing gradient. An assigned gradient that itself requires grad must not accidentally make the retained result differentiable merely because the normal binary helper propagates its inputs' requires flags.

### Reservation must not replace a binding before execution

Current `gradient_new` binds a leaf placeholder immediately and chooses the parameter's dense layout. Neither behavior can be reused unchanged for retained nonleafs. Preflight must reserve a fresh result without publishing it or replacing an existing G. The hook performs the actual binding swap only when that node becomes ready.

An existing held G, Saved use of G, and an Optimizer reference to G must continue observing the original object after the swap. Retain-before-release ordering is required for Storage/Version and gradient-binding references. A new per-backward reservation table is possible, but its unpublished Tensor reservations still need genuine strong ownership; a graph metadata count alone is insufficient to protect their Tensor rows. This ownership role must be explicit and audited, not hidden in an unrelated counter.

The original traversal limit is the output's internal ordinal. Reserved new Tensor descriptors are normally allocated above that limit. Cleanup must explicitly release every unexecuted reservation rather than assuming the existing `release_backward_scratch(limit)` loop visits those new descriptors. All exit paths must restore the existing incoming and GC scratch invariants.

### Execution order and error accounting

In the frozen current `partial_execute`, `partial_check_node` precedes `gradient_apply`. The new retain hook must run before that check. Otherwise an own-Saved-version or already-released-Saved failure incorrectly leaves the retained G unchanged. A failed node may have committed its retain prehook without counting as a successfully applied node; the structured completed-node counter must not include that hook as a separate completed node or leaf accumulator.

The old frame-atomic backward commands retain their explicit error contract. The structured command58 path is the place to compare real partial side effects. No all-graph Saved precheck may be reintroduced there. Reserve allocation failures remain admission failures with complete owner rollback; successful hook replacement must not later be silently retried from the old owner because an avoidable allocation was deferred into execution.

### Avoid an overly strong retention-flag invariant

Pinned `VariableTypeManual.cpp`'s `detach_` resets requires-grad, grad_fn, output number and forward-grad state without clearing the other AutogradMeta fields. Its code does not clear `retains_grad_`. Therefore a permanent audit invariant of “retains flag implies nonleaf and requires-grad” is not justified by the source.

The prepared reference explicitly excludes general inplace graph rebasing/detach of the nonleaf parameter. There is nevertheless an existing entry path to consider: Optimizer zero_grad(False) calls the gradient's in-place detach when an assigned gradient is nonleaf. If that assigned G has its own retained flag, a future implementation must either qualify the interaction or preserve an explicit supported-boundary refusal. It must not erase the flag solely to satisfy a guessed invariant. The normal fresh requiring leaf retain call remains a no-op; this does not prove that every historically detached Tensor's flag must be false.

## Minimal resource and failure controls after actual observations

The proposed12 variants already cover first/repeated retention, held aliases, assigned strided G, empty G, owned SGD/Adam/AdamW, own Saved failures and weak-versus-Optimizer lifetime. After those real observations arrive, the native qualification should additionally exercise:

- Public row closure while downstream metadata survives: no new retained binding; a held old G survives only through its own roots.
- Optimizer-held nonleaf through graph reset, owner growth and public re-publication: same incarnation and retention state, preserved graph and Saved dependencies.
- A later failing node after an earlier retained replacement: the reached hook's new G survives, unreachable reservations disappear, and the ready-node counters retain their existing meaning.
- Real resource exhaustion while reserving retained outputs: exact original owner bytes, IDs, graph state, old G and Versions remain unchanged; retry after releasing unrelated resources succeeds.
- Repeated out-of-place replacement and final root release: old externally held gradients remain valid, unheld old gradients expire, and all temporary roots/Storage/Versions return to their expected counts.

These are proposed controls, not executed tests. No native or Torch result is inferred from this document.

## Reference workflow review

The read-only review of `retained-nonleaf-reference-integration/proposal-02/added-reference-steps.yml` found no blocker in the four added steps. They pin the probe, oracle, lock, prepared receipt and independent review; compare the prepare-only output byte-for-byte; execute actual Torch only in the actual step; and emit the full result under `always()`. The accompanying document distinguishes preparation from actual observations and native compatibility, and keeps the original steps and references unchanged. This review did not rerun CI or Torch.

## Sources

All upstream references use commit `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`:

- [`VariableHooks::retain_grad` and `retains_grad`](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/variable.cpp), lines563–612.
- [Engine hook/apply/release ordering](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/engine.cpp), lines906–914,1114–1164,1244–1248.
- [Optimizer zero_grad and membership](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/optim/optimizer.py), lines1066–1114 and1188–1197.
- [In-place detach metadata changes](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/VariableTypeManual.cpp), lines305–330.

`source-review.json` binds the exact local source bytes, current HIA bodies, design/probe and workflow inputs. The frozen production package remains untouched; the whole-PyTorch denominator is2590 and completion remains false.
