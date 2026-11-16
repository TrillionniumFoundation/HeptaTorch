# Native owned SGD Optimizer profile — implementation contract

This new profile composes the frozen RNG + nullable-gradient core (`490fc144160f6e4440aed001363f86ccff56c335f9781e38b058fc6106b1be09`). It does not change that source/package. This document specifies the candidate being implemented; it is not a test receipt.

The external request/response remains `HTORCH03` / `HTORSP03` over the existing interactive `HTOWNR01` / `HTOWNS01` transport. Internal numeric/backing tags are new: `HTOPT001` / `HTOOPT01`. Tensor, Storage, Version, Weak, Saved, Hold, and Generator retain kinds 1–7; Optimizer is kind 8. Only dense CPU binary32, nondifferentiable, nonfused, non-foreach SGD is admitted in this first implementation. Adam/AdamW object construction is not silently mapped to SGD.

## Commands

Every command is eight little-endian u64 words `[opcode,a,b,c,d,e,f,g]`. Unused words must be zero. Payload offsets are absolute byte offsets in the inner request and may not overlap the command table. Typed handles and same-frame references use the existing checked resolver. A normal command failure rolls back the whole frame; command 58 retains its existing structured partial-backward semantics.

| Opcode | Fields after opcode | Result |
| --- | --- | --- |
| 71 create | algorithm=0, defaults offset, groups offset, 0,0,0,0 | New kind-8 handle; 4-word inventory described below. No momentum Tensor is created. |
| 72 step | Optimizer,0,0,0,0,0,0 | Scalar 1. Ordered groups/occurrences, real gradient-presence checks, native lazy momentum creation/update. This low-level command is frame atomic on error. |
| 73 zero_grad | Optimizer, set_to_none bool,0,0,0,0,0 | Scalar 1. The frontend default is True (`1`); False (`0`) retains an existing gradient. Neither creates a gradient from None. |
| 74 group | Optimizer, group ordinal,0,0,0,0,0 | Number of occurrences; group payload described below. Read-only. |
| 75 parameter | Optimizer, group ordinal, occurrence ordinal,0,0,0,0 | Public handle to the same retained Tensor/gradient binding. Repeated access returns the current public token; a closed token is replaced by a fresh generation and never reactivated. |
| 76 state inventory | Optimizer, group ordinal, occurrence ordinal,0,0,0,0 | Scalar 0/1 for absent/present state; 4-word state inventory. This does not insert a state entry. |
| 77 state Tensor | Optimizer, group ordinal, occurrence ordinal, field=0(momentum),0,0,0 | Actual momentum Tensor handle, or 0 when absent. This does not clone data or reset a Version. |
| 78 drop | Optimizer,0,0,0,0,0,0 | Scalar 1. Releases membership/state-key/state-Tensor references. Independent public Tensor/Saved references survive. |
| 79 add group | Optimizer, single-group offset,0,0,0,0,0 | Scalar 1 plus updated 4-word inventory. Existing state is retained. |

The 4-word creation/add inventory is `[group_count, occurrence_count, unique_parameter_count, duplicate_occurrence_count]`. A frontend can report the upstream duplicate-parameter warning from the final field; the native packet itself does not implement Python warning objects.

The group payload is `[group_ordinal, parameter_count, lr_bits, momentum_bits, dampening_bits, weight_decay_bits, nesterov, maximize]`, then one pair `[stable_Tensor_incarnation, current_public_handle_or_0]` per occurrence. Stable identity is an opaque native incarnation; it is neither a host address nor a Storage identity.

The state inventory is `[parameter_incarnation, field_mask, momentum_incarnation_or_0, current_momentum_public_handle_or_0]`. Bit 0 of `field_mask` means a real momentum Tensor exists. No other field is currently defined. Missing state has zero mask and zero momentum fields. State keys use Tensor identity, so two distinct Tensors sharing Storage have separate state, while repeated occurrences of the same Tensor share one final dictionary entry.

## Configuration payloads

Defaults are exactly six u64 words: `[lr_bits,momentum_bits,dampening_bits,weight_decay_bits,nesterov,maximize]`. Float words contain binary32 bits with zero upper 32 bits; booleans are 0 or 1. Constructor defaults follow the pinned scalar SGD predicates: negative lr, momentum, or weight decay are rejected; Nesterov requires positive momentum and zero dampening. NaN and infinity follow those actual predicates rather than an invented blanket finite-value check.

The groups payload begins with `group_count` and then concatenates groups. Each group starts with eight words `[parameter_count, override_mask, lr_bits,momentum_bits,dampening_bits,weight_decay_bits,nesterov,maximize]`, followed immediately by `parameter_count` u64 public Tensor handles or same-frame result references. Mask bits 0–5 select the supplied override values; unselected words must be zero. A command-79 payload contains one such group without a group-count prefix. Defaults are merged in native HIA. Scalar default-constructor predicates are not reapplied to arbitrary per-group overrides; typed-width/boolean validation still applies.

The top-level group list must be nonempty. An explicitly empty group is accepted. The same Tensor in different groups is rejected before membership changes. Duplicate occurrences inside one group are preserved in order. Limits are 1,024 groups and 4,096 occurrences per Optimizer, additionally bounded by the 1 MiB request and existing owned allocation/table limits. Tensor parameters must be current public leaf Tensors; requires_grad=False is accepted and has no gradient until supported by an explicit API. Nonleaf retain_grad/differentiable optimizer modes remain outside this interface.

## Native object ownership and reset

The kind-8 row owns one private configuration/membership Hold, with bounded relative offsets only. Each occurrence owns a private strong Tensor reference; a realized state dictionary key owns another parameter reference; each actual momentum Tensor has a private state reference. This retains the real Tensor/gradient binding, not only Storage bytes.

Node +312 holds the private strong-reference count. Node +320 holds a monotonic Tensor incarnation from global +120. These have no such meaning in the predecessor; the new internal tags are mandatory. The obsolete release-only +320 Hold consumer is removed. Metadata-edge count +208 remains separate and includes the private references for graph reachability. Existing matmul copy flags +344/+352 and gradient fields +360/+336/+368/+376 retain their meanings.

Graph reset revokes discarded public tokens and releases graph topology while preserving Optimizer-owned parameter/state nodes, their descriptor pages, incarnations, Storage/Version, and gradients. The next descriptor ordinal remains above the greatest preserved node. Ordinary prefix retention keeps its existing explicit contract. Public republishing gives a fresh typed generation after closure. Grow/migrate preserves all relative object bindings; callers never resend hidden state to reconstruct it.

## SGD state and duplicate ordering

None skips parameter data/Version and all state creation/update. A defined zero or empty gradient participates. With momentum zero there is no momentum state entry. First momentum is a real fresh Tensor cloned from the effective gradient, with independent Storage/Version 0 and preserved eligible gradient layout; first-step dampening is not applied. Later updates mutate that same state Tensor with multiply then add, advancing its Version twice, including empty and no-bit-change operations. A parameter advances once per participating occurrence.

For each group, all participating occurrences snapshot the pre-loop momentum presence/identity before updates. On the first duplicated parameter step, each absent snapshot constructs its own first buffer; after ordered updates the final occurrence's buffer becomes the dictionary state. On later steps the occurrences share the preexisting state and mutate it in order. This avoids treating first-occurrence construction as if it were already present when planning the next occurrence.

The current private gradient is stored in logical order. State layout must be derived from the qualified leaf gradient-layout contract, not assumed contiguous for a dense transposed leaf. Public `.grad` Tensor objects and their independently held aliases remain a separate unfinished interface; command 70 semantics are retained without claiming that broader API.

## Explicit limits

This component does not claim full Optimizer API or full PyTorch closure. Python hooks/warnings/state dictionaries/serialization, arbitrary late-error partial optimizer side effects, nonleaf retained gradients, differentiable/capturable/fused/foreach/sparse/complex/mixed-dtype modes, and Adam/AdamW object state are separate work. Existing raw-state optimizer commands remain available with their historical contracts and evidence. Internal reference counts are native ownership obligations, not numerical equality with PyTorch intrusive_ptr use_count.
