# Retained nonleaf gradients — candidate contract

This is a new native candidate built from the immutable combined05 runtime
`1fed8ea7ba5bf755a9df9ca1e61d34cfd4712cc2fd66e0728493091c81c24dae`.
Its predecessor and all existing references remain unchanged.
The implementation and its qualifications are separate: the fixed hosted
reference is `ca8b5d3a7c679dcc5ad9a3d7b2bbd38f664c90e6474cf3e3c28496d5b716d1b3`,
8 families / 12 variants, observed under the existing eleven source checks.

## Wire

The existing outer `HTOWNR01` / `HTOWNS01`, body `HTORCH03`, response
`HTORSP03` version 3 and eight-u64 command records stay in place. Commands
91/92, 71–79, 90, 95 and all other existing fields keep their contracts.

| Command | Words after opcode | Successful result | Rejection |
| --- | --- | --- | --- |
| 93 — retain_grad | Tensor, six zero words | scalar 1, no payload | invalid/stale/wrong-kind public Tensor: 1; requires_grad false: 5; nonzero reserved: 7 |
| 94 — retains_grad | Tensor, six zero words | scalar flag 0/1, no payload | invalid/stale/wrong-kind public Tensor: 1; nonzero reserved: 7 |

Both commands resolve the existing request-local FE reference in the Tensor
field. Their scalar results are not Tensor capabilities. Enabling retention
on a requiring leaf is a no-op, and repeated enabling on a retained nonleaf is
idempotent. A Tensor's flag and data Version do not change on querying. No
disable operation is introduced. A retained flag preserved by a separately
supported detach operation is not forcibly cleared merely because the Tensor
becomes a leaf.

## Private format and ownership

The new owner uses `HTRET001` and object-state `HTORET01`. The previous
`HTGAD002` / `HTOGAD02` owner is rejected by the new step, and conversely.
The real descriptor remains 576 bytes. Offsets 88, 312, 320, 384, 392 and 400
retain their existing AccumulateGrad, Optimizer and gradient-binding roles.

Kind-1 Tensor record word +56 gains two explicitly distinguished flags:

* bit 0 is retains_grad. It is part of the real Tensor identity, adds no
  strong reference, starts clear on every newly created Tensor, and survives
  re-publication of the same privately held Tensor.
* bit 1 is a private, transient retained-result reservation. It is coupled to
  one actual node reference and keeps that new Tensor row alive during the
  admitted backward. It is never user-controlled, never published, and must
  be cleared and released before any successful or structured-error reply.

Committed audits accept only word values 0 or 1. The backward reservation
table belongs to an actual temporary private Hold. It stores checked internal
node indices, never native pointers or fabricated owned resources. Each
reserved result owns its real Storage and Version through the ordinary
descriptor constructor. Unused reservations are released on every structured
exit. An ordinary admission failure discards the entire trial owner.

The installed retain behavior is weak with respect to its Tensor. Metadata
that survives after the public and private strong roots have vanished does
not recreate a Tensor or gradient. An Optimizer's existing private root can
keep the real nonleaf alive, including its graph and gradient binding.

## Backward and gradient objects

For each reachable live retained nonleaf, admission reserves a fresh result
Tensor/Storage/Version before the execution phase. No gradient binding is
replaced during admission. On execution, the hook runs before the node's
Saved-data and version checks, following the fixed upstream engine order.

With no existing gradient, the hook copies the incoming logical values into
the fresh contiguous Tensor. With an existing gradient, it computes a fresh
out-of-place sum, with layout selected from the existing gradient and the
actual incoming gradient. The old gradient's Tensor, Storage, Version and
values stay intact for external aliases and Saved dependencies. Fresh result
Version 0 is produced by allocation and writes, not by resetting an old
Version. The leaf AccumulateGrad copy/steal path is unchanged.

Command 58 commits a retain update even if that node subsequently fails with
Saved/version error 10/6. Its completed-node counter still counts completed
node applications, not a prehook alone. Unexecuted reservations are cleaned
before the structured-error response. Commands 7/29/57 retain their existing
whole-frame atomic contract; they execute retain hooks on their successful
paths without weakening the existing error behavior.

Existing 91/92 expose the same actual gradient object for nonleaves. The
legacy zero/None/read helpers extend their nonleaf admission specifically to
retained parameters. Owned SGD/Adam/AdamW admission becomes leaf-or-retained;
the parameter is not detached or converted into a leaf. Momentum follows the
real gradient layout, while Adam moments continue following the parameter
layout. Numerical optimizer kernels and state Versions are not changed.

## Scope and evidence

Primary sources are pinned at
`31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`:
`torch/csrc/autograd/variable.cpp` (retain hook), `engine.cpp` (prehook and
Saved release order), and `torch/optim/optimizer.py` (admission/clearing).
Their exact hashes and the immutable reference probe are recorded in
`retained-nonleaf-review/source-custody.json` and the independent source review.

The candidate targets the existing dense CPU binary32, rank 0–4,
first-order, single-output surface. Sparse gradients, custom hooks,
create_graph and general in-place graph rebasing remain outside this
candidate. Implementation/compile results alone are not Torch parity.
