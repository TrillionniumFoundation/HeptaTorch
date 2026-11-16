# CPU distributions wire profile 03

The unchanged owned session frame uses `HTOWNR01` requests and `HTOWNS01` replies, each followed by an unsigned 64-bit body length. The application envelope remains `HTORCH03` / `HTORSP03`; each command has eight unsigned 64-bit words. All unused words must be zero. Typed Tensor/Generator handles and earlier-command references retain the predecessor's checks.

The internal owned state uses prefix magic **HTDIST01** and backing-store magic **HTODIST1**. It keeps seven 64-byte object kinds and pool base `144 + 488 * slots`. It is intentionally incompatible with the frozen predecessor and with later Optimizer kind-8 state. Generator kind 7 still owns one private Hold containing the complete 5,056-byte MT state; no native owner descriptor is copied into the pool. Header fields 112/120 and node fields 312/320 are unchanged.

| Opcode | Fields after opcode | Result |
| ---: | --- | --- |
| 80 | Tensor, Generator, probability binary64 bits, no_grad | Same Tensor handle; scalar Bernoulli in place |
| 81 | Tensor, Generator, probability Tensor, no_grad | Same Tensor handle; float32 Tensor-p Bernoulli in place |
| 82 | probability Tensor, Generator, no_grad | New contiguous float32 Bernoulli Tensor |
| 83 | Tensor, Generator, probability binary64 bits, training, no_grad | Ordinary functional dropout Tensor, or the exact input handle on an identity path |
| 84 | Tensor, Generator, probability binary64 bits, training, no_grad | Ordinary in-place dropout, with the scoped no-grad contract |
| 85 | differentiable functional-dropout result Tensor | A new public Tensor alias of its live Saved noise, sharing Storage and Version |
| 86–89 | Reserved | Status 4 in this frozen profile |

`training` and `no_grad` are exactly 0 or 1. A binary64 scalar probability is passed as its unsigned IEEE bit pattern; it must be in [0,1], with both signed zeros accepted. It is not truncated through binary32. Generator and Tensor handles are validated even for no-draw paths.

Commands 80/81 increment shared Version once on successful execution, including empty tensors. They do not skip consumption at probabilities 0 or 1 for nonempty tensors. Command 81 reads probability data at each iteration, permits broadcasting and the source's accepted overlapping/alias layouts, and bypasses probability shape/value checks for an empty output after checking the handle.

Functional command 82 uses graph opcode 41 with a zero derivative, without a Saved probability-value dependency. The derivative produces positive zero rather than multiplying incoming gradients by zero. Commands 83/84 validate probability first. Probability zero, evaluation mode and empty tensors return the input unchanged with no Version increment and no draw. Nonempty probability-one dropout performs actual float32 multiplication by zero and consumes no RNG. Intermediate dropout consumes two MT32 words per output element. Command 84 increments Version once when it performs a write.

Functional differentiable dropout records a genuine binary-multiply graph with the noise Saved at Version 2. Node field 176 contains marker 1146244944 only on these dropout multiply nodes. Command 85 checks this marker and the live Saved Version before creating an alias; it does not expose a private descriptor or bypass a released Saved check. New nodes keep all established gradient-presence/staging rules.

The per-operation numerical work cap is **1,000,000 elements**. Tensor construction still has the predecessor's per-dimension limit 262,144, rank limit four, total-numel limit 16,777,216, nonnegative-stride and bounded owned-pool limits. A legal million-element fixture therefore uses shape (1000,1000), not an out-of-profile one-dimensional shape. An empty dropout returns before the numerical work cap because it has no work.

Status meanings are unchanged: 1 invalid/stale/private/wrong-kind handle; 2 incompatible shape or refused overlap; 3 capacity/work/allocation/output limit; 4 unsupported opcode; 5 scoped grad-mode or Saved-access misuse; 6 Saved Version error; 7 scalar/flag/state-contract error; 8 Version/generation overflow; 9 internal invariant; 10 backward through released Saved state; 100 malformed payload/envelope. No failure becomes a successful empty Tensor.

Commands 80–85 retain ordinary **whole-frame rollback**. PyTorch's late invalid Tensor-probability and in-place-leaf dropout errors may retain a prefix of writes/RNG consumption; four such fixed observations are explicitly open here. A subsequent ordinary error also rolls back an earlier successful random command. Existing command 58 retains its separate, documented backward-error prefix-commit/suffix-stop behavior; command 57 remains atomic. Future distribution partial-commit commands require a distinct profile and do not reinterpret this one.

This is a borrowed/owned producer-state contract, not an arbitrary external byte-state validator or raw-pointer ABI. Qualified private fault injection and the two-profile handoff harness are separate test interfaces.
