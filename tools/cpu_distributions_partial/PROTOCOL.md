# Explicit operation-error prefix contract

The transport remains `HTOWNR01` / `HTOWNS01` with16-byte framing and at most
1MiB bodies. Request and response bodies remain `HTORCH03` / `HTORSP03`.
This does not make state tags compatible: only `HTDIST02` / `HTODIST2` owned
states belong to this new service. `step(readbuf,mutbuf,buffer)->(buffer,u32)`
and `close(buffer)->()` preserve the existing ABI3 effects/ownership contract.

## Commands

| Code | Parameters after opcode | Meaning |
| --- | --- | --- |
|86|Tensor,Generator,probability Tensor,no_grad,0,0,0|Inplace float32 Tensor-probability Bernoulli with explicit partial error|
|87|Tensor,Generator,probability binary64 bits,training,no_grad,0,0|Ordinary CPU inplace dropout with explicit partial error|

Flags are exact0 or1; reserved words must be0. Probability Tensor broadcasting,
overlap checks, output-first multioperand iteration and per-element alias reads
are inherited from the frozen atomic implementation. No probability snapshot is
created. Scalar dropout probability is exact binary64; validation precedes its
identity/empty fast paths.

On a successful admission, each new operation has a normal record with record
status0, record value equal to the operation-error code, and a32-byte payload:

```text
u64 operation_error
u64 actual_public_Tensor_handle
u64 completed_Tensor_writes
u64 consumed_raw_MT32_words
```

All integers are little-endian. `operation_error=0` means the operation completed.
A nonzero operation error commits earlier commands and this operation's actual
prefix, stops the suffix, and returns a normal frame containing only the records
that executed. It is different from a top-level admission/transaction failure.
The record value is deliberately an error code, **not a Tensor handle**; use
the payload's explicit handle or the original input handle, rather than treating
the operation as a handle-producing request-local reference.

## Late-error behavior

For86, an invalid per-element probability returns operation error7. It retains
the valid prefix's Tensor writes and exactly one MT32 word per completed write.
It consumes nothing for the invalid element. It does not increment Tensor
Version on failure. Success increments Version once, including an empty Tensor;
empty tensors consume no RNG and follow the inherited empty-probability rules.

For87 with nontrivial probability, a real temporary noise Tensor is generated
using scalar-double Bernoulli, scaled, and set to the observed noise Version2.
If the input is a tracked leaf in normal gradient mode, error5 follows noise
generation. A stride0 expanded output can yield operation error2 after the
noise is generated. Both cases retain RNG advancement, perform no input writes,
leave its Version unchanged, and retire the temporary noise before return.
Normal-gradient-mode tracked nonleaf tensors remain an early unsupported
admission. Successful scoped inplace dropout writes the actual strides and
increments Version once. `p=0`, eval, or empty gives identity without Version
or RNG changes; `p=1` consumes no RNG and uses actual floating multiplication
by zero, including IEEE NaN behavior.

Earlier commands can themselves advance Version or RNG. A late error does not
undo those changes and does not add an extra Version increment. Already saved
aliases therefore continue to observe the same Version and the committed data
prefix. This is intentional, and is tested independently of the original four
hosted observations.

## Admission and resource behavior

Invalid flags/reserved fields, stale/wrong-kind handles, unsupported graph
cases, response capacity, work limits, Version overflow, state tags and memory
allocation failures retain ordinary whole-frame rollback. If86/87 succeeds but
a later command produces an ordinary frame error, the entire frame still rolls
back. If86/87 returns an explicit operation error, that suffix is not executed,
including any otherwise oversized response or invalid command in it.

A record needs64 bytes beyond the frame header:32 record bytes plus32 payload
bytes. The first record therefore fits exactly in96 output bytes. Capacity95
refuses before Tensor writes or RNG consumption. Capacity below32 returns0 and
retains the exact owner; the enclosing production transport closes according
to its already-qualified returned-length contract.

The owner is never copied as a native descriptor into a byte arena. State still
contains relative offsets and a single owned pool; Generator records own private
Holds of the exact5056-byte RNG state. The native transactional step returns
exactly one live owner on every normal return. Commands80–85 remain atomic and
unchanged. Commands88/89 remain unsupported; this successor does not claim
new dtype, `native_dropout`, nonleaf inplace-autograd or Optimizer semantics.
