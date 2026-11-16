# Independent Adam and distribution composition review

The pinned joint native candidate passed the scoped source audit and all six
new cross-feature native contracts. This review does not add API coverage or
replace the original 183 distribution observations or 24 optimizer observations.
Those frozen inputs remain unchanged; their full replays are separately owned
by the producer's qualification and are not counted again here.

| Identity | SHA-256 |
| --- | --- |
| Joint HIA source | `5d3a3c047d024a3880d257a317a04c2166a2e87bbc6b248bef62649a88546873` |
| Joint native ELF | `3fc59ab7ed9c32dd75ff440ef9bfbec31756f50edd77715d665bdac098cc3f22` |
| Independent executable test script | `bba48f2b6f2c2f87b9626bf6cc084a7ffef9a63de8c8d8fb07bb09252b8838f6` |
| Final native report | `1b98056eb86c4755debcd5f8429c04001da46cc30cc17c23879fd4884d16c575` |
| Source audit report | `467abf38020369b9962a94a39dd3782bfaa7719263e800bc4d55207c941ef48d` |

## What actually ran

Every native response arrived while stdin remained open in the same process.
All numerical execution, Tensor/Storage/Version mutations, optimizer state
creation, graph traversal, RNG consumption and lifetime changes ran in the native
HIA ELF. The test imports only the frozen wire codec and the own-child transport
observer. CPython's MT engine and exact dyadic arithmetic supply independent
test expectations; they are not a production runtime.

| Native contract | Frames | Result |
| --- | ---: | --- |
| Bernoulli opcode41, captured edge, private Optimizer roots, reset and final AccumulateGrad release | 63 | PASS |
| Dropout Saved noise, actual Adam state aliases, public Saved references, reset and final strong release | 53 | PASS |
| Adam step followed by command86 writing the actual first-moment Tensor, explicit partial commit, stopped suffix and ordinary-error rollback | 39 | PASS |
| AdamW state creation followed by command87 tracked-leaf and expanded-moment-view late errors | 57 | PASS |
| Adam: four actual dropout → backward → step → clear/reset updates | 125 | PASS |
| AdamW: the same four-update lifecycle, including decoupled decay | 125 | PASS |

Totals are **462 interactive frames**, **334 exact assertions**, **249 raw
binary32 value comparisons**, **22 complete 5,056-byte Generator state
comparisons**, and **4,892,928 bytes of complete-owner rollback comparisons**.
The assertion and word counts are separate counters, not additional cases.

The Bernoulli test proves that a nonfinite incoming seed still produces positive
zero gradients, with no Saved probability-value dependency. Changing the leaf
requires-grad flag does not invent or erase the captured forward edge. Public
Tensor closure and graph reset do not destroy the Optimizer's actual parameter
or its three actual state Tensors; re-publication produces fresh public tokens
with unchanged Storage and Version identities. Conversely, after the Optimizer
is dropped, an extant graph accumulator can keep the parameter alive until its
last graph root is released.

The partial-operation tests use genuine Adam state as a random-operation target.
Command86 retains the valid prefix of first-moment writes without adding a
Version increment for the failed operation. A Saved reference made after the
preceding Adam step therefore sees those changed values with the expected
counter. Command87 retains the draws made while producing a temporary noise
Tensor, then reclaims that temporary on either the tracked-leaf or overlapping
view error. Already completed Adam/AdamW parameter and state updates remain
committed; suffix mutations and optimizer deletion do not execute. An ordinary
admission error, or a later ordinary error after a successful distribution
operation, still restores the entire previous owner including Generator bytes.

The short training expectations use `beta1=beta2=0`, `eps=2`, `lr=1/8`, and
dropout probability1/2. Each incoming gradient is0 or2, so the adaptive update
is exactly0 or1/16; AdamW additionally multiplies by31/32. These values allow
strict binary32 comparison through four updates without introducing a new
tolerance. They are new finite source/model contracts, not new hosted PyTorch
observations. The independent MT initialization/serialization was first checked
against five unchanged states from the actual 183-case report at draw counts
0,1,2,4 and14.

## Source review

The review parses actual HIA function bytes without importing or executing the
producer. The joint source retains345 of352 Adam function bodies exactly. It
copies25 real distribution bodies exactly, preserves the old backward
dispatcher as a renamed real function, and adds a wrapper for graph opcode41.
The only `gradient_direct` change extends its recognized upper opcode from40
to41. Four state-tag functions change exact tag literals. Removing the added
distribution branch and reversing the suffix-stop condition restores the Adam
entry's exact normalized authoring text.

All complete function bodies in the actual production units match this
composed source. Every external declaration matches the real callee signature.
The RNG and owned step/close modules remain byte-identical to both predecessors.
The audit does not claim aggregate Core admission or substitute host checking
for the real native compiler and ABI3 linker.

The reviewed generic graph format has two parent slots; opcode41 uses its first
slot and a zero second slot. Existing capture and release rules account for
its metadata edge and leaf accumulator. There is no Saved value policy for41.
Dropout continues to use the existing multiply graph and its Saved-noise marker
at descriptor176, separate from Optimizer references312, incarnation320 and
accumulator88. No object-table or allocator helper was silently replaced by
the seven-kind distribution predecessor.

## Reproduce without modifying a frozen input

The following paths are examples within the repository's build layout. Each
output directory must be new. The scripts pin the exact source/image and the
original observed reports before execution, and verify source custody again
afterward.

```bash
python -B tests/adam_distributions_review/source_audit.py \
  --adam-build build/owned-adam \
  --distribution-build build/cpu-distributions-partial \
  --build build/adam-distributions \
  --output build/adam-distributions-source-review

python -B tests/adam_distributions_review/test_joint_contracts.py \
  --build build/adam-distributions \
  --protocol tools/cpu_distributions/protocol.py \
  --distribution-observations tools/cpu_distributions/evidence/torch-bernoulli-dropout.json \
  --adam-observations validation/adam-object-review/torch-optimizer-objects.json \
  --output build/adam-distributions-joint-review
```

The first audit assumes the production build retains its checked
`overlay-authoring.py`, complete runtime, `partition.json` and actual unit
sources. It never invokes another compiler or invents a stub callee.

## Preserved first attempts

The first native run completed228 frames: the four lifetime/partial-commit
cases passed, while both short training cases stopped at a faulty expected
forward value. Exact rational `Fraction` multiplication discarded the sign of
zero when a negative parameter was multiplied by a zero dropout mask. The ELF
correctly returned negative zero, also present in original actual report row74,
`dropout-layout-contiguous`. The expectation now uses an exactly representable
binary64 dyadic multiplication before encoding binary32. Bitwise comparison
remains strict. All old requests/responses are exact prefixes of the final
ones; no input, tolerance, native source or native image changed. The first
report, executed script, raw frames and exact one-expression correction remain.

A first source-audit assertion also conflated the literal descriptor offset40
with the comparison's opcode bound40. The corrected assertion identifies the
exact comparison SSA instruction. No native execution occurred during that
failure. An archive API attempt rejected an unsupported streamed-Tar `preset`
argument before adding any member; its32-byte partial archive and error receipt
are retained. The successful archive uses a separate XZ writer and decodes
every member for exact size, mode and SHA verification.

`all-raw-observations.tar.xz` contains all118 raw evidence files,18,928,809
expanded bytes, including first failures and final passes. Its96,304 compressed
bytes have SHA
`7b04428fa1098fa8e57c6a6006a0a01d3f4eb67faee7b98f7088c1e3217e5b74`.
`all-raw-observations.json` records each verified member. No producer or
canonical-worktree file was edited by this review.

## Remaining scope

This is a first-order CPU binary32 shared-owner composition. Existing bounded
rank, element count, object count, work and frame admission remain. It does not
close general optimizer modes, arbitrary floating step exponents, all RNG
distributions/dtypes, hooks, higher-order autograd, or the public-gradient-object
successor. Logical native reference counts are not PyTorch intrusive-pointer
counts. The complete PyTorch rewrite, all2,590 schemas and all17 acceptance gates
remain open.
