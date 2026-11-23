# logical_not(out=): disjoint dense F32 permutation destination

This follow-up to #65 adds only the zero-offset, positive-stride,
nonempty, retained F32 destination view of its complete ordinary
no-grad root. The contiguous source, exact shape, disjoint Storage and
Version, and existing opt-in protocol/commit rules remain unchanged.
The source edit commutes with the prior numeric-self gate. It changes
one existing HIA function and adds one bounded helper.

Two separately executed cold builds reproduced the same ELF. A fresh
locked CPU Torch process first produced the three immutable comparison
fixtures and exited. The separate native processes ran no Torch.
The predecessor refused all three valid view destinations, with exact
rollback. The candidate passed rank-2, rank-3 and singleton permutations,
including same-frame result reuse and base/sibling visibility. The
offline wire and owner reader verified the three successes, shared
Version +1 and 13 byte-identical rollbacks in 112 frames. It checked
a genuine repeated positive-stride overlap, a zero-stride case, and
the actual successful opcode90 root toggle: the view was already
synchronized, not stale. An already-grad view and a whole-frame
root-toggle then opcode128 both refused. No unsupported constructor
controls remained in the final capture.

The prior 79 positive cases and all 82 rollback controls were also
executed separately against their existing immutable locked Torch
fixture, retaining all previous numerical and owner assertions.
The original offline reader verified their 876 native frames. Previous
fixture resource stops and observer setup/correction attempts were
retained; they are not counted as new passing observations.

prepare_build_source.py projects the semantic change to the exact prior
monolith and the runtime800_12 unit. It adds only two needed imports and
one export, then runs the existing source checker. Build with the ordinary
exact compiler and linker; for a fully pinned repeat use
pinned-manifest.json against the projected source root. oracle_fixture.py
takes the published prior helper directory, oracle lock and pip install
receipt. native_observe.py takes that helper directory, original
protocol, exact ELF, fresh output directory and baseline or candidate.
pure_readback.py consumes both raw output directories, with the official
fixture in their parent. retained79_observe.py reuses the previous
official fixture and unchanged independent raw reader after checking the
new ELF identity. See [evidence](evidence.json) for exact raw digests.

This is scoped opt-in evidence only; it establishes neither arbitrary
views or strides nor an unchanged whole-runtime gate or production
activation. Unique raw output remains in temporary storage.
