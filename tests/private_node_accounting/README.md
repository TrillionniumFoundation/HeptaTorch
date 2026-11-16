# Exact private-node accounting for the frozen 35-case oracle

This is a read-only test observer for the published 355-function and
428-function native HIA owners. It changes no production HIA bytes and does not
implement PyTorch operations in Python. The full rewrite remains HOLD:
2,590 upstream schemas, 688 derivative entries, and the existing whole-system
gates remain the acceptance denominator.

## Why the extra observer is necessary

PR28 head 30ea38d3572a18ede2dda80decc2fcb4a83feb84 actually failed its hosted
355 frozen35 step: 3 cases passed and 32 failed. The old additional v3 bridge
asserted that the native physical node high-water equaled the legacy client's
logical creation count. Actual private gradient Tensor nodes make those two
quantities differ.

The original failure is preserved under
validation/private-node-accounting/first-hosted-failure-30ea38d/. Its 32 failed
close calls also prevented the original EOF/exit/empty-pipe checks and subsequent
parameter Version gate from running. The old artifact cannot be relabeled PASS.
The hosted common-428 live32/frozen35 steps were skipped in that run.

## Explicit observer contract

The original entry, bridge, numeric cases, tolerances, lifetime cases, reference
support, and native binaries are source-pinned and remain byte-identical.
run_frozen35_accounted.py temporarily wraps bridge.install and records the
replacement and its inverse. The original additional equality
stats[11] == self.next_id-1 is replaced explicitly; it remains in the output as a
diagnostic observation, not as an acceptance criterion.

The replacement requires exact relationships, not lower bounds:

* Each actual incarnation increment is accounted for by the unchanged Program's
  original construction tickets (including Linear/MSE count 3) or a new,
  reachable private gradient Tensor with matching identity and references.
* Graph, AccGrad, private-gradient, Storage, and Version references are counted
  independently from copied committed owner records.
* Non-reset allocation intervals and reset high-water/root closure are checked
  exactly. Unsupported same-frame backward/reset patterns are rejected.
* A failed frame preserves the owner address, structure, and full byte digest.
* The actual command-54 statistics equal independently reconstructed native
  statistics.
* The unchanged original Session.close and final parameter Version assertion
  must execute successfully for every one of the 35 cases.

Only the observer's own child is read through rb /proc access. There are no
native-memory writes. Original inner JSON is never reserialized or reclassified;
the additive report binds its exact byte length and SHA256.

## Reproducible checks

CPython 3.12 on Linux x86_64 and the repository's exact PyTorch oracle lock are
required for real oracle execution. A Mac syntax check is not native
qualification. The initial Mac Python 3.9 source-prepare rejection is retained;
a later existing Python 3.14 source-only preparation does not qualify the
pinned Torch runtime either.

The dedicated Private node accounting workflow uses two independent matrix
jobs. Each restores an exact published source/ELF pair from an archive, labels
that operation RESTORED_NOT_NEW_BUILD, runs three actual native controls, and
applies 18 independent negative controls to copies of four real snapshots.
The copies are never written back to the native owner. It then runs the
unchanged live32 and all frozen35 cases with the pinned reference, followed by
cross-report verification of every EOF, Version, source-custody, and provenance
gate.

The existing Native HIA workflow continues to build the frozen native images
and run its previous qualifications. Only its two new 355/428 frozen35 commands
use this explicit observer. The other 67 step blocks from the failed PR28
workflow remain byte-identical. Two additional steps collect and export the
raw observer evidence; the workflow declares 71 steps. Actual hosted step
counts and outcomes must come from the finished job, not that declaration.

The successful Native HIA report export is expected to contain 58 original
root JSON files and 30 report envelopes, including the two unchanged inner
reports as new separate files. These are expectations to verify against the
actual artifact, not asserted execution results.

## Evidence boundaries

owner_accounting.py and run_frozen35_accounted.py are the implementation under
test; test_native_accounting.py exercises actual HIA execution.
check_owner_snapshots.py is the independent copied-snapshot rejection suite.
restore_native_inputs.py restores exact archived inputs without compiling.
check_frozen35_accounted.py binds the additive and original reports without
changing either. collect_private_node_journals.py copies original report bytes
into a self-contained journal artifact and records every file hash.

The journal custody manifest binds the exact original and additive reports,
owner snapshots, wire traces, checkout tree, source head, run ID, and attempt.
Its COMPLETE/PARTIAL status describes evidence collection only; it cannot turn
a failed or unexecuted test into a passing test.

None of these suites constitutes full-schema parity, CUDA/distributed support,
higher-order autograd, hooks, or complete PyTorch compatibility.
