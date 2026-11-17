# Differentiable view native review

This directory contains the exact R3 independent native reader and its fixed
transport dependencies. HIA performs all Tensor arithmetic, automatic
differentiation, mutation and ownership operations. Python records requests and
compares the actual outputs with the immutable PyTorch reference.

## Restore the independently admitted reference

Run from the repository root with a fresh output directory:

    python3 -I -S -B tests/differentiable_view_native_review/prepare_reference.py \
      --output build/differentiable-view-reference14

This verifies the complete 52-member archive, every byte and original mode, the
single XZ stream and tar digest, then runs the unchanged independent admission
reader. The original PyTorch execution is not repeated. Its 14 actual cases,
209 operations and complete provenance remain byte-identical.

The restored input paths are:

- build/differentiable-view-reference14/payload/original/actual.json
- build/differentiable-view-reference14/payload/original/inputs/differentiable-view14-probe-1ca74384.py

## Replay a native image

Run replay_direct_view14.py --help for every required input. The caller supplies
the actual image, HIA source and build report together with their exact hashes.
The build must contain all 12 emitted units and corresponding objects. Use
protocol.py and native_observer_direct02.py for the current R2 image; the protocol
loads its pinned sibling owner_client.py. The direct01 observer remains the exact
historical R1 observer.

A replay requires a new output directory. It retains input copies, complete
transport transcripts, ownership state and individual case reports, including
FAIL results. The aggregate process exits nonzero unless all 14 cases pass,
every input is unchanged and every session reaches clean EOF.

The R3 diagnostic serialization fix changes only the encoding of an existing
protocol Payload in a report. It does not change execution, comparison or the
immutable reference. The R1 HIA image's first complete replay was 13 PASS / 1
FAIL: view history refresh incorrectly assigned retained gradients and failed
a final ownership audit. Its failure is retained separately. Installing the
reader or admitting the reference does not qualify that image.

Coverage is the fixed F32 cases and exact recorded identity relations, not
general Python wrapper semantics or complete PyTorch schema parity. All full
project acceptance gates retain their independent status.

The current R2 HIA image passed the same R3 reader on all fourteen actual cases:
15,135 frames, 96,461 assertions, 14,126 exact words and fourteen clean EOFs.
Original report SHA-256: 0b987fcb1d7fb74fe8a6d1edb994bed53851f07acbc3dfe6af6f754489e598d4.
The source and executable are pinned by the Make targets. See
../../docs/tensor-to-differentiable-views.md for the exact scope and remaining gates.
