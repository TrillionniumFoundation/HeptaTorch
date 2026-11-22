# Retained logical147 replay on the joint opt-in image

This separately runs one of the 35 mandatory suites on the same joint image
as [#63](https://github.com/TrillionniumFoundation/HeptaTorch/pull/63).
It exercises the original logical-reduction operators 118 and 119, not new
logical-not opcode 127. The original canonical qualifier still rejects all
noncanonical images. This adds an explicit, narrowly pinned execution path;
it neither changes that legacy gate nor activates an operator by default.

The exact on-disk logical147 controller, native observer, ordered cases,
reference source and two transport modules are hash-verified. The wrapper
checks that the unmodified canonical reader refuses the candidate at the
original manifest check. It substitutes one call in the controller in memory,
then uses the [separate offline identity checker](../opt_in_admission/README.md)
to admit only the exact joint image, including all 15 source/object pairs,
link packet, and scoped receipt. The substitution has a byte-exact inverse.
The original controller still launches the unchanged native observer, resolves
the original compressed PyTorch reference and original execution provenance,
and runs the same controls, comparisons and EOF checks. No new PyTorch
reference generation or tolerance changes are made.

## Reproduce

With a fresh checkout carrying #63, build the joint image as described in
[the dual qualification guide](README.md). Then run:

```sh
python3 -I -B tools/current_runtime/opt_in_dual_unary/replay_logical147.py \
  --repository "$REPO" --dual-build "$SCRATCH/dual-build" \
  --dual-scoped "$REPO/tools/current_runtime/opt_in_dual_unary/qualification.json" \
  --output "$SCRATCH/retained-logical147"
```

The original runner logs, reports, 150 child transcripts, and frozen
reference copy remain under the chosen output directory. On failure the
wrapper exits nonzero; it keeps original error reports and child artifacts
where available. The wrapper's `receipt.json` records the unchanged observer
identities, exact build and image hashes, and hashes of the actual original
reports. Actual build reports and raw observer reports may include local
paths/process timings, so their hashes are expected to vary across physical
runs. The semantic row/check/frame counts should match.

Both cold images produced the same candidate ELF and separately passed all
147 original ordered cases, 18 packet controls, both real allocation
failure/recovery tests, and all 150 child exit/EOF checks. Each run used 3,141
physical frames and 6,396 checks (6,330 per-case checks). The wrong baseline
and abs-only image, wrong scoped receipt, and a modified native observer were
refused before native observation. The compact
[recorded receipts](logical147-evidence.json) report those separate runs.

This is **logical147 only**. It does not claim the other 34 mandatory suites,
the canonical runtime gate, production activation, or complete PyTorch
parity. The historical signbit reader, for example, still requires unassigned
opcode 127 to be rejected; the joint candidate intentionally assigns it. Any
future successor must keep that old baseline observation intact and record
the candidate behavior separately.
