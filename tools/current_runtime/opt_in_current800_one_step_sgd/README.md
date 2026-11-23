# Current800 one-step ordinary F32 linear SGD

Exact existing #71 source f8f924a82763a28d8ad12adbd371f25458ed86d1fbdee2406bb93912847afb4b,
ELF 5c8dfb2d329e211e7871438d64754b0f8912cc687c1d9ae3fc432f20e95d165d.
One CPU F32 leaf parameter W starts at [1,1]. One owned HIA session
creates only the three external literal inputs and uses live results
for matrix multiply, subtraction, square and scalar sum. It runs one
scalar backward, reads gradient via op11, releases the old forward
graph in reverse order, then performs exactly one ordinary in-place
SGD update (op9, learning rate 1/8). It computes a fresh forward loss,
not a second backward. Python submits and reads the wire; PyTorch is
a separate locked oracle, never the application backend.

Exact arithmetic, locked Torch and HIA agree on gradient [-4,8],
weight after step [1.5,0] and loss 8 → 2.25. Tensor, Storage and
Version authority of W are unchanged, with the counter increasing
exactly 0→1. X and target stay byte/metadata-identical and have no
gradient. The same private real gradient remained live with the same
raw data and independent authorities, without publishing an op91
handle. Its owner rows and op11 words were independently verified.

Independent raw readback checks 140 valid frames, 11 owner snapshots,
57 live ordinary root/output records and 2 private gradient records.
It binds all 26 locked raw observations (including both losses, old
and updated parameter and mathematical gradient) and verifies the
operation ancestry, layout, stride/offset, Version, and approved
release order. The first offline helper incorrectly expected a public
marker on the private grad node; the preserved helper was corrected
to validate the existing private slot and incarnation. No native
rerun, changed value or new owner behavior was required.

This is one finite deterministic application case. No momentum, decay,
optimizer-state suites, second step, higher derivatives, faults,
invalid ownership, stress, main775, default or full training claims.
