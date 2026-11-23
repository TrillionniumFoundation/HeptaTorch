# Current800 two-layer F32 ReLU training step

Exact existing #71 opt-in source f8f924a82763a28d8ad12adbd371f25458ed86d1fbdee2406bb93912847afb4b,
ELF 5c8dfb2d329e211e7871438d64754b0f8912cc687c1d9ae3fc432f20e95d165d.
One HIA session creates four ordinary F32 roots and constructs every
intermediate from live native handles. Each graph is matmul, ReLU,
matmul, subtraction, square and scalar sum. One scalar backward binds
both weight gradients. Both are read through public op11; the old
forward graph is released normally before one SGD update per weight
(lr=1/32). The fresh forward produces a lower loss without a second
backward. There are no host-computed intermediates or PyTorch
application dependencies; the locked eager Torch process is an
independent oracle.

Exact analytics, locked Torch, and HIA agree on W1 gradient
[[4,-8],[8,-4]] and W2 gradient [[16],[-4]], with distinct intact
private gradient authorities before and after both steps. W1 becomes
[[15/8,-3/4],[3/4,9/8]], W2 becomes [[1/2],[-7/8]]. Ordinary input X
and target T remain byte- and metadata-identical. Both weight
Tensor/Storage/Version authorities stay the same, counters advance
once from 0 to 1, and fresh scalar loss falls from 8 to 12625/4096
(3.082275390625). All ReLU preactivations remain at least one unit
from zero and their masks stay unchanged.

Independent standard-library readback verified all 54 locked F32
words in wire logical and physical Storage order on 265 valid frames,
16 owner snapshots, 112 live ordinary records and 6 private gradient
records. It also verifies one backward, one update for each leaf and
the original graph release order. No op91 public gradient handle
was created. No code/build changes or additional native rereuns.

This is one deterministic bounded fixture. It does not verify a
second step, adaptive optimizer behavior, gradients beyond this
topology, general PyTorch training, performance, default parity, or
fault, invalid-owner, stress or held routes.
