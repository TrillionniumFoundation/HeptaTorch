# Opt-in ordinary rank-3 Bool nonzero

Exact parent #69 cf89ed3b9969ca2f548738b211e7b756ae2e0913, tree
c028424ac03c7e4900a450042efc8a4edfb5263b. Adds canonical ordinary
contiguous zero-offset rank-3 CPU Bool under the existing opcode 135.
The rank-1 definition remains byte-identical; rank-2 uses the same
shared scans and physical storage process, including all six admitted
dtypes, and keeps its original geometry and limits. No change to
default, higher ranks, non-Bool rank-3 inputs, tuple/out, or views.

Rank-3 coordinates are lexicographic plane/row/column triples. The
fresh exact I64 output has [K,3], element strides [1,K], Version0,
zero offset, 24*K bytes and physical column-major storage. It is
row-major contiguous exactly when K≤1. Actual axes and canonical
strides are checked before any division; empty input and K0 produce no
coordinate address or store. Local bounds are N≤262144 and K≤87381
so 3*K≤262144. Scratch is restored and stride metadata is verified
before publication, without changing allocator defaults/global flags.

Locked CPU Torch matches all 24 frozen cases. The 23 scoped ordinary
cases have 524,610 I64 words in each of the independently measured
logical and physical streams. The remaining K=87,382 case is Torch
reference only; no above-cap native call, fault, invalid owner,
pressure or allocation limit assertion. All three columns' individual
hashes are checked, including the empty and singleton cases.

Two clean cold builds reproduced ELF
7ebdbbae563577e908e01c8375a6047770db12781eaebbf5255a35fcbe8f1605
for exact HIA ce78e4be4c06b1b05dc3221d3108ecd4241a1787e8ba6422f17154f6c5b1df87.
The exact native parent refused all 23 valid rank-3 input roots with
status7 and no owner change. The candidate passed 23/23 (342 frames),
including ten K0, two K1, two exact N cap and one exact K cap. The
independent pure decoder checked each wire logical coordinate and
physical underlying column, their shape/strides/Version, distinct
fresh authorities and both drop orders.

The first build exited status2 in unit 11, before linking. The repaired
source changes only the three spelling identities of nz_rank2's
header formals to match the existing entry block. The original refusal
receipt and exact one-line patch are preserved. The first retention
preflight also did not launch below the memory floor; all actual
replays used the original unchanged guard.

All three exact ordinary successful historical campaigns replayed byte
for byte: 1,197 rank-1, 392 rank-2 Bool and 624 rank-2 dtype frames,
2,213 in total including 125 positive opcode135 cases. This does not
claim any unsupported-path, malformed or full default-suite retest.
