# Opt-in bounded symmetric masked_select for public CPU values

This is a source projection against HeptaTorch #67, exact git
0489d49f4f497f02a218a06d64d80081ae004a00 and tree
58ba08dbb8f14ac045e836f664b841e250280f7c. It does not alter the default
runtime, public opcode 130, accepted value dtypes (F32, Bool, I64, F64,
F16, BF16) or Bool mask requirement. Equal-shape and fixed-value
one-way paths retain their recognized behavior. Newly admitted inputs
must be ordinary owned CPU roots, value rank1–4 and max joined rank≤4,
and each true input count and logical joined element count must be
within 262,144.

The fallback joins right-aligned dimensions; a singleton can expand
to zero, and any incompatible axis rejects even if another makes the
output empty. The front-to-back bounded product keeps an overcap
sentinel, which a legitimate zero can reset, and checks every axis.
The trailing-coordinate mapper checks actual per-operand counts and
never divides by zero or reads any input payload for an empty lattice.
No expanded tensor or joined shape scratch is allocated. The existing
two-pass raw gather and rank1 output allocation remain. Output dtype,
element stride1, zero offset, live fresh independent Storage/Version0,
canonical selected Bool, and raw width-specific writes are retained.

Two independent cold builds produce the same exact image
77196c361782d00320b21b8becd8bb25777978aa7120341877271ef25166b748
from exact combined source
877bd9b77c4cf57cd67549c9e9a7ff6a5f4fb87c829587dab77e46347c1c2b3c.
The separate fixed CPU PyTorch and independent integer/Cartesian
oracles agree on 73 fixtures, including signed NaN payloads, retained
false Bool bytes, exact opposite singleton ordering, seven zero
lattices, and 12 locally overcap cases even with original roots of
only 512 and 513 elements and K of either 0 or 512.

Exact original native campaign: 873 frames with 18 retained positives
(150 words) and 55 status7 refusals in the new-scope matrix. Exact new
campaign: 1,491 frames, 61 successes (4,554 exact raw words), 86 whole
owner rollback comparisons. The 12 valid-geometry joined-overcap cases
returned status3; incompatible axes and true view/grad and suffix
rollback are checked independently. Rank5 mask and private tag3
rank1 requests could not create a public operand: factory opcode97
returned status2/status7 respectively. Those are setup-only, not
opcode130 refusals. No allocator OOM was solicited or claimed.

The replay retired only 14 context-bound physical occurrences of
eight newly valid original cases. Every exclusion is pinned to its
entire source stream, ordinal, old status and raw response, and its
actual previous owner byte hash and typed operand snapshots. Three
byte-identical requests in different previously successful owner
states remain byte-checked and are not excluded. One old F64 witness
belongs to an earlier already-qualified stage and is still marked
non-comparable, not a new retirement. Exact compare covers 6,436
request/response pairs across the explicit retained corpus, including
overlapping saved prefixes; this is not the entire hosted default gate.
