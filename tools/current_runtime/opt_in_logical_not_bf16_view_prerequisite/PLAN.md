# Public BF16 whole-root dense view: standalone opt-in draft

Exact constructor source base 29e3ef6794f1a670cabcc94de734a883738acf81c88adb0fd5f31b52782fdaea
from draft bf69188368ef73fc749b8e4c9211d9a2045dfc4e. F16 and BF16 require
separate exact tag+width checks; BF16 is public tag 6, width two.

Dispatcher admits new BF16 opcode 111 only. The existing F16 constructor
helper is renamed to reflect its exact tag whitelist; the existing F16/F64
gates and every other rng_strided_view caller are preserved. No new ABI
method, op128 write, or canonical/default activation. Retain exact same
nonempty ordinary no-grad root, actual live Storage/Version, root tagged
start plus width-sized extent, rank2/3, zero offset, all positive strides
and whole-root dense/injective geometry checks, before existing root
retention/publish. Root grad/edge checks stay fail-closed. Geometry rejects
may happen on the trial/provisional alias before retention/publish; show
exact physical whole-frame rollback and no leaked private retain.

The only changed split is unit 11. Its helper export is renamed with the
same signature; imports/dependency paths, all other source bodies, all
op128 view semantics and every previous provider remain the same. Static
body/arity/partition/inverse checks passed. --compose checks exact original
parent then permits a single separate logical-out helper body and only
unit12 changed from that parent. Actual pending F16-out projection was
applied in both orders in memory and produced the same aggregate bytes.
No toolchain/native/Torch is run by this draft.

## Finite real validation

1. Baseline: exact preceding ELF physically accepts BF16 factory op97 and
   metadata/readback at width2, then actually refuses all three valid
   opcode111 requests (rank2 permutation, rank3 permutation, rank2
   row-major whole-root). Retain old refusal and exact rollback. If composing
   after F16-out adoption, use that exact new image as predecessor and retain
   the uncomposed baseline only as separately labelled older evidence.
2. Independent locked CPU Torch-only raw u16 fixture for BF16, not rounded
   via float conversion: include both signed zeros (0000/8000), signed
   minimum subnormals (0001/8001), ±inf (7f80/ff80), distinct quiet and
   signaling NaN raw words (7fc1/7fa1/ffc3/ffa3), and varied finite words.
   Geometry [3,2] -> [2,3]/[1,2], [2,3,4] -> [3,4,2]/[4,1,12],
   and [6] -> [2,3]/[3,1]. Same storage address, metadata, sibling and
   shared version delta 0 independently observed. Preserve F16 raw subtype
   fixtures and truth checks, not inferred from BF16.
3. Matching two cold executables. Native no-Torch op111 plus op13/op98;
   independently compute permutation and little-endian output words,
   compare all words to official, ensure root raw unchanged, no fp cast.
   Verify root/view have distinct handles, raw tag6/width2, same live
   Storage/Version IDs, tagged address exact, descriptor +232 and kind1 +32
   both zero, kind1 +40 remaining backing capacity exactly 2*count; root
   and sibling lineage and raw shared Version counter unchanged.
4. Both release orders plus final weak expiry: source/root handle drop
   before child and sibling still read, child-first keeps root/sibling
   valid, op42/43/45 live storage lease after the other handle drops, final
   sibling release followed by op43 status6 and full rollback, release
   weak via 44. Confirm root incarnation and private keepers from raw;
   construction must not allocate replacement storage or bump Version.
5. Negative exact whole-owner rollback: nonwhole, rank1/rank4, empty,
   positive offset, repeated/zero active strides, gapped inbounds
   nonwhole, nested retained view, valid constructor followed by later
   invalid selector, and BF16 out op128 status7 against successfully
   constructed new BF16 view. A requires-grad attempt unsupported by
   BF16 factory/flag APIs is a setup limitation, not a successful grad
   construction or tighter runtime proof. Ensure private dtype3 and
   existing out refusals retained, zero regressions across F16
   constructor and any separately published F16-out positive route,
   all 44 existing view output cases and retained79/82.
