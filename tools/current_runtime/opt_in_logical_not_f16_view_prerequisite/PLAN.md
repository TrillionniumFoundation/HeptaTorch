# Public F16 whole-root dense view: standalone opt-in draft

Base `c6a26c9` projected runtime SHA-256
`5c62c11900d20782c58b18b948b4074385e816d1936da2d7a03710715dba4f48`.
This is a narrow successor in that scratch profile, not a claim on the
canonical default or project-wide rewrite. F16 constructor only; no op128
extension and no BF16, private F64, offset, empty or autograd extension.

- `to_command`: only opcode 111 is newly exempted for F16. BF16 and all
  other prior rejection/exemption rules stay exact.
- `to_strided_view`: route tag 5 to the new checked helper, otherwise keep
  original explicit F64 and RNG routes. The independent
  `rng_strided_view` users/definition stay untouched.
- Helper: validate public caller-resolved direct ordinary contiguous F16
  root, zero grad/edge/alias/offset, nonempty capped count, exact 2-byte
  dtype, live Storage+Version and exact tagged root address/extent. Call
  the checked scalar config/parser and alias constructor. Before retained
  view publication require the provisional view to be rank 2 or 3,
  nonempty, same total root count, byte offset zero, all strides positive,
  and dense/nonoverlapping. A bad provisional view returns failure; outer
  trial session must discard all partial retains/metadata on this failure.
  Success uses unmodified root retention and publish. No data writes or
  version bump. Width-generic op98 reads are not modified.

Aggregate change two existing bodies plus one new body; unit 11 only.
Unit header exports the helper because the builder requires every core
body to be exported; it is private to the HIA runtime API (not an opcode
or a public caller entry). Two external declarations bind existing
unit-00 `is_dense` and `so_version_live` with exact signatures. Aggregate
and unit file projections both reverse byte-for-byte. `--compose` accepts
only a verified original-parent-compatible variant with the one distinct
logical_out layout body modified, identical signatures and only unit 12
changed. It preserves that variant's modifications and pins, except
candidate ELF/unit11 pins (no measured pin until a cold discovery build).
Without `--output`, the script does not write projected runtime sources.
Static aggregate/split, dependency, branch and call arity checks only.

## Finite physical checks, not executed yet

1. Exact original candidate ELF: explicit F16 root from op97, actual 111
   with genuine in-bounds [2,3]/[1,2] at element offset zero rejects;
   whole trial owner delta zero. Check rank 3 actual old rejection too.
2. Official locked CPU Torch in separate process: raw uint16 root words
   with ±0, subnormals, infinities and distinct quiet/signaling NaNs
   reinterpreted as half **without a floating conversion**. Rank 2 root
   [3,2] -> view [2,3]/[1,2], rank 3 root [2,3,4] -> view [3,4,2]/[4,1,12].
   Optional rank-2 row-major whole-root [2,3]/[3,1] positive. Compare raw
   u16 flatten order independently by index map, not PyTorch float cast.
3. Two cold fully pinned candidate outputs; no-Torch native op111 actual
   constructor, op13 metadata/layout and op98 payload reply; compare
   logical view raw words and untouched contiguous root with official
   fixture and independent decoding. Root and child public handles
   distinct; exact shared Storage/Version IDs and tagged origin, element
   offset zero equals descriptor +232 and kind1 +32 byte offset; +40
   is the remaining backing capacity (entire root size here), not an
   arbitrary occupied span. Retained base lineage authenticated. Separate
   sibling and constructor repeated; no shared-Version increment for
   any constructor. Release in both orders, proving root-token can drop
   before child/root sibling reads, final owner references settle and
   generation/stale IDs do not address new objects.
4. Negative actual constructor/frame cases: BF16 tag6, F16 positive
   offset, empty, shape not whole, rank1/rank4, duplicate stride and
   intersecting views, stride-zero broadcast, gapped positive strides,
   nested direct child, grad-marked root / edge if obtainable, malformed
   config before mutation, and valid 111 then invalid op in the same
   frame. Track distinction between early raw/setup rejection and
   post-construction geometry rollback; claim only what reached native.
   No reclassification of a legacy refusal. Prove full byte rollback,
   no published handle or leaked retained root, no version tick.
5. Explicit unchanged op128 status7 when newly admitted F16 view is
   out target (constructor standing alone), BF16 op111 still refuses,
   existing F16/BF16 ordinary contiguous logical behavior and prior
   32 layout / 79 historical positives plus corresponding controls
   retained on same ELF. If live keeper/descriptor acquisition or
   raw dump fails, diagnose without opening broader AD or dtype routes.
