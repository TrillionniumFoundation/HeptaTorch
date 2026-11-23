# logical_not(out=): direct I64 dense permutation

Adds only disjoint, zero-offset, positive-stride, nonempty, whole-root
dense I64 permutation destinations, retaining the existing F32 and Bool
arms. The ordinary direct root is retained through existing opcode111.
The guard explicitly admits I64 (dtype 2) with a validated eight-byte
width; the validated width is passed unchanged to layout and storage
extent. No new layout family, dtype, default activation, or whole-runtime
qualification is claimed.

Both cold builds produce the same ELF. The pinned CPU Torch fixture ran
and exited first. The predecessor constructed and refused all three
I64 view destinations with exact rollback. A separate no-Torch runtime
replay matched three cases each of I64, Bool and F32 and 35 exact rollback
controls in 300 frames. It used I64 minimum/maximum, negative and wide
positive values and varied logical versus physical output order. The pure
reader reparses actual opcode111 commands, dtype/bytes and physical
mapping, unchanged disjoint sources, shared root/view/sibling identity,
and a single shared Version increment. No controls failed at construction.
The earlier 79 positives and 82 rollbacks were also replayed using the
untouched fixture; their 876 raw frames match byte for byte.

prepare_i64.py takes the previous projected source root, new output,
existing build.py and optional discovery build receipt (a nonexistent
path leaves the affected object and final ELF initially unpinned).
The only HIA semantic edit extends the existing width whitelist; other
function bodies are byte-compared except the helper's renamed call.
Use pinned-manifest.json for a fully pinned repeat. The observation
script arguments are unchanged from the prior Bool package. To verify
the retained 79 cases use the pinned original numeric-self readback.
See evidence.json for exact local raw receipts and hashes. Unique raw
output remains in temporary storage.
