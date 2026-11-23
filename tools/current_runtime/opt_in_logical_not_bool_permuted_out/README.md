# logical_not(out=): direct Bool dense permutation

Adds only disjoint, zero-offset, positive-stride, nonempty, whole-root
dense Bool permutation destinations on top of the admitted F32 arm.
The Bool view is created by the existing opcode111 on an ordinary
contiguous root. Both destination types are explicitly whitelisted and
their validated dtype widths are used for the root and byte extent. All
other guards and the shared opcode128 read/write and commit are retained.
No default or production activation, other dtype, view producer, or
full-runtime qualification is claimed.

Both cold builds produce the same ELF. The locked CPU Torch fixture was
generated in its own process before any native comparison. The published
predecessor refused three real Bool constructions and restored the
entire owner snapshot unchanged. The candidate matched three Bool and
three retained F32 outputs spanning rank-2, rank-3 and a singleton
permutation; its 24 rollback controls passed and all view controls were
actually constructed. The separate raw reader checks the actual opcode111
config and response, Bool and F32 byte order, root/view/sibling owners and
shared Version +1, disjoint unchanged source, and exact denial rollback.
The existing 79 positives and 82 rollback controls were replayed with
their earlier untouched fixture, and all 876 frames match byte for byte.

prepare_bool.py takes the previous projected source root, new output,
existing build.py and optional discovery build receipt (a nonexistent
path leaves the affected object and final ELF initially unpinned).
It changes one HIA helper and its name/reference in the existing call
and export; all other definitions are compared byte for byte. The
existing build.py verifies complete source and partition consistency.
Use pinned-manifest.json for a fully pinned repeat. The other scripts
take the same arguments as their direct F32 predecessors. For retained
79 readback use the original numeric-self readback.py and fixture pinned
in evidence. No new raw data is embedded; exact hashes are below and the
physical observations remain in temporary storage.
