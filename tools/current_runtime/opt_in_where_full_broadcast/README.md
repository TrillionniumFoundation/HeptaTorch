# Native where.self: three-operand CPU broadcasting without dtype promotion

Adds the previously missing shape fallback when the Bool mask and two
same-dtype ordinary CPU roots jointly determine the output shape. It supports
right-aligned rank up to four, independent x/y expansion and mask-driven
output rank growth. Shape compatibility is established for every axis,
including empty results, before any shape scratch write or allocation.
Only exact one or equal extents join; zero does not join a non-singleton
positive extent. All previous fixed-shape fast paths remain. The new
nonempty output is capped at 262,144 elements. Views, promotion, autograd,
out=, and default/hosted activation remain outside this slice.

The original known bshape workspace is saved in four SSA words, reused by
the original typed allocator, and restored before validating the freshly
copied result shape. Actual Half multiply seeded scratch [2,3,0,0] on both
exact ELFs with identical request and response bytes. A subsequent live
three-way where on the successor preserved all 32 workspace bytes.
Independent Cartesian readback verified fresh Storage/Version/Tensor ownership,
zero initial Version, exact unmodified input bytes/metadata, both lifetime
orders and local-result reference. A real valid where followed by an invalid
operation rolled back the whole trial.

Two cold builds reproduced the same final ELF. Official locked CPU PyTorch,
a separate native no-Torch observer, and an independent reader agreed on 12
fixtures and 262,282 output words. They include all six public dtypes,
rank-growing scalar values or condition, three different contributing axes,
two empty cases including [512,512,2,0], and the exact 262,144-element Bool
capacity boundary with every raw byte read back. A geometrically compatible
262,656-element join from small roots returned capacity status 3 and left
the entire owner image unchanged; incompatible join of zero and a
non-singleton returned shape status 7 even after an earlier compatible empty
axis. Predecessor 12 genuine fixture refusals and one matching scratch control
are archived separately (134 raw frames). The successor's new suite contains
296 frames with 25 full-owner rollback controls.

Exactly four historical controls moved to four explicit positive witnesses:
three value broadcasts and one mask-driven rank expansion. Original status 7,
raw input words, geometry, and untouched owner proofs remain preserved.
All non-retired prior #66 and #65 frames match byte-for-byte. Across all
successor suites, 4,261 frames and 452 rollback controls were checked.
The prior real allocation pressure proof was retained; no new dynamic
pressure test is claimed for this feature.

The first cold unit12 compile rejected the original source because four
SSA Bool names duplicated block arguments. The archived receipt and original
draft difference retain this failure. The final source only renamed those
four Bool locals in that draft before two cold builds. Earlier observer
diagnostics (stride-zero oracle flattening, list-vs-tuple assertion, and a
survivor read using the output count) were corrected without changing the
final executable and are not counted.

Reproduce source verification with isolated Python:
`prepare_full.py EXACT_PARENT EXACT_BUILD_PY` (optional new final directory
materializes). Build cold twice; the supplied manifest pins the second.
Use `oracle_full.py PINNED_HELPER_DIR ORACLE_LOCK INSTALL_REPORT FIXTURE` only
in the locked CPU Torch interpreter, or restore the exact fixture with
`expand_fixture.py full.fixture.json.zlib.b85 FIXTURE`. Then use
`native_full.py PINNED_HELPERS PROTOCOL_PY ELF FIXTURE OUT baseline|candidate`
without Torch for each exact executable, and independently
`pure_full.py FIXTURE BASELINE_CAPTURE CANDIDATE_CAPTURE`.
`verify_retired.py OLD_GENERAL OLD_TRAILING OLD_SCALAR RETAINED_ROOT BASELINE
NEW_CAPTURE FIXTURE` checks the precise archival removals and matched scratch.
