# Standalone public F16 whole-root dense view

Adds a narrow direct F16 (tag 5) view constructor to the opt-in runtime.
It admits only zero-offset, nonempty, whole-root rank 2 or 3 dense views
with positive strides over an ordinary no-autograd contiguous root with
exact live two-byte storage. Only opcode 111 changes for this dtype.
The BF16 view, the F16 view as a logical-not output destination, other
layout extensions and default activation remain out of scope.

Two independent cold builds produced byte-identical executables. A separate
official locked CPU PyTorch process produced three exact raw F16 layout
fixtures (rank-2 permutation, rank-3 permutation, rank-2 row-major), including
both zero signs, subnormals, infinities and distinct signed q/s NaN payloads.
No numeric float cast was used. The exact prior image physically refuses all
three real constructors. The independent no-Torch physical comparison
passed three new layout cases, each with two successful native constructors
(view and sibling), 82 request frames, and 21 full-owner rollback controls.
The two release orders keep bytes and the private root live while any view
holds it; the final view release expires the weak Storage. Every constructor
keeps the exact shared Version unchanged, reuses the root Storage, and
retains exact dtype, layout, byte offset and remaining extent. Three physical
op128 writes to these new F16 views still refuse; BF16 direct view still
refuses. A request to mark an F16 root requires-grad was itself refused
with status 5, and is recorded separately as a setup refusal, not as a
constructed grad-marked-root case or a passing rollback control.

On the same executable all 44 published typed layout positives, 143
rollbacks, and 1262 raw frames match byte for byte. A separate original
retained-79 replay has 79 positives and 82 rollbacks and byte-identical
request/reply streams for all 876 frames. These do not prove a hosted
aggregate or default/public runtime activation.

`prepare_f16_view.py` takes the published F32 row-major source and original
build.py, checking the exact typed row-major 4c95906 source as `--compose`.
The included pinned manifest fixes the subsequent build. The official fixture
stage, native driver and pure reader are separate processes.
`derive_retained_observers.py` verifies the prior published observers and
fixture, replacing only the executable pin: run the retained layout observer
with its unchanged owner reader, and the original retained79 fixture/reader
as before. See the plan for bounds and the evidence for exact proof hashes.
