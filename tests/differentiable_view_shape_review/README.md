# Fixed view/reshape and wrapped-scalar native observations

These standalone readers send commands to the HIA native image and inspect its
owned bytes. Tensor arithmetic and gradients execute in HIA. Expected numeric
words are taken from the immutable, provenance-bound PyTorch observations.

replay_shape26_epoch_r1.py uses the original 26-case reader's unchanged case,
status, and reference-admission functions. Its nine explicit source changes are
recorded in reader-source-inverse.json and reverse exactly to the included
replay_shape26_r3.py. The additional adapter only distinguishes recorded raw
Tensor/Storage addresses after the old Saved scalar's lifetime is proven to have
ended. It requires all old Saved getters to report release, the exact successful
default-backward boundary, and absence of live aliases or prior observed gradient
StorageWeak handles. It never uses a native result, case number, or expected
gradient to grant a new identity epoch. The complete JSON inverse and all
nonidentity fields remain unchanged. The global native reverse identity maps
remain injective. This evidence does not add a new PyTorch Weak observation.

The original strict 05 and 06 failures remain unchanged in the fixed Git
[scalar05 history](https://github.com/TrillionniumFoundation/HeptaTorch/tree/0e006aebddfc9de564d5d14879d7aca190817ce7/validation/view-reshape-r4-scalar05-history) and
[scalar06 history](https://github.com/TrillionniumFoundation/HeptaTorch/tree/0e006aebddfc9de564d5d14879d7aca190817ce7/validation/view-reshape-r4-scalar06-history); a new run never replaces them.

replay_wrapped_scalar32_r1.py compares the original eight-byte Saved scalar,
physical and logical tensor bytes, geometry, Version, AD edges, retained and leaf
gradients, complete generator state, and EOF against 32 fixed observations. Its
reference has no Weak-expiration observation. Native observer Weak handles are
cleanup-only in that suite.

The original 14-case and four-input-boundary readers remain in
tests/differentiable_view_input4_review; their comparison code is reused
unchanged with explicit image and source bindings.

Every native run must bind the exact source, ELF, build receipt, all freshly
compiled units, protocol, observer, support reader, original reference and probe.
An installed reader or a successful source-only preparation does not qualify a
native runtime. Finite results do not close an entire PyTorch schema. The whole
project remains HOLD with 0 fully closed schemas out of the pinned 2,590.
