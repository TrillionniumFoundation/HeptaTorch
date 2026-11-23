# Standalone public BF16 whole-root dense view

In the opt-in runtime this adds only a direct public BF16 (tag 6) opcode111
constructor for nonempty zero-offset rank2/rank3 whole-root dense views with
positive strides, without autograd. The exact F16 route stays in place; no
BF16 view can be an opcode128 out destination. The canonical/default runtime,
other tensor methods, independent RNG callers, and general layout admission
are unchanged.

The two cold builds match byte for byte. A separate locked official CPU
Torch process produced raw BF16 fixtures for rank2 permutation, rank3
permutation and rank2 row-major geometry, including signed zeros,
subnormals, infinities, and distinct signed quiet/signaling NaN payloads.
No float conversion re-encodes those words. The exact published predecessor
accepted the BF16 root and refused all three real view requests with
complete rollback. The no-Torch candidate's independent raw reader checked
three successes with two published views each, 80 frames, 20 exact rollbacks,
same live width2 Storage/Version and byte span, unchanged Version on
construction, exact root/sibling words in both drop orders, weak Storage
survival while one view holds and expiry after final release, and three
actual op128 destination refusals. BF16 grad marking itself is a separate
status5 setup refusal, not a tested constructed grad-marked root.

The F16 constructor's six published views and both lifetime orders repeat,
with 20 unchanged rollback predicates. The single old BF16 constructor
denial from that matrix is preserved as exact old raw and not called an
unchanged predicate. The same new image retains all 47 typed output layout
positives, 159 rollbacks and three F16 written lifetimes within 1377 frames;
both request/reply streams match the preceding image byte for byte. The
original 79/82 and both streams of 876 frames also match. A first retained
layout observer attempt used the wrong neighboring fixture path; its short
abort is labelled separately and not counted.

Run prepare_bf16_view.py with exact standalone F16 constructor projected
source and original build.py, and --compose with the published F16 output
source. No output directory is materialized unless requested. The included
pinned manifest fixes candidate build outputs. derive_bf16_tests.py derives
all observers from the original F16-ctor and F16-out packages; keep
retain_layout.py alongside the F16-out fixture in a separate directory.
Official fixture, native observer, and pure reader use distinct processes.
See evidence.json for exact source, result, and retained frame hashes.
