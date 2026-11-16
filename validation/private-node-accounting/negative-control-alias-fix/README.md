# Preserve the first observer-control failure and correct test-copy isolation

The first observer CI ran head 1d45ea911399b2d4815edc6bb7bf970e2b6e2e68,
tree 73b501635e8ef7a9ed4c0fa975183b618600bda2, in run 43860338027.
Both Linux matrix jobs passed all three actual native controls (13 frames).
Each copied-snapshot checker then passed four original samples and its first
16 rejection controls, and failed at failed-frame-raw-owner-digest-changed.
Torch installation, live32 and frozen35 were consequently skipped in that
workflow. Its partial custody manifests correctly preserve those missing
reports. The separate Native HIA workflow is a different execution.

The failed frame's equal before/after snapshots had been deduplicated to one
Python object. copy.deepcopy retained this shared identity, so the checker
changed both digests when it intended to corrupt only after.full_sha256.
The native observer correctly saw equal digests. This was a test-input
construction error; the exact rollback assertion must remain.

The correction makes independent JSON value copies, requires the two sides
to be distinct objects, and asserts that every mutation leaves before
byte-for-byte unchanged. A byte inverse recovers the original checker.
The 18 case names, sample categories and predefined rejection-rule ASTs
are identical. Observer e900466b, launcher 65c0d687, all native/source pins,
all original oracle cases, tolerances, EOF and Version gates are unchanged.

Original checker:
48e3b0592585ba721265536fc11cfb460d870e88cb20dbfd9ecb41d6f299215f
Corrected checker:
c4873900abd44f2c74c2632d662d932ee0912501b7a127771c2bfecccc0f2db6

The two original CI artifacts are retained byte-for-byte:
* observer-355-first.zip: 71,232 bytes, SHA256
  5d3eb225f7dedf877481e288a28210d2177e6e4d2308302b41a1590b7442dcbf.
* observer-428-first.zip: 71,614 bytes, SHA256
  15e127281791b8ce4e0bab04f6a3ec085ecf606f254d2a29a446ec3b36058e68.

Each contains its exact PARTIAL custody manifest and all 34 inventoried
files. The actual failed reports remain FAIL. The corrected checker was
then executed on copies of these exact hosted journals on Mac CPython 3.14;
both profiles passed four original samples and all 18 exact rejection
controls. This replay runs no native process and imports no Torch.
repaired-controls-exact.tar.xz retains every replay input, output and
mutated negative sample without reserializing the original reports.
The new GitHub Linux execution is still required for the corrected candidate.

The full PyTorch rewrite remains HOLD. No schema-parity count is increased
by these transport and observer controls.
