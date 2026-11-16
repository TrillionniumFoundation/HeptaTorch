# Mixed10 tombstone legacy-refusal regression addon

This additive test corrects a host test dependency. The frozen source19 package
(manifest 347a9078...) and its original `test_storage_tombstone.py` stay intact.
The mixed10 native program and current 8361/58c9/aa8 observers are unchanged.

## Actual outcomes

The unchanged original test executed three native frames, then failed because
it loaded the already-corrected aa8 predicate while expecting original6ea to
reject the legal Storage tombstone. Its failure cleanup called a nonexistent
`Session.stop`; this secondary exception prevented its report from being
written. The complete original stderr, requests/responses and snapshots are
preserved. A separate read-only Python trace execution of the same source
captured the primary FAIL and secondary exception; its driver performed real
EOF cleanup, separately recorded. Neither original run is called a PASS.

The new entry actually passed one native case in six frames and four copied
JSON refusal controls, with EOF exit zero. It binds the exact original6ea file
before import and after execution. Only SCHEMA/SOURCE/IMAGE/PROFILE module data
identify the copied JSON snapshot; no old function body is changed. Every
original assertion remains, and seven explicit edits invert to the entire
original test bytes. Failure cleanup uses the existing real EOF method and
preserves any primary error.

## Repository CLI

Run from the repository root after the frozen source19 package and its normal
dependencies are installed:

```sh
python -S -B tests/mixed_frozen35_accounting/test_storage_tombstone_legacy_refusal.py \
  --binary build/mixed-integer-autograd/tensor-mixed-autograd-session \
  --source build/mixed-integer-autograd/runtime.hepta \
  --protocol tests/retained_resource_review/protocol.py \
  --output NEW_TOMBSTONE_OUTPUT_DIRECTORY
```

The sole additional dependency is the already-published
`tests/dtype_owner_accounting/owner640.py`, exact SHA256
6ea9e8b33164089034a2a71a03f4850e1ec18ed1a09155d7841f700ef311c833.
The script rejects a different dependency before native execution.

The regression is ownership admission, not another numerical API. Its four
negative cases alter copied JSON only. It does not replace the real full35
EOF/Version gates or the original copied18 suite. The actual ROG live32/full35
evidence remains separately frozen and did not run this extra regression.
Global PyTorch qualification remains HOLD and full parity false.

## Evidence

`raw-evidence.tar.xz` contains every original/fixed raw file, exact test and
observer inputs, protocol/client and executed native/source bytes.
The manifest binds each member's original path, size, SHA256 and mode. A complete
single-XZ EOF plus tar member set, bytes and modes check was performed before
freezing; original inputs were re-read afterward. Child JSON absolute paths and
all original report bytes are preserved. The outer archive and manifest exclude
themselves; the package manifest separately binds all delivered payloads.
