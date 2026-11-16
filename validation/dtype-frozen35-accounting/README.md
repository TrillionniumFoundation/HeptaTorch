# Frozen35 on the 523-function, 640-byte runtime

This is an additive test observer and launcher. The HIA runtime, original35
oracle, numeric expectations, tolerances, explicit state Version differences,
EOF checks and post-close parameter Version assertions remain unchanged.
The whole PyTorch rewrite remains **HOLD**: denominator 2590, full parity false.

## Exact runtime and source inputs

- Source: `6ffe509afce28ff15705b4c7b269eaef8f8f89f9d8b2486d4d9fa45b6be66ef3`.
- Linux native image: `830f0b4ef8d6ad8c6a105be56d383628b45c84e3e43905fee6812c7c52d35a51`.
- Original35 entry: `149b41c455444edeb8d25d0ad98045e29683e780f37f7e210e7985242d359597`.
- Original six oracle/transport sources are bound individually by the launcher.
- Existing `tests/dtype_owner_accounting/owner640.py` must remain exactly
  `6ea9e8b33164089034a2a71a03f4850e1ec18ed1a09155d7841f700ef311c833`.
- Existing 355/428/446 observers and their rules are not edited.

Every actual descriptor is captured as all 80 words. The original fixed35
snapshot rule is reused with exactly one width predicate changed from 72 to
80; its inverse restores the complete original AST. The original transition,
Journal and Session overlay are reused under an AST identity proof. Full640
structural verification additionally runs on the complete snapshot. No fake
576-byte projection is produced.

The fixed35 command domain does not invoke NativeDropout99, detach_101,
retained hooks or owned Optimizers. This is checked at every successful frame,
and all new dtype/history fields are explicitly required to be zero at every
committed snapshot. Source pins bind the real capture call chain; ordinary
binary/matmul record operations only route an existing capture. Constructor
counts are derived from successful HIA operations, including optional affine
arguments. Counter differences are never assigned to an unexplained residual.

## Storage tombstone correction and preserved failures

The original35 run on the new image failed 32 of 35 cases at the obsolete
public-client ordinal check. The first full640 run passed 31 and failed four
nonempty Weak-lifetime cases in observer6ea. Actual `so_storage_release`
clears offset, rounded extent and strong count while preserving requested
bytes in the Weak tombstone. Observer6ea incorrectly required requested bytes
to be zero as well. These original failures and original source bytes are
preserved, and neither is reclassified as a successful execution.

`owner640_storage_tombstone.py` is a separately named successor. Its exact
inverse restores6ea: expired offset/rounded remain zero, requested bytes are
bounded by the pool. The successful transition checker additionally requires
same-generation requested bytes to remain unchanged and forbids payload
revival. All other live extent, strong/weak, Version, dtype, graph, cycle and
partition checks are unchanged. A real six-frame native control reproduced
12 requested bytes surviving release of a16-byte extent, Weak-lock refusal,
final Weak destruction and EOF0; four malicious copied variants were refused.
The original63 snapshots,64 isolated ledgers and24 malformed copies were also
replayed through the successor with their original outcomes. Four original
setup events remain explicitly snapshot-only.

## Portable execution

Run from the repository root after the actual11-unit native build. The output
directory passed to the observer must not already exist. The original report
is emitted directly by the frozen entry and is never rewritten by the wrapper.

```sh
python3 -B tests/dtype_frozen35_accounting/run_frozen35_accounted.py \
  --legacy-directory tests/tensor_object_oracle \
  --native-source build/native-dropout/runtime.hepta \
  --observer-output build/frozen35-dtype-node/523 \
  --observer-report build/pytorch-native-dropout-frozen35.json -- \
  build/native-dropout/tensor-dtype-dropout-session \
  build/pytorch-native-dropout-frozen35-original.json \
  --owned-reference tests/owned_tensor_oracle/pytorch_owned_tensor_oracle.py \
  --session-reference tests/pytorch_session_oracle.py \
  --reference tests/pytorch_oracle.py \
  --lock compatibility/pytorch-oracle-lock.json --repo . \
  --expected-native-sha256 830f0b4ef8d6ad8c6a105be56d383628b45c84e3e43905fee6812c7c52d35a51 \
  --trace-directory build/frozen35-dtype-node-wire/523

python3 -B tests/dtype_frozen35_accounting/check_frozen35_accounted.py \
  --observer-report build/pytorch-native-dropout-frozen35.json \
  --original-report build/pytorch-native-dropout-frozen35-original.json

python3 -B tests/dtype_frozen35_accounting/check_owner_snapshots.py \
  --observer tests/dtype_frozen35_accounting/owner_accounting523.py \
  --journals build/frozen35-dtype-node/523/owners \
  --output build/frozen35-dtype-node/523/copied-controls \
  --expected-profile dtype-native-dropout-523
```

The exact report gate requires35/35,6344 frames,2590274 numeric comparisons,
283 explicitly retained state-Version differences, zero parameter-Version
mismatches, all11 original provenance checks, and execution of all35 original
EOF and post-close Version gates. The four positive copied events and18
original malicious-copy rules retain their original expected outcomes. Only
their observer pin, profile name and actual descriptor width are adapted.

The ROG actual second run met every one of these gates with the fixed image;
it is explicitly not a new native build. CI must run them on its cold-built
image and retain the original inner report, wrapper report, full owner
journals and raw transport directories. The existing journal collector can
copy both reports into the new roots; old profile collectors remain unchanged.

## Evidence status

Actual live32 also ran with its unchanged554671 source and passed32/32 with
all11 provenance checks. Final original35 report SHA is
`206a6a62a2c40a7aeb2964257388daefa051326c951c57f65a1a7a243b917d98`;
wrapper SHA is
`3d61a7f685719d42b004d89433431f234707d4af1663f2b8a14d6e474d23de5d`.
The complete remote raw archive is separately retained and verified:
450 members,94644488 expanded bytes,15587512 archive bytes,
SHA `1c3ef93dd3e7910b10dd12287959b6e841e5b2e2e71ffb09793fcc1597b4d993`.
This source package does not assert that the separate local archive transfer
has finished. Its byte-copy receipt is issued independently after full SHA,
XZ EOF and every archive member are verified locally.
