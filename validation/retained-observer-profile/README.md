# Exact retained446 profile for private-node accounting

This patch admits one additional, fixed native program to the af33 read-only
observer. It preserves every accounting function and class body, all six original
oracle source pins, the five native function pins, and all 18 copied-snapshot
negative controls. The complete byte inverse is in `inverse-proof.json`.

## Fixed input and actual execution

| Item | Bound value |
| --- | --- |
| Baseline Git head | `af33d1065a3b097990e9536df5b22b1a3fb315cf` |
| Baseline tree | `88ed6bd6a41ca0e58e653b3d76718090b3bbba64` |
| New profile | `retained-gradients-446` |
| Native HIA source | `666230f05cc6942470006d1f184135321d8cbc26a6c6ba0bf7e3cf9857fe515b` |
| Native executable | `7568c0a4ec40d82d3a0b46d1e8a27499dd96e8fe3fd15433523131a196da89b5` |
| State / backing tags | `HTRET001` / `HTORET01` |
| Descriptor size | 576 bytes, unchanged |
| Native input archive | `0cc512e60904cf32812efdb1aba166f01ff190cea6966a75a1c50e888325aec5` |
| Restored inputs | 33 exact files under `build/retained-root-cold-01/native/` |
| Native controls | 3 passed; 13 interactive frames, including successful original EOF and post-close assertions |
| Copied JSON controls | 4 positives and 18 exact expected rejections passed |
| Actual snapshot checks | 16 committed snapshots; 39 live Tensor-row observations with flags exactly zero |
| Native memory writes by observer | None |
| 446 live32 / frozen35 | Not run by this handoff |

The native inputs were restored from the existing checked archive. This is not a
new compilation or a claim that the 446 PyTorch oracle has already run.

`raw-controls.tar.xz` contains all 35 original execution and restoration files.
Every archive member was read back and byte-compared. Absolute execution paths in
the reports are preserved. Reports are not rewritten to appear to originate in a
different environment.

## Four modified files

`profile-446.patch` changes only these existing paths:

- `tests/private_node_accounting/owner_accounting.py`: add the exact image/source/tag profile.
- `tests/private_node_accounting/check_owner_snapshots.py`: add its CLI choice and repin the observer source.
- `tests/private_node_accounting/check_frozen35_accounted.py`: add the exact profile and repin the observer source.
- `tests/private_node_accounting/restore_native_inputs.py`: add the exact archive, prefix, source and executable.

The launcher, three-case native runner, and journal collector are included as
unchanged byte copies for convenience. No workflow or Makefile is modified.
The frozen 35-case driver, all numeric and Version assertions, all tolerances,
original workload counts, and all 11 PyTorch provenance requirements remain
unchanged.

The observed original workload does not issue retain commands 93/94. Consequently
the existing strict Tensor-row flag check remains **exactly zero**. This patch
does not make the observer a general retain-graph observer and does not add a
permissive flag mask.

## Portable commands after applying the patch

Run from the repository root, using new output directories:

```bash
python3 -I -S -B tests/private_node_accounting/restore_native_inputs.py \
  --profile 446 --repo . \
  --output build/observer-retained446/native \
  --report build/observer-retained446/restoration.json

python3 -I -S -B tests/private_node_accounting/test_native_accounting.py \
  --repository . \
  --binary build/observer-retained446/native/tensor-retained-session \
  --source build/observer-retained446/native/runtime.hepta \
  --output build/observer-retained446/native-controls

python3 -I -S -B tests/private_node_accounting/check_owner_snapshots.py \
  --observer tests/private_node_accounting/owner_accounting.py \
  --expected-profile retained-gradients-446 \
  --journals build/observer-retained446/native-controls/owners \
  --output build/observer-retained446/copied-controls
```

The ordinary frozen35 launcher CLI is unchanged. Its explicit native-source and
native-image arguments select this profile. After a real frozen35 run, invoke
`check_frozen35_accounted.py` on the actual original and additive reports, as with
355 and 428. This handoff supplies no fabricated frozen35 result.

## Reversibility and limits

`build_profile.py` accepts the pinned af33 script directory and regenerates the
seven-file candidate. It asserts a unique exact replacement for each change and
reverses all replacements to the original bytes. Every observer function/class
body is also compared separately, both as bytes and AST. The copied-control AST
is equal after reversing only the observer hash and additional argparse choice.

`freeze_qualification.py` reads the already-executed evidence, checks exact case
names, row counts, frame totals, expected rejection rules, tags, flags, image
identity and source custody, and creates the raw archive without rerunning a
native program. Both utilities are development and qualification tools.

The complete PyTorch rewrite remains open. This handoff qualifies one bounded
observer profile; it does not increase the upstream schema coverage claim.
