# Mixed10 original35 accounting source

This independent namespace targets source `5f8ab1f7ca1593dc21b8b6a3537e210cfbc42b37ddefcb322239c5a316d88c49` and ELF `69045063c00395d07fc6c0101ed1b8aba18376cb1daea54e0a01c2c5caeb49b6`: 556 core functions, 11 actual native compilation units, 640-byte descriptors, HTMXAD02 / HTOMXD02.

The full35 domain consists of the unchanged original F32 commands 1 through 56. Ordinary mul/div now keep original Tensor references in a real private 32-byte Hold at node584/592. Both Saved slots contribute separately to the exact node600 and node208 sums. Root reachability and reset closure traverse these real Tensor references. No unexplained allocation remainder is accepted. All full 80-word snapshots also pass the frozen general mixed10 checker. Dtype, history99/101, mask and I64 view roles outside this command domain are explicitly rejected by this profile.

The old 355/428/446/523/544 source profiles remain unchanged. The ten selected original pure definitions are bound to their exact legacy source. Snapshot has seven explicit modifications (one is width), and transition has one reset-closure modification; their inverses restore the complete original ASTs. Five adapted files have whole-byte inverse evidence in source-adaptation.json. The original35 numerical, metadata, Version, optimizer-state, EOF and post-close assertions remain in the unchanged original scripts.

## Exact entry points

Run from the repository root with the fixed native image and its actual source. Paths below are examples; the scripts resolve their sibling dependencies portably.

```sh
python tests/mixed_frozen35_accounting/run_frozen35_accounted.py \
  --legacy-directory tests/tensor_object_oracle \
  --native-source build/mixed-integer-autograd/runtime.hepta \
  --observer-output build/frozen35-mixed-node/556 \
  --observer-report build/pytorch-mixed-integer-frozen35.json -- \
  build/mixed-integer-autograd/tensor-mixed-autograd-session \
  build/pytorch-mixed-integer-frozen35-original.json \
  --owned-reference tests/owned_tensor_oracle/pytorch_owned_tensor_oracle.py \
  --session-reference tests/pytorch_session_oracle.py \
  --reference tests/pytorch_oracle.py \
  --lock compatibility/pytorch-oracle-lock.json --repo . \
  --trace-directory build/frozen35-mixed-node-wire/556 \
  --expected-native-sha256 69045063c00395d07fc6c0101ed1b8aba18376cb1daea54e0a01c2c5caeb49b6

python tests/mixed_frozen35_accounting/check_frozen35_accounted.py \
  --observer-report build/pytorch-mixed-integer-frozen35.json \
  --original-report build/pytorch-mixed-integer-frozen35-original.json

python -I -S tests/mixed_frozen35_accounting/check_owner_snapshots.py \
  --observer tests/mixed_frozen35_accounting/owner_accounting_mixed.py \
  --expected-profile mixed-integer-556-fixed35-F32-Saved \
  --journals build/frozen35-mixed-node/556/owners \
  --output build/mixed-frozen35-copied18
```

The unmodified live32 runner remains a separate command. The original collector can mirror both original and outer report bytes and index the new journal and wire roots, independently of every older profile.

## Evidence boundaries

The included local reports are only a real 9-frame native smoke and the original 4 positive / 18 copied-negative control rules applied to its journal. The two source reviews executed no native program or Torch. They do not substitute for full35.

The original source-adaptation receipt is preserved as PREPARED_NOT_RUN. Separate ROG jobs later execute the original live32 and complete full35 on exact bytes from the independently recorded portable cold build. Their complete original reports, process identities, raw journals, wire logs and verified archive will be provided as a separate evidence packet; this source manifest makes no new execution claim. All 11 fixed Torch provenance checks and all original35 admission thresholds remain mandatory.

This component is not general mixed-I64 full35 coverage or complete PyTorch. The independent general owner controls are separately scoped. Overall status remains HOLD, 2590 API denominator, zero globally closed schemas, full=false.
