# Public Saved Tensor lifetime: fourteen fixed observations

The real fixed CPU Torch build observed fourteen recipes with all eleven provenance checks.
The recipes retain an original Saved Tensor after removing its original public binding,
then execute default backward or destroy the graph before releasing the held Saved Tensor.
They include mixed float/int64 mul/div in both orders, an int64 view with holes, and
F32 self-mul/self-div with two Saved slots pointing at the same original Tensor.

Profile09 actually passed thirteen and failed the int64-view/default-backward recipe:
a descriptor whose Tensor row had expired still held its ultimate base strongly through
graph metadata. Profile10 releases that base when the Tensor row dies while preserving
the graph metadata reference. The same fourteen expectations now all pass. Every original
09 failure, first diagnostic-serialization failure, native image, module, frame and
captured complete owner is in the raw history archive. The 09 PASS results from other
suites are not reclassified as having passed these fourteen cases.

The current reader differs from its preceding diagnostic-safe reader only in the exact
observer source identity. That observer differs only in the two state-format tags.
All value, identity, Version, weak-lifetime, full rollback and full RNG assertions remain.
This finite scope does not provide complete PyTorch API or Python-wrapper parity.

Replay from a repository containing this handoff, replacing BUILD and OUTPUT with new
absolute paths (OUTPUT must not exist):

```sh
python3 -I -S -B tests/saved_public_lifetime_review/replay_saved_public14.py \
  --binary BUILD/tensor-mixed-autograd-session --source BUILD/runtime.hepta \
  --build-report BUILD/build.json --protocol tests/retained_nonleaf_review/protocol.py \
  --observer tests/saved_public_lifetime_review/native_observer.py \
  --reference validation/saved-public-lifetime/fixed-reference-original.json \
  --output OUTPUT \
  --expected-source-sha256 5f8ab1f7ca1593dc21b8b6a3537e210cfbc42b37ddefcb322239c5a316d88c49 \
  --expected-native-sha256 69045063c00395d07fc6c0101ed1b8aba18376cb1daea54e0a01c2c5caeb49b6
```

The protocol and owner client remain the repository's fixed original files, SHA-256
7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9 and
dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b.
The byte verifier checks every package and raw member, modes and complete compressed EOF;
it runs no native executable and no Torch. All seventeen whole-system gates remain open.
