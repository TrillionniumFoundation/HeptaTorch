# I64-capable image: original F32 35-case accounting profile

This source handoff is **PREPARED_NOT_RUN for the new full35/live32/18 suites**. It does not transfer the earlier 523-image PASS to this image. The exact current profile is `integer-tensor-544-fixed35-F32`, runtime source `adf6cae67666273aa6fb77989080ca3b1e3632188d2d7956a6d8b71808787e62`, and ELF `b7324639bade3312981d1cbf9c2491b9fcff04fb6c3f808e9c5e1a7c2ca88f08`.

The observer admits only the original fixed35 command domain (1 through 56). It reads all 80 descriptor words and requires every actual word at byte offsets 576 through 639 to be zero. I64 Tensors, I64 view-base roots, promoted temporary Tensors, commands 99/101/103–106, and general new dtype/history ownership are outside this gate. There is no 576-byte projection, residual allocation balance, relaxed Version threshold, or replacement numerical runtime. Independent I64 resource and typed owner handoff suites have their own reports.

The original35 driver, bridge, numerical references, inputs, tolerances, EOF assertions, and post-close parameter Version assertions remain byte-identical. All 18 copied-snapshot refusal definitions are retained. The corrected Storage tombstone rule retains historical requested byte size while requiring dead payload offset/rounded extent to be zero; transitions reject requested-size mutation or resurrection within the same generation. This is the same source-proved rule as the parent 523 profile.

`profile-adaptation.json` contains exact whole-file inverses for each adapted source. The producer has 544 pinned actual HIA function bodies: 509 of the 523 parent bodies are identical, 14 changed, and 21 were added. The 10 original35 construction anchors are identical. Actual caller sets are checked, including the new I64 view/promotion/dropout/scalar helpers. The new I64 retain branch in `record` is guarded by dtype 2 and cannot execute for this observer's admitted all-zero dtype words.

The inherited launcher's descriptive `scope` text mentions its parent 523/640 lineage. The mandatory current execution identity is the separately bound `profile.name`, source SHA, ELF SHA, and full544 source-function map above; no 523 result is accepted as a 544 result. This wording is preserved by the exact inverse and does not replace any machine admission check.

## Files and dependencies

The six runnable scripts plus the profile, legacy rules, tombstone correction, and structural inverse are in `tests/integer_frozen35_accounting`. The exact original `tests/dtype_owner_accounting/owner640.py` (SHA `6ea9e8b33164089034a2a71a03f4850e1ec18ed1a09155d7841f700ef311c833`) is included unchanged; installing over an existing copy is permitted only when bytes/mode are identical. Old 355/428/446/523 namespaces are otherwise untouched.

## Full35 invocation

From a repository containing the unchanged original oracle sources:

```sh
python -B tests/integer_frozen35_accounting/run_frozen35_accounted.py --legacy-directory tests/tensor_object_oracle --native-source build/integer-tensors/runtime.hepta --observer-output build/frozen35-integer-node/544 --observer-report build/pytorch-integer-tensors-frozen35.json -- build/integer-tensors/tensor-integer-session build/pytorch-integer-tensors-frozen35-original.json --owned-reference tests/owned_tensor_oracle/pytorch_owned_tensor_oracle.py --session-reference tests/pytorch_session_oracle.py --reference tests/pytorch_oracle.py --lock compatibility/pytorch-oracle-lock.json --repo . --trace-directory build/frozen35-integer-node-wire/544 --expected-native-sha256 b7324639bade3312981d1cbf9c2491b9fcff04fb6c3f808e9c5e1a7c2ca88f08
python -B tests/integer_frozen35_accounting/check_frozen35_accounted.py --observer-report build/pytorch-integer-tensors-frozen35.json --original-report build/pytorch-integer-tensors-frozen35-original.json
python -I -S -B tests/integer_frozen35_accounting/check_owner_snapshots.py --observer tests/integer_frozen35_accounting/owner_accounting_int64.py --expected-profile integer-tensor-544-fixed35-F32 --journals build/frozen35-integer-node/544/owners --output build/integer-frozen35-copied18
```

The report checker requires 35 original cases, 6,344 original frames, 2,590,274 compared values, 283 explicitly observed state absolute-Version differences, zero direct parameter Version mismatches, all 11 exact provenance checks, and all original EOF/post-close gates. The collector may reuse the unchanged transport-only collector with independent `build/frozen35-integer-node` and `build/frozen35-integer-node-wire` roots and must mirror both new outer/inner reports plus the actual installation report.

The unchanged original live32 oracle is invoked separately with `--native`, `--source`, `--expected-native-sha256`, `--expected-source-sha256`, `--repo`, and `--output`; it is not folded into the original35 count.

Whole PyTorch remains **HOLD**, full closure is **0**, denominator **2590**, and full rewrite complete remains false. This packet has source custody and a source-only peer review, not new hosted qualification. Native source and ELF are supplied by the separately verified I64 producer and are not rebuilt by these observer scripts.
