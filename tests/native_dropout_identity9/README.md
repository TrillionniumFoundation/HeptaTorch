# Nine actual empty NativeDropout history recipes

The immutable runner replays the nine independently recorded PyTorch recipes, including five distinct weak-lifetime variants for one recipe. It executes 13 native processes and preserves all request/response streams, full owner snapshots, original EOF outcomes and source/build inputs.

The candidate11 run passed 963 frames and 6,888 assertions, with 8,388,288 bytes of exact rollback comparison and 353,920 bytes of full MT state comparison. Counts overlap other suites and are not upstream API counts. Python only encodes requests, reads native bytes and checks the immutable actual reference; all Tensor/Storage/Version, arithmetic, history, backward and lifetime behavior is native HIA.

## Reproduce against a newly built candidate11

```sh
python3 -I -S -B tests/native_dropout_identity9/replay_actual_identity9.py \
  --binary "$NATIVE_BUILD/tensor-dtype-dropout-session" \
  --source "$NATIVE_BUILD/runtime.hepta" \
  --build-report "$NATIVE_BUILD/build.json" \
  --protocol tests/retained_nonleaf_review/protocol.py \
  --reference validation/native-dropout/reference/torch-empty-dropout-identity.json \
  --output "$FRESH_REVIEW_DIRECTORY"
```

Every output directory must be new. The fixed runner enforces the exact source, ELF, 11 compiled units, 523/6881/25112/640 build statistics, full actual reference identity and all eleven provenance predicates. This does not rerun PyTorch; the original actual reference and its recorded provenance remain frozen.

## Original failures and scope

Candidate07 genuinely failed the legal existing-optimizer audit recipe. Its original complete archive and the follow-up diagnostic are retained under `history/candidate07-original-failure/`. Candidate10 passed these nine recipes but failed independently recorded optimizer-layout cases in the producer package. Its original nine-recipe result is retained separately; passing this suite does not qualify other cases. Candidate11 changes only fixed build identities in this runner, with an exact inverse patch receipt. No recipe or expected result changed.

The original candidate07 source-only review and original execution failures remain in the archives. Source-only review never stands in for native execution. Absolute Python graph sequence numbers, Python wrapper lifetimes, warning text and exception text are outside this native replay's claimed surface. Whole PyTorch parity remains open.
