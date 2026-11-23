# Generator131: retained replay on the exact joint image

This qualifies the original Generator131 suite, not the entire current
runtime gate. The exact joint #63 executable `5af79385…` was used for one
actual `--group all` invocation and a separate, read-only independent
physical audit. The canonical source, executable, gate, historical source
and report files are unchanged. No existing PASS was transferred.

| Group | Cases | Actual evidence |
| --- | ---: | --- |
| Original pinned PyTorch | 24 | 2,297 frames, 26,466 exact values, 4,181,312 serialized state bytes |
| Original native Generator | 11 | 1,715 frames, 121,611 exact output values, 471 full state checks, 72 rollback controls |
| Original entry/ABI | 59 | 71 independently audited tool/probe processes |
| Original retained dispatcher controls | 37 | 6 live production cases and 31 separately audited ABI probe processes |

The independent audit verified 37 persistent children, 4,035 physical
frames, and 41,573 bound files. It reconstructed the real `HIASTKR1`
request with all 15 candidate objects and the one freshly compiled probe
object, and the real `HIACGS01` request with those same 15 objects reversed.
Both match actual link process input and output bytes with clean exit/EOF.
The resulting ELFs are distinct from each other and from the production and
baseline ELFs. The 31 dispatcher probes reused the first link's actual
candidate probe, including full raw before/after owner checks. The original
Torch reference and all 11 provenance checks ran, and there were four
original nondeterministic entropy observations. No random values were
substituted or replayed from the baseline.

## Exact identity boundary

The portable original qualifier and independent reader are authenticated
by source hashes and the original package verifier. Both use the separate
all-15, four-body reversible #63 source projection. The old reader still
refuses the joint build at its exact original manifest check. A narrow
in-memory bridge changes the producer admission's source, image and manifest
hashes and proves the original source-only 712 predecessor separately from
the actual candidate source/cold receipts. The original compiler/linker
proof, original 26-file package proof, exact tool-body predicates, case
payloads, and process/owner predicates remain active. The reader changes
only its two identity strings and its in-memory package-loader target;
it independently rechecks prior admission and the original package binding,
reconstructs both link requests, and audits original frame/owner bytes.
An abs-only build and a preparation receipt submitted as execution
evidence each fail. Neither starts any new native observation.

## Run and read back

Use Python from the repository's pinned CPU Torch wheel, run with `-I -B`,
and use a fresh temporary path for each `--projected` destination. Supply
the exact canonical repository, restored portable qualifier package,
source-only candidate-ready bundle, original26 bundle, fixed stack build,
live entropy-tool build, both approved source transforms, and exact cold
joint build and scoped receipt. These are the normal repo fixtures and
tools; no historical application executable is substituted. The live
entropy receipt in this exact replay is path-bound. The wrapper pins its
recorded SHA and audits its actual tool streams, authoring, and product
bytes; an independently relocated/bootstrap run requires its own explicit
fresh receipt profile. Do not rewrite the original report paths.

```sh
# Use an ample executable, local TMP. Choose fresh output paths.
args=(--repo "$CANONICAL_REPO" --package "$PORTABLE_QUALIFIER"
      --candidate "$SOURCE_ONLY_CANDIDATE" --original "$ORIGINAL26"
      --stack "$STACK_BUILD" --tool "$LIVE_ENTROPY_BUILD"
      --prepare "$DUAL_PREPARE" --logical "$LOGICAL_TRANSFORM" --abs "$ABS_TRANSFORM"
      --build "$DUAL_BUILD" --scoped "$DUAL_SCOPED_RECEIPT" --output "$TMP/actual")
"$TORCH_PYTHON" -I -B replay_joint_generator131.py prepare "${args[@]}" \
  --projected "$TMP/projected-prepare" --evidence "$TMP/prepare.json"
"$TORCH_PYTHON" -I -B replay_joint_generator131.py execute "${args[@]}" \
  --projected "$TMP/projected-native" --evidence "$TMP/native-receipt.json"
"$TORCH_PYTHON" -I -B replay_joint_generator131.py readback "${args[@]}" \
  --projected "$TMP/projected-readback" --native-receipt "$TMP/native-receipt.json" \
  --evidence "$TMP/physical-receipt.json"
```

Use separate producer/reader processes and reserve at least 700 MB in
the temporary evidence budget. The observed peak was 478,126,080 bytes;
available memory stayed above 5.9 GB. A supervisor polled disk and memory
and would stop before the agreed limits. Large raw process logs and file
trees remained only in temporary storage; this package does not claim
durable custody or transfer of them. [Compact evidence](generator131-joint-evidence.json)
binds the exact report, both receipts and all scoped counts. Paths/timing
and entropy values can differ in another actual execution.
