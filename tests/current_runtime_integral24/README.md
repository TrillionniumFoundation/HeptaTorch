# Original Integral24 on the current runtime

This entry runs all 24 original Integral24 cases and their 260 reference events
against the same cold-built 700-function HIA image used by Bool27. It reads the
original fixed-wheel observations. It does not import Torch, rebuild the image
or rerun the original 699-function job.

## Unchanged comparisons and source custody

The runner recovers all 19 native24 source payloads from the committed reviewed
intent, verifies their complete bytes and original modes, and leaves them
unchanged. The original replay module validates the whole package, every
parent native32 payload and all recorded source inverses. Its original
IntegralCase constructor, comparison methods and run method execute directly.
Every case and event remains in order; unsupported observations and actual
mismatches retain their original BLOCKED or FAIL records.

The original reference validator also remains byte-identical. A separate path
adapter calls its original file reader on explicitly relocated files and
returns each file's original logical path. The validator still verifies all
original bytes, modes, manifests, primary-source Git blobs, input inventories,
observations and wheel provenance. Both the real local paths and original
logical records receive final input rechecks. The path adapter never changes
the reference JSON or a recorded Tensor value.

The build admission reuses the fixed Bool27 admission for the complete source,
ELF, all 14 fresh compilation units, link request and 15 completed processes.
The owner bridge has a whole-file inverse for format tags and the 700-function
profile. Its separate Bool-success rule retains the original I64 width-eight
predicate and is needed when executing the historical dtype controls below.
All original Integral24 owner and comparison predicates remain active.

## Inputs and command

~~~sh
python3 -B tests/current_runtime_integral24/run.py \
  --build-directory /absolute/current-cold-build \
  --expected-build-sha256 ACTUAL_BUILD_REPORT_SHA256 \
  --reference /absolute/original-reference24.json \
  --reference24-source /absolute/sum-integral-reference24-source-01 \
  --reference32-source /absolute/sum-dims-f32-reference32-source-01 \
  --integral-source /absolute/sum-integral-source-01 \
  --parent-native-package /absolute/sum-dims-native32-source-03 \
  --pip-report /absolute/original-environment/build/pytorch-install.json \
  --upstream-lock /absolute/original-environment/compatibility/upstream-lock.json \
  --output /new/integral24-current-run
~~~

Add --prepare-only for full source, reference and build admission without
starting the native image. Every output directory must be new.

The source directories are the complete original packages with their original
file modes. In particular, --integral-source is the 26-payload HIA authoring
package with manifest cbdb6b7331f1d4f4cf2e1d88cb9306cc17875d9051feca47c958646f04dec6a6.
The separate 19-payload native24 package is recovered automatically from the
reviewed intent. The original actual reference has SHA-256
6c9df4b579664c89c200b9d796367dd6d41f93f8260e8ff012f17d5041d2278b.

## Historical dtype controls

The original five-control file and all its expected values remain unchanged.
It originally required all five dtype conversions to be refused. The current
runtime adds successful Bool-to-Bool and I64-to-Bool sums, so the two original
refusal expectations now fail. Their actual FAIL reports remain intact.

The new outer report identifies this exact contract change only when both
native commands actually succeed and each old test fails solely at its first
cause-7 expectation. It requires all three remaining old refusals to pass
their original complete owner rollback, RNG and cleanup assertions. It does
not report the historical five-control suite as passing.

The current dtype contract has a separate mandatory gate in
tests/sum_bool_review/native.py: two successful Bool destinations and five
scope refusals, including the three retained historical refusals.

## Result contract

The top report schema is heptatorch.current-runtime.integral24.v1. Its PASS
status requires all of the following:

- Exactly 24 original cases pass, with zero failed or blocked cases.
- All 260 original reference events complete.
- original_controls_contract_changed is 2, with original indices [0, 2].
- retained_refusals is 3, with original indices [1, 3, 4].
- All 29 native children finish with exit 0, stdout EOF and stderr EOF.
- Every protocol frame has complete owner boundaries and all bound inputs
  retain their exact bytes and modes.

The historical_controls field contains the unchanged original five-control
result, including status FAIL, passed 3 and failed 2. This is an explicit
compatibility change, and the separate Bool7 gate checks the current expected
behavior. Full raw requests, responses, owner blobs and transport receipts
remain in the native directory. No previous native PASS is transferred.

This report qualifies the original Integral24 observations on the new image.
It leaves schema_closure and full_transition_recomputed false and the full
project status HOLD.

## Actual 2026-11-18 execution

The first current-image execution passed all 24 original cases and all 260
events. Its report SHA-256 is
0070418f2333edf09d34c94b53a56ed4aa07cd9185b5ba8241f59b5b67dba2a0.
The 29 children completed 14,512 frames with exit 0, both output EOFs and
complete owner boundaries. All 178 bound physical inputs and 122 original
logical input records passed their final rechecks.

The original historical-control result remains FAIL with three passes and
two failures. Controls 0 and 2 returned native cause 0 and one real result;
each failed only its unchanged cause-7 expectation. Controls 1, 3 and 4
returned cause 7 with no result and passed their original complete checks.
The report explicitly records this behavior as the two new Bool success
paths, and does not claim all five historical refusals still hold.

The executed run.py SHA-256 is
845f38cbe7588bc181009bd40637c2597a84fea5fb036ee52fb0af60af90cd4c.
Its source was unchanged throughout the execution. The native source and ELF
remain the fixed Bool700 pair, respectively
b83a8346b162e12c5ed77e0287d626619a18f84bf7a1ab02a2514d30ae4295b4 and
c6afeedc14ba8cc408a62d6fbef0abb5b78dff0039b8099f3449d813e290a90e.
