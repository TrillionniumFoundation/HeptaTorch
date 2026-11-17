# Backward private preserve cache resource control

This is one resource control for the fixed final 07b runtime. It tests actual
backward private preserve allocation failure, complete owner/maps/RNG rollback,
restoration of the owned child's address-space limit, and an identical successful
backward command afterward. It adds no numerical oracle.

The source02 basis actually passed one case, 22 frames, 46 checks, and 23 committed
84-word owner snapshots. That original report is retained byte-for-byte. The
portable wrapper and validator in this package have not been run while packaging.

## Required sibling package and fresh build

Install this package at tests/backward_private_cache_resource_review and keep the
fixed tests/wrapped_scalar_resource_review package unchanged. Its complete
source_check and fresh_build functions are reused directly. They admit the
already executed scalar07b portable build's 13 real unit compilations, link
request, ELF and complete input/output custody; this wrapper does not compile.

The declared dependency pins and scope are in profile.json, DESIGN.md and
provenance/parent-dependency.json.

## Invocation after installation

python3 -E -S -B tests/backward_private_cache_resource_review/run_resource.py \
  --repo REPOSITORY \
  --builder-output FRESH_SCALAR07B_BUILDER_OUTPUT \
  --expected-build-sha256 ACTUAL_NATIVE_BUILD_JSON_SHA256 \
  --reference ORIGINAL_WRAPPED_F64_REFERENCE32 \
  --probe ORIGINAL_WRAPPED_F64_PROBE32 \
  --output NEW_OUTPUT

Add --prepare-only to admit source, fresh-build and child inputs without starting
a native Session. The exact current build report hash is required; the historical
build hash cannot silently substitute for a relocated fresh report.

For source custody alone after installation:

python3 -E -S -B tests/backward_private_cache_resource_review/validate_package.py \
  --repo REPOSITORY --output NEW_SOURCE_REPORT_JSON

Neither command has been executed in the source-only packaging step.

## Preserved semantics and failure evidence

driver.py differs from the actually executed source02 only at five interface or
reporting edits: two parent source pins, explicit build argument and its syntax
check, and current build fields in the report. Its six non-main function bodies,
including run_case, geometry, slot and extent accounting, are byte-identical.

The validator checks those definitions and complete forward/inverse bytes. It
also retains source01 and the exact source01-to02 EOF report-persistence fix.
A failing child, bad EOF, changed input or missing record remains a failure and
its stdout/stderr/report are retained. No original three resource recipes execute.

The current package manifest specifies Git installation mode 0644. Actual staging
modes and original provenance modes are mapped explicitly in the delivery and
mode custody records; staging is not an installed or validated checkout.

Full raw evidence of the original source02 run is owned and sealed separately by
root. The original 77,420-byte report is included here as a fixed historical
binding, not as a substitute for that raw archive or a new portable run.
