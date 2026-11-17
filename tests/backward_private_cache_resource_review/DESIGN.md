# Backward private preserve cache resource test: portable source design

Status: source authoring only. This directory has not been installed or executed.

## Fixed boundary

The numerical runtime remains source c436e21c45355891f8423f174a47e7c40fc8e1d6aef9a112043fac8787255d7a and ELF 0799496559ffdfb051e34bc7a543db238b81d09c970a5cb4a8235151e0335b74, with 658 HIA functions, 13 units, and 672-byte descriptors. This package adds no production operation or numerical expectation.

The source basis is root's source02 driver, 5183c0d6cd0274d29e4fd9e5c7dac1e89d79e8bf35ff0bec6fdcdb1432ee498c. Its original actual run completed one resource case, 22 frames, 46 checks, 23 committed 84-word snapshots, exact full-owner rollback and successful identical-command recovery. That is historical evidence for source02; a new portable wrapper run remains NOT_RUN.

## Dependency and reuse

The fixed sibling package is tests/wrapped_scalar_resource_review. Its run_resources.py is 00becd65f6897286e82723e0e4042eca6c0dbce18e2719855f57af0262aaca53 and package manifest is f38c0cc3c2105e6180798a1fbccbe7fd84426bea18ef95ef14306cd8f63ce98d.

The new wrapper imports that immutable module and invokes its source_check() and fresh_build() directly. It does not invoke its main() or any of its three resource recipes. This retains the full 13-source/object/link-input/ELF and builder-input/output custody checks, including only the explicitly declared relocation of build-report input paths.

The child driver imports the sibling portable07b driver 9925244a326107793255fe8ea2681797e8f7bf00adf6d1b03022a94c6f3a3e72 and source manifest a499508477bd78e6fa28fd09fab364af6eb38ebdcd56c79672d37d57f0e9f031. Its Case, owner observer, protocol and transport remain the existing frozen versions.

## Permitted child adaptation

Only the two parent driver/manifest pins, a required --expected-build-sha256 argument, validation of that argument, and reporting of the actual build hash/function/unit fields are changed from source02. All definitions other than main remain byte-identical. In particular guard, original_module, slot_headroom, consume_extent_budget, geometry and run_case are preserved without a predicate, allocation size, command, limit, numeric or ownership change.

The complete original source01 and source02, exact final-report fix, whole-byte forward/inverse of the portable interface, obligations, independent review and original actual report are retained under provenance.

## Entry point

After root installs the package at its declared repository path with Git modes:

python3 -E -S -B tests/backward_private_cache_resource_review/run_resource.py \
  --repo REPOSITORY \
  --builder-output FRESH_SCALAR07B_BUILDER_OUTPUT \
  --expected-build-sha256 ACTUAL_FRESH_BUILD_JSON_SHA256 \
  --reference ORIGINAL_WRAPPED_F64_REFERENCE32 \
  --probe ORIGINAL_WRAPPED_F64_PROBE32 \
  --output NEW_OUTPUT

--prepare-only runs the source and already-fresh build checks and the child's input admission only; it never starts the HIA executable. Without it, the wrapper executes exactly the new backward resource case, never the original three sibling recipes.

## Qualification and failure behavior

A successful child must have the exact source/image/build identity, one exact named case, error3 refusal with whole-owner/maps/RNG rollback, actual limit restoration and identical-command recovery, committed84Q checks, unchanged inputs, and final EOF0. Both stdout and stderr are retained, including failed or timed-out child attempts. The wrapper does not replace a child FAIL with an aggregate PASS.

The source validator is separate and performs only file, mode, manifest, AST/function-byte and exact inverse checks. It checks the historical source02 report by its complete original hash. It does not rerun a native process or recompute a Torch oracle.

## Staging versus installation modes

The source files are authored in a distinct staging directory. The package manifest declares the intended Git installation mode 0644. The delivery inventory records the real staging modes and explicit 0600/0660 to 0644 installation mapping. Root owns installation; no repository file is modified here. The package validator is intentionally not run against an uninstalled staging tree.

## Remaining execution

No portable validation, preparation, native, Torch or compiler process is run while authoring this package. Final source publication follows root review. Whole-PyTorch status remains HOLD with zero closed schemas.
