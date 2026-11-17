# Portable original 07b resource controls

This package prepares the already observed three finite resource recipes for a repository checkout. It does not run them during package preparation. The original namespace, driver, owner equations, source obligations and historical evidence remain independently verifiable.

## Layout

- `original07b/`: all 59 original payloads plus the unchanged source manifest (fefb9247...).
- `portable07b/`: the same 59 payload names and 58 identical payloads; only four explicit driver interface edits are permitted. A separate source manifest binds this version.
- `run_resources.py`: admits an actual fresh scalar07b portable build before launching the original recipes.
- `validate_package.py`: source/member/mode/whole-inverse and exact recipe checks only.
- `provenance/`: original b31 build receipt, actual historical request, shape-parent basis, fixed root portable-builder source, complete byte inverse and explicit mode/path mappings.

## Interface change

The original driver pins the full historical build.json SHA b31a45a850f06196432b2786ad0a39b56088ccf6c7b100d0da3517a44de5ba48. Fresh build receipts have different declared paths. The derived driver adds required --expected-build-sha256, uses that value at the same complete-SHA binding point, validates its syntax, and reports that current receipt. BUILD remains the unchanged historical constant. Every other predicate and all three recipe bodies remain byte-identical.

The wrapper does not treat an arbitrary caller hash as a fresh-build qualification. It requires the fixed 9f6f02a7 portable builder's success.json/report.json, real 13 compiler and one linker completion metadata, all declared before/after inputs, the exact path-relocation map, every fixed original build field, the ordered 13 source/object pairs, the actual ELF, source/toolchain pins, complete HIASTKS1 link bytes and complete output inventory. Only the explicitly declared input paths and path-dependent parent/current execution receipt hashes may vary.

## Source check (no native, Torch, compiler or emitter)

    python3 -I -S -B tests/wrapped_scalar_resource_review/validate_package.py --output build/wrapped-scalar-resource-source-check.json

The output file must be new. The preparation receipt for this package does not claim that the fresh-builder admission branch or resources ran.

## Future CI invocation

    python3 -I -S -B tests/wrapped_scalar_resource_review/run_resources.py \
      --repo . \
      --builder-output "$WRAPPED_SCALAR_BUILD_OUTPUT" \
      --reference "$WRAPPED_SCALAR32_ACTUAL" \
      --probe "$WRAPPED_SCALAR32_PROBE" \
      --expected-build-sha256 "$WRAPPED_SCALAR_BUILD_SHA256" \
      --output build/wrapped-scalar07b-resources

builder-output is the directory containing the completed scalar07b success.json, report.json, command.json and native/. Reference/probe paths are explicit and may be repository-relative. Their original full SHA pins remain unchanged. The expected build SHA must come from that builder report's new_build_report.sha256; the wrapper independently validates the entire custody chain.

The wrapper output must not exist. It creates only its outer directory; the original driver creates observed/ itself. The same three cases run in the original order with no --case selection. --prepare-only validates the already completed fresh build and invokes only the original driver's input preparation, without creating a native Session.

The wrapper accepts -I because it loads its sibling validator by explicit filename. Its child uses the original -S -B interface so the unchanged sibling imports in the resource driver work. Do not add -I to that child command.

## Make/CI integration suggestions (root owns edits)

Add one target depending on the complete scalar07b portable builder success receipt, this package manifest and all source payloads, plus the original32 actual/probe. Pass the current build receipt SHA explicitly. Keep stdout/stderr in build-level files outside the fresh observed output. Preserve the actual command exit code when copying logs to CI stdout.

Always export the entire build/wrapped-scalar07b-resources directory, the source-check report, and the target's Make stdout/stderr. The wrapper keeps original child reports and raw files under observed/, plus its own command, before/after custody and complete child stdout/stderr. A FAIL remains a FAIL; no report is replaced with historical PASS.

## Qualification boundary and preserved history

The historical 07b run has three PASS cases, 243 frames, 771 checks, 250 committed 84Q snapshots, 54,698,688 rollback bytes, 1,203,328 full RNG bytes and three EOF exits. Those numbers belong to the original run and are not asserted as a new portable execution. Its 509-member archive and manifest are external immutable evidence recorded in provenance/historical-evidence.json.

These controls cover the original entry routes, original8B Saved-scalar/temporary ownership, and forward114 allocation failure/recovery only. They provide no new backward private-preserve-cache allocation-failure coverage. All old isolated-transition ViewInfo exclusions, fullowner equations, dtype3 restrictions, numeric reference bytes and original failure history remain unchanged. Whole PyTorch gate: HOLD; closed schemas: 0.
