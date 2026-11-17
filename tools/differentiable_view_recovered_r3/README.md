# Recovered differentiable view producer R3

This package contains the frozen R3 authoring inputs and the HIA emitted by the actual R3 build. Its native image has passed the fixed original 14 cases and the four input geometry cases on the pinned reference. The full project remains **HOLD**, with **0 closed API schemas** out of the existing 2590 denominator. The original 16 bit suites, broader inherited suites, and new resource/OOM controls have not been executed on this R3 by this package.

## Required parent and build invocation

`--previous` must point to a genuine R2 build with source `197bd52db4815f575ea3e9055d3f5556900314bd2f4fbb262e80cea76ea16ef2`, image `13cee2163e8f62081db35e831885714789ac5117fb80c030a1ccaf7194e0821c`, and all 12 source/object units described by its successful `build.json`. The Make integration prepares this at `build/differentiable-view-r2-parent`. A direct 09b parent does not satisfy the R3 loader.

From the repository root, use the following command with actual absolute paths. The output directory must not already exist.

```sh
python3 -B tools/differentiable_view_recovered_r3/build.py \
  --previous /ABS/REPO/build/differentiable-view-r2-parent \
  --repository /ABS/REPO \
  --upstream /ABS/PINNED_HEPTA \
  --compactor /ABS/REPO/tools/owned_optimizers/compact_functions.py \
  --compiler /ABS/STACK/native/stack-module-compiler \
  --linker /ABS/STACK/native/stack-linker-session \
  --output /ABS/CREATE_ONLY_R3_BUILD
```

The frozen build script imports sibling authoring modules; the invocation uses `-B` as the actual build did. `--prepare-only` emits the HIA without invoking the native tools. The regular build compiles all 13 units and links a new image; it does not reuse parent object bytes in the new link. `input-contract.json` lists every actual parent, repository and upstream input by relative role and original hash. A fresh parent's `build.json` contains its own paths, so its admission follows the frozen ParentEmitter schema plus complete source/object pins; the original report hash remains historical custody.

## Exact identities and finite results

- Runtime source: `3c2c0a125bcac6d47075bf58fc3f78c0e7b9a0b1370757981b99dcd3793b21c6`.
- ELF image: `7841b19051960b6fa13cb98841dbb8e4d6ead97f03e2ad65af415e1a35b03221`.
- Actual build report: `47a17c04789e1cdc843ea186de211d4b5b279b526961a0713d24c9a9073b6bdb`.
- Core: 627 functions, 8245 blocks, 29636 instructions; 13 units, 672-byte descriptor, `HTVAD003` / `HTOVAD03`.
- Original 14: 14 passed, 15135 frames, 96461 assertions, all 14 EOF exits zero; report `c3147b161f32a85a87ccf00da47f3d2fc8bc2a5c47e159cf1f7ad1f8938606a1`.
- Input geometry 4: 4 passed, 12353 frames, 77432 assertions, all 4 EOF exits zero; report `59851d848c383305d8e827d0ef3f1ef1b5fba6f513f5ecfda088fa9ced544d68`.

The two original reports are copied byte for byte under `provenance/`. The actual build report retains `runtime_qualified: false`, because compilation alone does not establish runtime qualification. The two separately executed runtime reports establish only the finite scope stated above. Full raw build and runtime archives remain in the separately preserved build/runtime history packages.

## Preservation and installation

All 21 frozen author payloads retain their exact bytes. Their original manifest and single-XZ archive are preserved under `provenance/`; original author files have mode 0600. Git installation copies use 0644 with an explicit per-file mode map. Actual generated HIA, all 13 unit sources, source maps, original build report, and two actual runtime reports have separate copied-byte records. The installed package manifest excludes only itself.

The original builder and emitted custody contain the historical text “original files remain inaccessible”. That text is intentionally preserved as part of the frozen author bytes. Subsequent exact recovery of the original a6f coordinator, package 5179 (309 archive members), and cold01 (795 members) is recorded in `validation/tensor-to-original-source-recovery/` and `validation/tensor-to-original16-recovery/`. It updates the accessibility fact; it does not reclassify the original SIGKILL failures or any R1/R2 native failure.

The old install request `ht-d266-r3-v3-frozen-producer-install-20261116-v1` remains UNKNOWN. This independently authorized delivery uses a separate staging directory and the new canonical name `tools/differentiable_view_recovered_r3/`; the unknown request's old v3 target was not reused or overwritten. Root owns the final repository import. The prior a679 parent recovery, R1 package, and c03c R2 package are unchanged.
