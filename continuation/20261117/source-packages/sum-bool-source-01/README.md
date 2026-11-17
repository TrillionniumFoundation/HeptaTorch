# Bool output for integral SumDims: source-only candidate

This isolated candidate extends command 115 only for Bool and I64 input with an explicit Bool destination. It is based on the frozen 699-function runtime afd54444849ed10f7b4af53231c9db6542854b70f33e3927f8b05dfbb23ed1e3. The original 699 source, image, reference24 and five dtype controls remain historical originals.

## Concrete implementation

The candidate adds the real HIA function sb_sum_bool. It reuses the bound sd_axis_mask, si_plan, tget and taddr bodies. It creates a fresh owning Bool Tensor with typed_descriptor(..., 0, 1, 1), initializes every output byte to zero, converts each logical input value to canonical zero/one before accumulation, and uses bit.or for Bool addition. This is the Bool operation selected by the fixed CPU SumKernel; it cannot turn integer cancellation or overflow into false. Logical iteration follows actual input strides and offsets, including holes, overlap and zero stride.

The original si_sum_integral body is untouched. si_dispatch_sum gets one explicit dtype1 branch inside its existing Bool/I64-input branch. to_command admits dtype1 while preserving all other packet checks and their order. Four format functions change only HTSINT01/HTOSIN01 to HTSINT02/HTOSIN02. The ABI remains 672 bytes. The prospective body inventory is 700 functions: 693 old bodies unchanged, six old bodies changed, one body added. No actual emission or compiler count is claimed.

The addition is mechanically derived from the complete, actual si_sum_integral HIA body with seven exact edits. body-derivation.json records those edits and the six existing-body edits. body-patch.json contains every before/after body, the complete 699-body inventory and original order, and the future whole-runtime inverse specification. The original integer function remains an independent body.

## Source evidence

The official source review pins sixteen complete files from PyTorch commit 31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457. Explicit Bool chooses a Bool reduction input; I64 is converted before reduction. Bool addition stores back to Bool and has false identity. None/[] and signed-axis handling follow the existing axis helpers. Bool cannot require gradients.

The helper basis contains eleven complete HIA bodies read from immutable Git blob 8c61bb4667af857b0dafe7077c5fe88d291dd6fd (c436 parent) and individually SHA-matched to the actual699 body ledger. It covers real one-byte allocation, new Storage/Version, strided reads, byte stores, and public Tensor publication.

## Future source application

The following command is a source-construction interface, not an executed request:

    python -I -S -B apply_source.py --parent-runtime ORIGINAL_699/runtime.hepta --package THIS_PACKAGE --output NEW_SOURCE_DIRECTORY

The input must be the exact 2312858-byte, mode0600 parent. The package is intended to be materialized with mode0644 for every listed payload. apply_source.py verifies the complete payload set, full parent bytes and all 699 body hashes before constructing a new runtime. It requires a fresh output directory, preserves all unmodified bodies, and proves forward and inverse whole-byte equality in the future output report. It does not parse a Python AST, run an emitter, generate module units, invoke a compiler, execute a Tensor image, or calculate an expected Tensor result. Its Python source has not been imported, AST-checked or executed in this task.

## Qualification boundary

This package is CACHE_ONLY / SOURCE_ONLY_NOT_APPLIED. The local string-level body forward/inverse checks were completed; the whole runtime application awaits the actual parent bytes in a future authorized invocation. New HIA syntax/type verification, unit generation, cold compilation, reference observation, native replay and owner/resource gates have not run.

A new fixed-PyTorch reference is required for Bool raw values, shape/stride, one-byte Storage, Version, no-AD and measured lifetimes. The old I64 output observer cannot be generalized by accepting any width. A new observer must explicitly check Bool output and preserve all old predicates. future-qualification-contract.json proposes a new seven-control denominator; it does not change the old five-control source or results. No old PASS is transferred.
