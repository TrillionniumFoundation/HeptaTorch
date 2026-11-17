# R3 preserve resource controls

This additive evidence qualifies three finite resource compositions on source 3c2c0a125bcac6d47075bf58fc3f78c0e7b9a0b1370757981b99dcd3793b21c6 / image 7841b19051960b6fa13cb98841dbb8e4d6ead97f03e2ad65af415e1a35b03221 (627 functions,13 native units,672-byte descriptors). The first run remains FAIL: its tail-reuse and Saved-error-prefix cases passed, while OOM setup requested an invalid single dimension332548. The successor capped filler chunks at the existing262144 limit and executed only OOM, which passed. Both original reports and every raw frame remain unchanged.

The OOM setup had65536 free pool bytes; the source-bound bound for all prior small allocations was448 bytes, while the intended preserved extent was524300 requested/524304 rounded. Under temporary RLIMIT_AS the same-capacity query passed and backward returned3 with the complete2271888-byte owner, pointer/length,5056-byte RNG and maps unchanged. Restoring the original limit allowed the exact same backward wire request to succeed. The first two successful cases were not rerun.

The original prefix uses two actual op44 nodes then Saved Mul3: result(6,2,0,3), where the final3 is the failing operation code. Its57 variant retains whole-owner rollback. Every remaining committed resource is an actual descriptor page after cleanup. The full84-word committed-state predicates were retained; the old isolated-transition guard still refuses ViewInfo and is not claimed as executed. Whole-PyTorch remains HOLD/0.

## Portable CI invocation

Run from the repository root with an already freshly built exact R3 image and the restored original input4 reference. The output directory must not exist.

```sh
python3 -S -B tests/differentiable_view_preserve_resource_review/test_preserve_resources.py --native-build build/differentiable-view-recovered --profile tools/differentiable_view_recovered_r3/profile.json --protocol tests/differentiable_view_preserve_resource_review/protocol.py --reference /ABSOLUTE/PATH/TO/RESTORED/actual.json --output /ABSOLUTE/NEW/OUTPUT
```

Default --case all executes all3; --case preserve_capacity_oom_recovery is the exact successor selection. --prepare-only validates all source, native source/image,13 unit source/object pins and original reference provenance without starting native or Torch. Do not use -I, because this frozen runner imports siblings.

The source02 manifest remains fd8e0c7473983681cb0951b719cf884cddf8c565dc0d364aa978b17ec926e86b. The independent package-manifest records all21 installed files and explicit0644 mode mapping. The raw archive preserves original modes and all absolute paths inside child JSON. Its member map reproduces source01/02, both invocations, preparation/diagnostics, all consumed runtime/unit/reference inputs, controller receipts, and independent source-review evidence. The archive and manifest exclude no child output; only outer packaging files are separate. This evidence does not update historical fields in the frozen producer profile.
