# F32 abs backward: preserve bit patterns at the tagged edge

Deploy-disabled continuation of the failed [#64](https://github.com/TrillionniumFoundation/HeptaTorch/pull/64) prototype at `9bc5faf1345ef88d7ac378329f9c55451d83ef1b`. It fixes the observed raw F32 signed-zero and NaN differences in that prototype's direct contiguous CPU abs gradient path. It does not admit the candidate, make a full F32 autograd claim, or activate anything.

## Native change and exact checks

The pinned PyTorch 2.15 CPU rule is `grad * self.sgn()`. The #64 candidate canonicalized NaNs in its generic F32 multiply and erased the sign of an initial negative zero by adding it to an initialized positive zero. This successor changes only two HIA bodies:

* In `back_unary`, only internal tag 201 (F32 abs) uses the specialized F32 word product. The real sign factor is −1, +0 or +1. The product quiets an incoming NaN while preserving sign/payload, yields x86 negative indefinite for infinity times zero, and preserves the appropriate signed zero. It leaves generic `mulf` and `addf` unchanged.
* In `gradient_direct`, only the new tag joins the existing first-contribution bitmap path. This makes the first edge write a copy, not a floating-point add to an artificial +0. Other existing tag rules are unchanged.

The source transform binds the #64 source/manifest and full 15-unit roster, with byte-exact inverses. Only the objects in partitions 02 and 03 differ from #64. Two cold builds produced the same ELF `3e427044155ae8787621a926425a1ab0911611c69cd2f3da7f08417dfa9d7774` and all 15 identical objects.

The pinned live PyTorch wheel was `2.15.0+cpu` at `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`, using AVX2 and one thread. A separate frame reader reproduced these observations from physical captures:

| Scope | Result |
| --- | --- |
| Previous F32 abs corpus | All 142 frames replayed. The 18 earlier signed-zero and NaN raw-word differences are gone. Retained/default backward, exact targeted forward/gradient reads and the actual OOM refusal plus byte-identical forward retry were retained. |
| Additional direct-leaf accumulation | 40 exact gradient reads using different signed qNaNs/sNaNs and signed-zero seeds in both arrival orders. Lengths 1, 7, 8, 9, 15, 16, 17, 31, 32 and 33 exercise scalar and vector/tail sizes. The tested first and repeated accumulation words match. |
| Downstream nonleaf | At the retained abs operand, raw adjoint words match. The subsequent multiply backward still canonicalizes four NaN words on the leaf, which does not match PyTorch. |
| Fan-in | Two graph operand orders still canonicalize four NaN words each at the shared leaf. Neither branch orders nor expectations were relaxed. |
| Partial backward transport | All seven additional whole-owner byte predicates from #64 still differ on nested refusal values 6/10. The existing partial wire58 contract permits prefix commit; this extra observation is not proof of a leak or a changed public gradient. It is independent of F32 bits. |

Total: 15 isolated native children, 422 frames, 73 value reads, 66,388 compared words, with 12 remaining discrepancies in the three broader graph controls. No child-exit failures. Both the passing bounded cases and the failing boundaries are in [the readback](physical-readback.json). The old #64 physical failure remains unchanged. Neither wider autograd, all layouts, nor the original 35 suites were qualified with this image. No CUDA, API registration or default-path claim.

## Reproduction

Use the exact #64 prototype scratch source and ELF as `F32_PREDECESSOR_SOURCE` and `F32_PREDECESSOR_ELF`; keep the original #63 joint ELF as `JOINT_ELF`. Use the exact #62 oracle loader and adjacent `prepare.py` pinned by the observer, and the #61 runtime builder plus pinned stack tools. Readback needs no native binary or Torch.

```sh
python3 -I -B prepare.py "$F32_PREDECESSOR_SOURCE" "$SCRATCH/f32-bits-source"
python3 -I -B "$REPO/tools/current_runtime/build.py" \
  --source-root "$SCRATCH/f32-bits-source" --manifest "$SCRATCH/f32-bits-source/manifest.json" \
  --compiler "$COMPILER" --linker "$LINKER" --output "$SCRATCH/f32-bits-build"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$TORCH_PYTHON" -I -B -W error observe.py \
  --loader "$ABS62/qualify.py" --lock "$REPO/compatibility/pytorch-oracle-lock.json" \
  --install "$TORCH_INSTALL_REPORT" --protocol "$REPO/tests/integer_tensor_review/wire/protocol.py" \
  --prior "$JOINT_ELF" --base64-elf "$F32_PREDECESSOR_ELF" \
  --base64-readback "$ABS64/physical-readback.json" \
  --candidate "$SCRATCH/f32-bits-build/heptatorch" --raw "$SCRATCH/f32-bits-observation"
# Expected observer status is FAIL because of the retained wider boundaries.
python3 -I -S -B readback.py "$SCRATCH/f32-bits-observation"
```
