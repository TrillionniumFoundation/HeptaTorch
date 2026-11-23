# Opt-in dense F32 multiply-backward word profile

This deploy-disabled experiment builds on exact [#64 6bacbeb](https://github.com/TrillionniumFoundation/HeptaTorch/commit/6bacbebd627d3bbc6bb889e555c0c364ebfa22e1). It fixes the pinned CPU F32 single-NaN and infinity-times-zero cases at the two saved-operand multiply sites of the existing real tensor multiply-backward tag 3. The exact overlay changes one HIA body (`scalar_previous_back_binary`) in one of 15 objects; it binds and reverses the whole source transformation. Generic `mulf`, `addf`, fan-in, canonical source and default behavior are unchanged.

The experimental word profile applies only when both input tensors and the binary result are real F32, contiguous, and same shape. It uses the existing multiplication for ordinary values, all out-of-scope inputs, and **both-NaN** pairs. A sole qNaN or sNaN keeps its sign and payload and has its quiet bit set; either order of Inf×zero produces pinned x86 `ffc00000`. This is a measured wheel/CPU compatibility choice, not an IEEE-mandated NaN propagation rule. The pinned upstream implementation uses [`grad * other.conj()`](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/torch/csrc/autograd/FunctionsManual.cpp#L853-L859), with real F32 operands entering the pinned [CPU multiply kernel](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp#L126-L166).

## Physical observations and remaining failures

* **Pinned Torch preflight:** real AVX2 PyTorch 2.15.0+cpu, one thread. Rotated 14×14 raw operand grid at lengths 1/7/8/9/15/16/17/31/32/33, 1,960 rows and 66,248 words across the two active gradient slots. Sole-NaN/invalid-product outcomes were uniform in this census. Distinct dual-NaN payload priority was **not** uniform: both gradient and saved-peer precedence were observed at lengths 7, 15, and 31, in both slots. Other tested lengths chose the peer. See the complete compressed preflight and its digest in the receipt.
* **Known predecessor failure fixed:** the exact nonleaf `abs(x*2)` retained operand adjoint and leaf gradient now both match at all eight words, including the four special leaf NaNs previously canonicalized. The earlier 142-frame baseline and 40 extra direct F32 abs accumulation comparisons remained exact for numeric reads.
* **Native both-slot matrix:** 202 value reads. All 1,472 sole-NaN, 214 Inf×0, 672 ordinary zero and 1,100 other observed words match pinned Torch. All 314 dual-NaN words retain the native fallback and intentionally **fail** raw equality. There is no relaxed comparison.
* **Still open:** both fan-in graph operand orders still have four NaN payload differences each. The seven already observed extra whole-owner predicates on nested partial wire58 continue to differ; these are not evidence of leaked resources or public gradient changes. No op registration, performance, compiler/capability portability, general multiply-forward, noncontiguous/broadcast or complete autograd claim is made. The extra guard reads have not been performance-qualified.

Two cold builds reproduced the same 15 objects and ELF `9074399826154287c2a97bd7172d6d86562eb5dfe1f5ab641ac33d6a03fb59de`. An independent reader (no Torch, compiler, or runtime execution) bound the real matrix input words from the captured F32 factory frames and verified the actual active backward slot. Across all cases it parsed 16 children, 2,443 frames, 275 value reads, and 70,160 raw compared words. [Readback](physical-readback.json) preserves all failures. This image is not admitted and has not run the 35 mandatory suites.

## Reproduction

Use the prior #64 bits experiment's exact scratch source/executable and physical readback, the exact #63 baseline joint ELF (`JOINT_ELF`), the #61 builder and its pinned tools, an existing pinned Torch CPython 3.12 environment, and the previously bound #62 oracle loader.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$TORCH_PYTHON" -I -B -W error probe.py \
  --observer "$ABS64_BITS/observe.py" --loader "$ABS62/qualify.py" \
  --lock "$REPO/compatibility/pytorch-oracle-lock.json" --install "$TORCH_INSTALL_REPORT" \
  --output "$SCRATCH/torch-preflight.json"
python3 -I -B prepare.py "$PREDECESSOR_SOURCE" "$SCRATCH/mul-source"
python3 -I -B "$REPO/tools/current_runtime/build.py" \
  --manifest "$SCRATCH/mul-source/manifest.json" --source-root "$SCRATCH/mul-source" \
  --compiler "$COMPILER" --linker "$LINKER" --output "$SCRATCH/mul-build"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$TORCH_PYTHON" -I -B -W error observe.py \
  --loader "$ABS62/qualify.py" --lock "$REPO/compatibility/pytorch-oracle-lock.json" \
  --install "$TORCH_INSTALL_REPORT" --protocol "$REPO/tests/integer_tensor_review/wire/protocol.py" \
  --prior "$JOINT_ELF" --base64-elf "$PREDECESSOR_ELF" \
  --base64-readback "$ABS64_BITS/physical-readback.json" \
  --candidate "$SCRATCH/mul-build/heptatorch" --raw "$SCRATCH/mul-observation"
# An observer exit of 1 is expected: it preserves the dual-NaN and fan-in failures.
python3 -I -S -B readback.py "$SCRATCH/mul-observation"
```
