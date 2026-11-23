# CPU contiguous F32 tensor abs: opt-in prototype, not admitted

This is a deploy-disabled experiment stacked on the joint [#63](https://github.com/TrillionniumFoundation/HeptaTorch/pull/63) source, not a canonical runtime change or proof of autograd parity. The source transform adds native F32 handling of `(3, 7, tensor)` in a separately built, exact scratch image. The previous joint executable refuses this command for F32 (status 4, no response records). I64/F64 abs and the typed logical-not route are unchanged by the transform. It does not change the historical 35-session qualification or any predicate in its readers.

The transform admits contiguous CPU F32 inputs, including scalar and empty. It reuses the F32 descriptor allocator, native element traversal, sign-bit-clear `absf`, and the existing saved-operand/version and gradient graph paths. Wire unary subtype 7 is distinct from the experimental internal graph ordinal 201; internal ordinal 7 already denotes a reduction. There are five changed HIA bodies across three partitions and no new functions. The aggregate and each affected physical partition are reversible to the exact #63 joint input. Two cold builds emitted identical 15 objects and ELF `f9408443b278cd4c5d2131845ae31b3ad0e03469ca5e6e3c69c031e31694fe46`.

## Retained physical outcomes

Using the pinned official PyTorch 2.15.0+cpu wheel with AVX2 and one thread, the focused observer and independent physical reader covered 11 process cases and 142 request/response frames, with zero child-exit failures. **The parity result is FAIL.**

| Behavior | Observation |
| --- | --- |
| Contiguous forward | Raw words and checked metadata/components matched for the exercised finite, signed zero, subnormal, infinity and NaN cases, including marked-grad forward, no-grad forward, scalar and empty. Checked output Storage and Version identities were disjoint from the input. |
| First-order backward | Both retained and default calls returned nested success; examined finite dense, nonleaf multiply-to-abs, scalar and empty raw gradients matched, including the second accumulation. |
| Signed-zero gradient | Three raw words were `+0` where Torch returned `-0`, both on the first and the second accumulation. The mismatch was not relaxed to numerical equality. |
| NaN gradient | The infinity/NaN case had six mismatched raw gradient words in each comparison, including one signed-zero mismatch and five canonicalized NaN sign/payloads. For example, Torch returned `ffc00000` for the observed infinity-times-zero rows and `7fc45678`/`ffc45678` for observed incoming NaN payloads; native returned `7fc00000`. |
| Saved version and release | Actual Torch raised for stale saved input and for reuse after final backward. Native wire58 returned outer success with nested 6 and 10 respectively. These seven frames changed the physically committed owner pointer and bytes (20–59 bytes after released backward; 102 bytes after stale-version backward), thus failing the additional exact whole-owner equality check. This is not evidence of a leaked tensor or changed public gradient. The existing partial wire58 protocol deliberately permits commit of prefix work on nested errors; these changed bytes were not classified against a public-state contract. No rollback is inferred from a nested value alone. |
| Ordinary transaction failures | The original F32 refusal, invalid later command and actual constrained address-space allocation refusal preserved the exact whole owner. Restoring the real address-space limit and resending the byte-identical allocation request passed a 65,536-word pinned forward read. The same-frame relative-handle forward was also observed. |

[Physical readback](physical-readback.json) lists every differing word, the independently read owner differences, frame hashes and retained control outcomes. The capture archive contains only synthetic focused cases, readback metadata and actual native frames/owner snapshots. The first readback attempt had applied an outer-vs-nested wire58 interpretation that was inappropriate for this ABI; the corrected observation keeps the same candidate image, calls the historical partial command, and preserves all real numeric and physical-owner mismatches without changing the expected raw gradient words. Existing protocol examples: `tests/native_dropout_review/test_dtype_allocation_v2.py` (nested released value 10) and `tests/native_dropout_review/test_history_mul_controls.py` (outer-success partial command with committed prefix).

No admission, frontend registration, higher-order AD, noncontiguous layout parity, or default routing is proposed. The targeted next question is bit preservation through native gradient arithmetic and accumulation, including the first-contribution write and subsequent accumulation. Separately test which public state changes on failed nested wire58 commands before proposing any shell-wide rollback rule.

## Reproduce in fresh scratch directories

Use #63's exact joint source output and executable (see its `prepare_dual.py`), the #61 builder and pinned stack tools, and #62's `tools/current_runtime/opt_in_wide_abs/qualify.py` at commit `5127e9e548736eaf169b590023e8349458a7d1fe` with its adjacent `prepare.py`. The observer binds both loader files and the lock and transport hashes. Use a CPython 3.12 pinned-wheel environment and its existing install report; no wheel is bundled.

```sh
python3 -I -B prepare.py "$JOINT_SOURCE" "$SCRATCH/f32-source"
python3 -I -B "$REPO/tools/current_runtime/build.py" \
  --source-root "$SCRATCH/f32-source" --manifest "$SCRATCH/f32-source/manifest.json" \
  --compiler "$COMPILER" --linker "$LINKER" --output "$SCRATCH/f32-build"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$TORCH_PYTHON" -I -B -W error observe.py \
  --loader "$ABS62/qualify.py" --lock "$REPO/compatibility/pytorch-oracle-lock.json" \
  --install "$TORCH_INSTALL_REPORT" \
  --protocol "$REPO/tests/integer_tensor_review/wire/protocol.py" \
  --prior "$JOINT_ELF" --candidate "$SCRATCH/f32-build/heptatorch" \
  --raw "$SCRATCH/f32-observation"
# The observer exits 1 for the recorded parity failure. The pure reader
# independently reproduces that failure census; successful readback is not parity.
python3 -I -S -B readback.py "$SCRATCH/f32-observation"
```
