# Native opt-in F32 trunc, floor and ceil

This adds `aten::trunc(Tensor)`, `aten::floor(Tensor)` and
`aten::ceil(Tensor)` as wire command 3 operations 19, 20 and 21,
respectively, in an opt-in successor of [#93](https://github.com/TrillionniumFoundation/HeptaTorch/pull/93).
These kernels use a single HIA loop. The word helper masks fractional
bits, handles signed zero and subnormal words, leaves integral words and
infinities unchanged, and quiets NaNs without losing sign or payload.
Where floor or ceil must round one unit away from zero, it adds signed
one to the integral truncated result. That step is exact in F32 because
this path only runs when a fractional bit existed (magnitude < 2^23).

Test scope: ordinary contiguous zero-offset CPU F32 tensors, ranks 0–4
including scalars and empty dimensions, no gradients. Independent
native Storage/Version is verified against physical bytes and logical
wire readback; the result survives input release. Tested grad-tracking
sources, genuinely non-contiguous sources, positive-offset sources,
and F64 input refuse. Contiguous zero-offset views, other dtypes,
AD, out/in-place, and accelerators are not qualified. Nothing changes
the default runtime, public coverage registry, or full PyTorch decision.

## Reproduce

Start from the exact `#93` recipe to build the parent aggregate source
and executable, to identify its own exact parent ELF and F32 copysign
parent ELF, and to locate its verifier directories. Use the same pinned
compiler/linker and real CPU PyTorch installation. From the repository
root, using absolute paths for the inputs and a new scratch directory:

```sh
R=tools/current_runtime
D=$R/opt_in_rounding_f32
python3 -I -B "$D/prepare.py" --parent-source "$EXACT_93_SOURCE" \
  --output "$P/source"
python3 -I -B "$R/build.py" --manifest "$P/source/manifest.json" \
  --source-root "$P/source" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/build-a"
python3 -I -B "$D/prepare.py" --parent-source "$EXACT_93_SOURCE" \
  --output "$P/pinned" --expected-build "$P/build-a/build.json"
python3 -I -B "$R/build.py" --manifest "$P/pinned/manifest.json" \
  --source-root "$P/pinned" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/build-b"
cmp "$P/build-a/heptatorch" "$P/build-b/heptatorch"

for BUILD in a b; do
  "$HEPTA_TORCH_PYTHON" -I -B "$D/qualify.py" --repository "$PWD" \
    --parent "$EXACT_93_ELF" --candidate "$P/build-$BUILD/heptatorch" \
    --output "$P/qualification-$BUILD"
done
cmp "$P/qualification-a/qualification.json" "$P/qualification-b/qualification.json"

"$HEPTA_TORCH_PYTHON" -I -B "$D/replay_predecessor.py" \
  --previous-verifier "$R/opt_in_unary_copysign/qualify.py" \
  --repository "$PWD" --parent "$EXACT_MAIN_ELF" --f32-parent "$EXACT_84_ELF" \
  --candidate "$P/build-b/heptatorch" --output "$P/qualification-regression" \
  --unary-dir "$EXACT_85_TOOLS" --copysign32-dir "$EXACT_90_F32_TOOLS" \
  --copysign64-dir "$EXACT_90_F64_TOOLS" --square-dir "$EXACT_74_TOOLS" \
  --reciprocal-dir "$EXACT_79_TOOLS" --sign-dir "$EXACT_81_TOOLS"
```

`HEPTA_PYTORCH_INSTALL_REPORT` must identify the actual pinned PyTorch
installation as in #93. The exact #93 verifier is reused with only the
candidate ELF pin rebound. See [evidence.json](evidence.json) for local
build and qualification digests.
