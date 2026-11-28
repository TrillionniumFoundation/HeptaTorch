# Native opt-in F64 trunc, floor and ceil

This successor of [#97](https://github.com/TrillionniumFoundation/HeptaTorch/pull/97)
keeps the existing native F32 bodies and dispatches wire 3/19–21 to a
new binary64 body when dtype is exactly F64. It masks fraction bits and
uses exact signed-unit adjustment below 2^52. Signed zeros are retained,
including negative results rounded to zero. F64 trunc preserves raw NaN
words, including signaling NaNs, while F64 floor and ceil quiet them.
The locked AVX2 CPU oracle exercised both scalar and packed cases; the
NaN rule differs from the existing F32 implementation.

The scope is ordinary contiguous zero-offset F64 CPU inputs, ranks
0–4, no gradients. For tested F64 grad/layout/offset cases the existing
upstream factory refuses before dispatch on both images; these are
separate from the wrong-dtype operator refusal. Fresh F64 Storage and
Version, native physical bytes and public typed readback are checked,
including after inputs are released. A separate mixed session chains
native copysign into rounding in both dtypes. The pinned #97 F32 suite
also replays with one guard fixture switched from now-supported F64 to
I64, which remains unsupported.

## Reproduce

Use the exact #97 recipe to regenerate its aggregate `runtime.hepta`
and executable, and to obtain its own #93 parent executable. Use the
same pinned compiler/linker, CPU-wheel Python and pip install report.

```sh
R=tools/current_runtime
D=$R/opt_in_rounding_f64
python3 -I -B "$D/prepare.py" --parent-source "$EXACT_97_SOURCE" --output "$P/source"
python3 -I -B "$R/build.py" --manifest "$P/source/manifest.json" \
  --source-root "$P/source" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/build-a"
python3 -I -B "$D/prepare.py" --parent-source "$EXACT_97_SOURCE" \
  --output "$P/pinned" --expected-build "$P/build-a/build.json"
python3 -I -B "$R/build.py" --manifest "$P/pinned/manifest.json" \
  --source-root "$P/pinned" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/build-b"
cmp "$P/build-a/heptatorch" "$P/build-b/heptatorch"

for BUILD in a b; do
  "$HEPTA_TORCH_PYTHON" -I -B "$D/qualify.py" --repository "$PWD" \
    --parent "$EXACT_97_ELF" --candidate "$P/build-$BUILD/heptatorch" \
    --output "$P/qualification-$BUILD"
done
cmp "$P/qualification-a/qualification.json" "$P/qualification-b/qualification.json"
"$HEPTA_TORCH_PYTHON" -I -B "$D/replay_f32.py" \
  --f32-verifier "$R/opt_in_rounding_f32/qualify.py" --repository "$PWD" \
  --parent "$EXACT_93_ELF" --candidate "$P/build-b/heptatorch" --output "$P/f32-replay"
"$HEPTA_TORCH_PYTHON" -I -B "$D/mixed.py" --repository "$PWD" \
  --candidate "$P/build-b/heptatorch" --output "$P/mixed"
```

See [evidence.json](evidence.json) for local receipts. This does not
enable the candidate by default or qualify autograd, dtype promotion,
arbitrary views/layout, out/in-place, accelerators, or full replacement.
