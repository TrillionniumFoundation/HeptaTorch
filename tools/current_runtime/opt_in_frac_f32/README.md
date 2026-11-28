# Native opt-in F32 frac

This adds `aten::frac(Tensor)` to wire command 3 / subop 22 on top of
[#107](https://github.com/TrillionniumFoundation/HeptaTorch/pull/107).
It reuses the native F32 rounding allocator and loop; the word helper
uses the existing bit truncation and native F32 subtraction for finite
fractional values. It returns positive zero for finite integral inputs
(including negative zero), preserves nonzero subnormal sign and
payload, returns PyTorch's negative quiet NaN for either infinity, and
quietens input signaling NaNs while retaining input sign and payload.

Qualification is ordinary contiguous zero-offset CPU F32, rank 0–4,
no gradient. It verifies raw logical and physical bytes, exact shape
and stride, fresh native Storage/Version and release-safe readback.
Real grad/offset/noncontiguous and F64 requests for frac refuse.
Predecessor F32/F64 rounding responses are checked byte-for-byte on
the same image; F64 frac, AD, out/in-place and other layouts/devices
are not qualified. The canonical default and acceptance are unchanged.

## Reproduce

Use #107's exact projection and build recipe to obtain the predecessor
aggregate source and ELF. Use its locked real PyTorch CPU-wheel Python
and actual pip report, pinned compiler and linker, and new scratch:

```sh
R=tools/current_runtime
D=$R/opt_in_frac_f32
python3 -I -B "$D/prepare.py" --parent-source "$EXACT_107_SOURCE" --output "$P/source"
python3 -I -B "$R/build.py" --manifest "$P/source/manifest.json" \
  --source-root "$P/source" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/build-a"
python3 -I -B "$D/prepare.py" --parent-source "$EXACT_107_SOURCE" \
  --output "$P/pinned" --expected-build "$P/build-a/build.json"
python3 -I -B "$R/build.py" --manifest "$P/pinned/manifest.json" \
  --source-root "$P/pinned" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/build-b"
cmp "$P/build-a/heptatorch" "$P/build-b/heptatorch"

for BUILD in a b; do
  "$HEPTA_TORCH_PYTHON" -I -B "$D/qualify.py" --repository "$PWD" \
    --parent "$EXACT_107_ELF" --candidate "$P/build-$BUILD/heptatorch" \
    --output "$P/qualification-$BUILD"
done
cmp "$P/qualification-a/qualification.json" "$P/qualification-b/qualification.json"
```

See [evidence.json](evidence.json) for source, build, and receipt hashes.
