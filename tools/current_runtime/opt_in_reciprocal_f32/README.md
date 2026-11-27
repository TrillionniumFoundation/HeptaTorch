# Opt-in HIA `aten::reciprocal(Tensor)` for ordinary CPU F32

The exact main source at `e9444902df9878060ab389a70ee957c1b7f4a887` rejects
unary command 3, operation 17. This projection adds that operation to a
separate 776-function native image without changing the main runtime files,
default dispatch, canonical acceptance registry, or existing unary branches.
The loop is HIA: it uses the existing native F32 division kernel with numerator 1.0, explicitly
preserves and quiets NaN payloads, and writes into a fresh F32 Tensor, Storage
and Version. Python only projects source, orchestrates builds, and observes
the independent Torch and native processes.

The verified scope is ordinary owned, contiguous, zero-offset F32 CPU input,
ranks 0–4 (including empty extents), no gradients, and an owned contiguous
result. It does not claim gradients, non-contiguous or offset inputs, dtype
promotion, other dtypes, or `out`/in-place variants. Zero-offset contiguous
views are not part of this test scope; the native guard checks layout and byte
offset, not view provenance.

## Reproduce

Use the exact source and the compiler/linker from the repository build guide,
plus the repository's locked CPU PyTorch installation and pip report. From
the repository root:

```sh
D=tools/current_runtime/opt_in_reciprocal_f32
R=tools/current_runtime
P=/absolute/path/to/fresh-reciprocal-scratch
python3 -I -B "$D/prepare.py" --output "$P/source"
python3 -I -B "$R/build.py" --manifest "$P/source/manifest.json" \
  --source-root "$P/source" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/build1"
python3 -I -B "$D/prepare.py" --output "$P/pinned" \
  --expected-build "$P/build1/build.json"
python3 -I -B "$R/build.py" --manifest "$P/pinned/manifest.json" \
  --source-root "$P/pinned" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/build2"
"$HEPTA_TORCH_PYTHON" -I -B "$D/qualify.py" --repository "$PWD" \
  --parent "$EXACT_MAIN_ELF" --candidate "$P/build2/heptatorch" \
  --output "$P/qualification"
```

The two cold builds produced identical objects and ELF. In qualification,
48 real native requests matched 12,819 raw F32 words from the locked PyTorch
2.15.0 CPU wheel (AVX2, one thread), including signed zeros, subnormal
boundaries, dense signed encoding windows around overflow and subnormal thresholds, rounding inputs, overflow, infinities, and NaN payloads. The
unchanged parent refused all 48 new operations. The candidate refused actual
gradient, non-contiguous, and positive-offset requests with byte-identical
owner state on each refusal. Native public metadata/readback and independent
physical Storage decoding matched Torch, and each result survived releasing
its input. 28 response frames for seven prior F32 unary operations replayed
byte-for-byte on the two images.

These are scoped, opt-in results; neither the complete current-runtime suite
nor a complete PyTorch schema or subsystem is qualified by this slice.
[Execution identifiers](evidence.json) bind the test and frame receipts.
