# Opt-in first-order F32 square graph experiment

This projection extends the exact [no-grad square #74](https://github.com/TrillionniumFoundation/HeptaTorch/pull/74)
candidate at `82184e4401eebe1d69ed8742485a9f603fd49146`. It does not modify
the parent directory, the default runtime, canonical accounting, or any
existing graph helper. The established no-grad path is unchanged. The
requires-grad path reuses the native F32 multiplication node with both
operands referring to the same input. The existing native engine performs
both derivative contributions and accumulates them into the leaf's gradient.

The verified scope is **bounded finite dyadic input and seed fixtures** on
owned, contiguous, zero-offset CPU F32 leaves with ranks 0–4, including empty
tensors. It is a functional experiment, not full `PowBackward` equivalence:
the runtime's geometric admission does not enforce this numerical test
scope. Specifically, at input `0x1p-74` and seed `0x1p-76`, PyTorch
produces the smallest positive subnormal gradient while this native multiply
graph produces zero. The observed mismatch is retained as a separate
negative result. No claim of higher-order AD, in-place/alias behavior,
general dtype/layout, or a ready-for-default implementation is made.

## Reproduce

Start from the #74 head plus this directory. Use the pinned repository HIA
compiler and linker, the locked Torch installation, and a fresh scratch
directory `P`. `SQUARE_NO_GRAD_ELF` must be the exact predecessor ELF.

```sh
D=tools/current_runtime/opt_in_square_f32_ad
R=tools/current_runtime
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
  --parent "$SQUARE_NO_GRAD_ELF" --candidate "$P/build2/heptatorch" \
  --output "$P/qualification"
```

Two cold builds matched all 15 objects and the ELF. Fourteen scoped fixtures
executed two separately seeded backward calls each, accumulating into the
same input leaf. On the frozen CPU wheel, both the 660 forward words and the
660 resulting gradient words matched bitwise. The exact predecessor
actually refused all fourteen requires-grad requests without owner changes.
The old and new images produced identical native response bytes for 148
frames across all 37 original no-grad cases. See [evidence.json](evidence.json).
