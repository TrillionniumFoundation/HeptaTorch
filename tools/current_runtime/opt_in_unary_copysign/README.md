# One opt-in CPU runtime for unary F32 and F32/F64 copysign

This combines the pinned source from [#85](https://github.com/TrillionniumFoundation/HeptaTorch/pull/85)
(square, reciprocal, sign) and [#90](https://github.com/TrillionniumFoundation/HeptaTorch/pull/90)
(F32 and F64 copysign). The projector requires that the unary preparation
changes only the known unary definition and appends its helpers, and that
the copysign preparation has not changed that original unary definition.
The resulting HIA image preserves the five forward kernels; no default
runtime or acceptance change.

## Reproduce

Reproduce the exact source projections using the original PR instructions,
on pinned canonical main `e9444902df9878060ab389a70ee957c1b7f4a887`:

- #85 commit `c4d0aeaa4cd14c94e3e962f720b9096a7319bd82`, using exact #74,
  #79 and #81 operations and verifiers as pinned in its README.
- #90 commit `69d711dfb53dfab14ca944724ec5293e3938c0fb`, including its
  exact F32 predecessor source, single-dtype verifier and reproducible ELF.

The test inputs are graph-free, owned, same-shape where binary, contiguous,
zero-offset CPU tensors in ranks 0–4. The following uses the aggregate
`runtime.hepta` from each preparation and the exact verifiers from those
commits. Set `HEPTA_PYTORCH_INSTALL_REPORT` as in the original suites.

```sh
D=tools/current_runtime/opt_in_unary_copysign
R=tools/current_runtime
P=/absolute/path/to/fresh-mixed-scratch
set -- --unary-source "$EXACT_85_SOURCE" --copysign-source "$EXACT_90_SOURCE"
python3 -I -B "$D/prepare.py" "$@" --output "$P/source"
python3 -I -B "$R/build.py" --manifest "$P/source/manifest.json" \
  --source-root "$P/source" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/build-a"
python3 -I -B "$D/prepare.py" "$@" --output "$P/pinned" \
  --expected-build "$P/build-a/build.json"
python3 -I -B "$R/build.py" --manifest "$P/pinned/manifest.json" \
  --source-root "$P/pinned" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/build-b"
cmp "$P/build-a/heptatorch" "$P/build-b/heptatorch"
"$HEPTA_TORCH_PYTHON" -I -B "$D/qualify.py" --repository "$PWD" \
  --parent "$EXACT_MAIN_ELF" --f32-parent "$EXACT_84_ELF" \
  --candidate "$P/build-b/heptatorch" --output "$P/qualification" \
  --unary-dir "$EXACT_85_TOOLS" --copysign32-dir "$EXACT_90_F32_TOOLS" \
  --copysign64-dir "$EXACT_90_F64_TOOLS" --square-dir "$EXACT_74_TOOLS" \
  --reciprocal-dir "$EXACT_79_TOOLS" --sign-dir "$EXACT_81_TOOLS"
```

The verifiers retain their pinned original parent images and use the one
combined candidate ELF. A distinct live session feeds F32 copysign output
into reciprocal, square and sign while interleaving an F64 copysign.
Native public readback and physical Storage are checked against pinned
PyTorch, with fresh Storage/Version and readback after release of all
inputs and the shared intermediate. The source input and the original
verifier bytes are checked by SHA-256.

[Execution evidence](evidence.json) is scoped to these opt-in forward
operations. This does not qualify AD, arbitrary layout, other dtype,
broadcasting, mixed dtype, or full-schema/subsystem acceptance.
