# Native opt-in F64 frac

This extends command 3 / subop 22 from [#109](https://github.com/TrillionniumFoundation/HeptaTorch/pull/109)
to ordinary F64 CPU tensors while keeping its F32 kernel. The binary64
body uses the existing native F64 bit truncation and a native exact
residual subtraction. F64 frac yields positive zero for integer
inputs including negative zero, preserves nonzero subnormals, returns
`fff8000000000000` for either infinity, and quiets signaling NaNs
without losing their sign or payload. The last rule is not the same as
F64 trunc, which deliberately preserves signaling NaN bits.

The verification uses the locked AVX2 CPU PyTorch for contiguous
zero-offset no-grad F64 tensors, rank 0–4, and checks physical as well
as wire bytes, fresh typed Storage/Version, and readback after release.
The scope controls distinguish a wrong-dtype operator refusal from
existing F64 constructor refusals for grad/layout/offset on both images.
Previously accepted F32 frac results are unchanged. The mixed test
feeds native copysign into frac and floor/ceil in one interleaved
F32/F64 process. Default runtime, acceptance, AD and non-CPU coverage
are unchanged.

## Reproduce

Follow #109 to obtain its exact aggregate source and executable and
the previous #107 executable. Use the same compiler, linker, locked
PyTorch binary, and install report. Use new scratch subdirectories:

```sh
R=tools/current_runtime
D=$R/opt_in_frac_f64
python3 -I -B "$D/prepare.py" --parent-source "$EXACT_109_SOURCE" --output "$P/source"
python3 -I -B "$R/build.py" --manifest "$P/source/manifest.json" \
  --source-root "$P/source" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/build-a"
python3 -I -B "$D/prepare.py" --parent-source "$EXACT_109_SOURCE" --output "$P/pinned" \
  --expected-build "$P/build-a/build.json"
python3 -I -B "$R/build.py" --manifest "$P/pinned/manifest.json" \
  --source-root "$P/pinned" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/build-b"
cmp "$P/build-a/heptatorch" "$P/build-b/heptatorch"
for BUILD in a b; do
  "$HEPTA_TORCH_PYTHON" -I -B "$D/qualify.py" --repository "$PWD" \
    --parent "$EXACT_109_ELF" --candidate "$P/build-$BUILD/heptatorch" \
    --output "$P/qualification-$BUILD"
done
cmp "$P/qualification-a/qualification.json" "$P/qualification-b/qualification.json"
"$HEPTA_TORCH_PYTHON" -I -B "$D/replay_f32.py" \
  --f32-verifier "$R/opt_in_frac_f32/qualify.py" --repository "$PWD" \
  --parent "$EXACT_107_ELF" --candidate "$P/build-b/heptatorch" --output "$P/f32-replay"
"$HEPTA_TORCH_PYTHON" -I -B "$D/replay_mixed.py" \
  --mixed-verifier "$R/opt_in_rounding_f64/mixed.py" --repository "$PWD" \
  --candidate "$P/build-b/heptatorch" --output "$P/mixed"
```

The F32 replay swaps only the former F64 refusal fixture for an I64 fixture.
The new mixed replay substitutes frac of the copysign output for the prior
rounded-output trunc operation. See [evidence.json](evidence.json) for
source, test, and native build identities.
