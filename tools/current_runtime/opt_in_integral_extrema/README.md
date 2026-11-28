# Opt-in native CPU integral whole-tensor extrema

This extends [#100](https://github.com/TrillionniumFoundation/HeptaTorch/pull/100)
at exact head `6c54b9f6eaa3a1353c6546b0f277480e0175fc10` with two native
HIA unary routes: wire 3/23 for the CPU Bool/I64 slice of
`aten::min.default` and 3/24 for the same slice of `aten::max.default`.

Both reduce all logical elements into a fresh scalar of the input dtype.
They respect existing valid public strided descriptors and refuse empty
logical tensors before allocation. I64 comparison flips each operand's
top bit before an unsigned comparison, preserving signed ordering even
for INT64_MIN and INT64_MAX. The same helper reduces Bool using its native
0/1 ordering.

This is values-only whole-tensor min/max; index-producing, dimension,
keepdim, out, other dtypes, accelerator and differentiable variants are
outside the slice. No default runtime or acceptance ledger changes.

## Reproduce

From this checkout, first follow the preceding
[whole-tensor product directions](../opt_in_integral_prod/README.md)
to get the exact #100 aggregate source and its native ELF from two cold
builds on canonical main. Set `PROD_SOURCE` to the generated runtime.hepta
and `PROD_ELF` to that exact ELF. Keep the same stack-budget compiler/linker
and locked PyTorch CPU install/report.

```sh
set -eu
R=tools/current_runtime
D=$R/opt_in_integral_extrema
P=/absolute/path/to/fresh-extrema-scratch
python3 -I -B "$D/prepare.py" --parent-source "$PROD_SOURCE" --output "$P/source-a"
python3 -I -B "$R/build.py" --manifest "$P/source-a/manifest.json" \
  --source-root "$P/source-a" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/build-a"
python3 -I -B "$D/prepare.py" --parent-source "$PROD_SOURCE" --output "$P/source-b" \
  --expected-build "$P/build-a/build.json"
python3 -I -B "$R/build.py" --manifest "$P/source-b/manifest.json" \
  --source-root "$P/source-b" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/build-b"
cmp "$P/build-a/heptatorch" "$P/build-b/heptatorch"
for round in a b; do
  "$HEPTA_TORCH_PYTHON" -I -B "$D/qualify.py" --repository "$PWD" \
     --canonical "$EXACT_MAIN_ELF" --parent "$PROD_ELF" \
     --candidate "$P/build-$round/heptatorch" \
     --install-report "$HEPTA_PYTORCH_INSTALL_REPORT" --output "$P/qualification-$round"
done
cmp "$P/qualification-a/qualification.json" "$P/qualification-b/qualification.json"
```

The verifier checks bit-identical result words on real PyTorch, pristine
input storage and metadata, and independent physical Storage/Version per
live result. It also feeds the new native max result to the existing native
product route while prior handles are still live. Outputs survive release of
the input and any base view. The complete #100 suite is replayed with only
candidate identity rebound, and every original report field and request/
response frame hash reproduces. See [build evidence](evidence.json) and
[case and transport receipt](qualification.json).
