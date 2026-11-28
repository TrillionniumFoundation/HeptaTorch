# Opt-in native CPU last-axis integral product

Stacks on exact [#105](https://github.com/TrillionniumFoundation/HeptaTorch/pull/105)
head `398decf3f5b9d4aa21ac0db3d803312b5b7c6d7f`. Wire command 3, operation 25
implements CPU Bool/I64 `aten::prod.dim_int` for the last logical axis
(`dim=-1`), implicit I64 output dtype and `keepdim=False`. Scalars reduce
to a scalar. The native kernel accumulates each logical row with wrapping
64-bit multiplication, independent of the source's physical strides. Empty
rows emit one; an empty output has zero values. Every result has fresh
physical Tensor, Storage and Version. Other axes, dtype arguments, out
forms, keepdim, other dtypes, devices and default dispatch are unchanged.

## Reproduce

Follow the [product setup](../opt_in_integral_prod/README.md) and
[extrema setup](../opt_in_integral_extrema/README.md) on this checkout to
recover the exact parent sources and ELFs, plus the canonical ELF and the
pinned real PyTorch install. Then:

```sh
set -eu
R=tools/current_runtime
D=$R/opt_in_integral_prod_last
P=/absolute/path/to/fresh-last-axis-prod-scratch
python3 -I -B "$D/prepare.py" --parent-source "$EXTREMA_SOURCE" --output "$P/source-a"
python3 -I -B "$R/build.py" --manifest "$P/source-a/manifest.json" \
  --source-root "$P/source-a" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/build-a"
python3 -I -B "$D/prepare.py" --parent-source "$EXTREMA_SOURCE" --output "$P/source-b" \
  --expected-build "$P/build-a/build.json"
python3 -I -B "$R/build.py" --manifest "$P/source-b/manifest.json" \
  --source-root "$P/source-b" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/build-b"
cmp "$P/build-a/heptatorch" "$P/build-b/heptatorch"
for round in a b; do
  "$HEPTA_TORCH_PYTHON" -I -B "$D/qualify.py" --repository "$PWD" \
     --canonical "$EXACT_MAIN_ELF" --prod-elf "$PROD_ELF" --parent "$EXTREMA_ELF" \
     --candidate "$P/build-$round/heptatorch" \
     --install-report "$HEPTA_PYTORCH_INSTALL_REPORT" --output "$P/qualification-$round"
done
cmp "$P/qualification-a/qualification.json" "$P/qualification-b/qualification.json"
```

The verification also reads actual physical output Storage, reuses axis
results in whole-tensor product, checks release-safe selection, and replays
the exact previous extrema and whole-tensor product matrices on the new
native ELF. See [build evidence](evidence.json) and the
[case and transport receipt](qualification.json).
