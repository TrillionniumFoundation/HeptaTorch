# Opt-in native CPU integral whole-tensor product

Implements the CPU Bool and I64 slice of `aten::prod.default` with implicit
output dtype. The existing Tensor wire accepts it as unary command 3, operation
22. The kernel reads all logical input elements with the existing native
strided reader, multiplies with wrapping 64-bit arithmetic, and creates a new
rank-zero I64 Tensor, Storage and Version. Empty logical tensors reduce to
one. Zero-stride and overlapping views multiply each logical occurrence.

Other operation numbers follow their original routes. For 3/22, unsupported
input dtypes also keep their original failure routes. The projector works only
from the exact default typed-ReLU775 source; the canonical image and
acceptance ledger are not changed. Axis selection, keepdim, out variants,
explicit dtype, other dtypes, autograd and accelerators are out of scope.

## Reproduce with the locked compiler/linker and PyTorch install

Use canonical main `e9444902df9878060ab389a70ee957c1b7f4a887` and its exact
parent ELF (SHA-256 `ef82340209edc4dfb8d7ed0ecbc65e144fe26078cb29a0ba327798533476e86e`).
The pinned PyTorch 2.15.0+cpu environment must retain its real pip installation
report and PEP 610 wheel provenance. From the repository root:

```sh
set -eu
R=tools/current_runtime
D=$R/opt_in_integral_prod
P=/absolute/path/to/fresh-product-scratch
python3 -I -B "$D/prepare.py" --output "$P/source-a"
python3 -I -B "$R/build.py" --manifest "$P/source-a/manifest.json" \
  --source-root "$P/source-a" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/build-a"
python3 -I -B "$D/prepare.py" --output "$P/source-b" --expected-build "$P/build-a/build.json"
python3 -I -B "$R/build.py" --manifest "$P/source-b/manifest.json" \
  --source-root "$P/source-b" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/build-b"
cmp "$P/build-a/heptatorch" "$P/build-b/heptatorch"
for round in a b; do
  "$HEPTA_TORCH_PYTHON" -I -B "$D/qualify.py" --repository "$PWD" \
     --parent "$EXACT_MAIN_ELF" --candidate "$P/build-$round/heptatorch" \
     --install-report "$HEPTA_PYTORCH_INSTALL_REPORT" \
     --output "$P/qualification-$round"
done
cmp "$P/qualification-a/qualification.json" "$P/qualification-b/qualification.json"
```

Two independent builds produced identical 15 object digests and native ELF.
Both strict oracle runs match all 119 result words (41 I64 and 78 Bool inputs,
25,142 logical input words), including 27 view cases, empty ranks, an offset
scalar, signed overflow and zero behavior. Each case checks input metadata
and underlying Storage bytes before and after execution, fresh physical
result Storage and Version, and readback after input/base release. The parent
refuses all 119 new inputs. Four negative controls per image preserve the
owner byte-for-byte; sampled previous unary and sum response frames are
unchanged. See [build evidence](evidence.json) and the
[case and transport receipt](qualification.json).
