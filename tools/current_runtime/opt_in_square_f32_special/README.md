# Opt-in exceptional F32 square gradient words

This candidate extends #87's ordered reverse AD on the exact 777-function
parent source. Its new HIA helper changes only the contribution for square
node 51: a NaN from the input takes precedence over a NaN from the incoming
seed, and the selected NaN is quieted without dropping sign or payload.
For a zero multiplied by infinity (including infinity created by the
rounded doubling step), the x86 oracle's negative indefinite word
`ffc00000` is used. The finite ordered product and graph bookkeeping are
unchanged.

The pinned single-thread AVX2 CPU eager oracle and native candidate agree
on 1,147 forward and 1,147 gradient words across 265 dedicated fixtures:
a complete 16×16 input/seed cross-product at rank 0 and in packed arrays,
plus additional rank 1–4 layouts with small and tail sizes. The exact #87
ELF had 168 mismatching gradient words in that scalar cross-product and
168 in its first packed replay; the candidate has zero. The test also
replays #87's 32 finite fixtures, 46 backward rounds, 148 unchanged no-grad
frames, four subnormal/fan-out controls, input-version refusals and
unrelated graph regression. Every report field other than ELF identity,
including the wire frame hashes, matches #87's evidence. Both candidate
cold builds reproduced all 15 native object pins and the ELF.

The exceptional-word result is scoped to single-square backward on the
tested owned contiguous zero-offset CPU F32 inputs and incoming seeds,
under the pinned x86-64 AVX2 PyTorch installation. It is not a claim about
all NaN encodings, operands routed through general graph accumulation,
alias/views, higher-order AD, other devices/dtypes, or the default image.
No canonical accounting changes.

## Reproduce

Build the exact #77 split candidate and the exact #87 ordered candidate
using their recipes. With `SPLIT_ELF` and `ORDERED_ELF` set to those
executables, use the same pinned compiler/linker and CPU wheel. Paths
must be absolute where noted by the predecessor and output paths new.

```sh
R=tools/current_runtime
D="$R/opt_in_square_f32_special"
python3 -I -B "$D/prepare.py" --output "$P/source"
python3 -I -B "$R/build.py" --manifest "$P/source/manifest.json" \
  --source-root "$P/source" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/build-a"
python3 -I -B "$D/prepare.py" --output "$P/pinned" \
  --expected-build "$P/build-a/build.json"
python3 -I -B "$R/build.py" --manifest "$P/pinned/manifest.json" \
  --source-root "$P/pinned" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/build-b"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$HEPTA_TORCH_PYTHON" -I -B "$D/qualify.py" --repository "$PWD" \
  --split-parent "$SPLIT_ELF" --ordered-parent "$ORDERED_ELF" \
  --candidate "$P/build-b/heptatorch" --output "$P/qualification"
```
