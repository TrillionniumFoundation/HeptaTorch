# One opt-in F32 CPU image for square, reciprocal, and sign

Isolated results do not imply coexistence. This projection integrates the
exact graph-free HIA bodies from #74 (square, wire 3/16), #79 (reciprocal,
3/17), and #81 (sign, 3/18). It extends the single canonical typed-ReLU 775
dispatch before the unchanged fallback and binds all three input bodies by
SHA-256. It does not include the square backward prototype, arbitrary
layouts, other dtypes, new PyTorch acceptance credit, or any default
runtime change.

Two independent fifteen-object builds produced the same 778-function ELF.
All three original qualification suites executed against that ELF with only
their expected candidate SHA-256 rebound: 133 positive cases and 32,305
bit-exact F32 words against locked CPU PyTorch 2.15.0, 133 parent refusals,
three boundary refusals per operation, and the same 28-frame old-unary
regression in each suite. This is one ELF, not three separately built
candidates. A separate single native process interleaved all three opcodes
for each input, with 15 results and 225 exact words over 70 frames. It
independently decoded the physical owner and verified distinct Storage and
Version for each live output and unchanged input; every output remained
readable after releasing the shared input.

## Reproduce

Use the canonical repository build tools, exact main ELF, and the pinned
compiler/linker from the build guide. Set `HEPTA_PYTORCH_INSTALL_REPORT`
to the locked wheel's pip installation report. The scripts take source
and verifier bytes explicitly; the following fetches the three published
drafts and reads the immutable commits (not moving branch heads):

```sh
set -eu
D=tools/current_runtime/opt_in_three_unary_f32
R=tools/current_runtime
P=/absolute/path/to/fresh-three-unary-scratch
mkdir -p "$P"
git fetch origin refs/pull/74/head refs/pull/79/head refs/pull/81/head
for pair in \
  square:82184e4401eebe1d69ed8742485a9f603fd49146 \
  reciprocal:3919e25bc0d9c9094e56a6daed87d3ac1eff1565 \
  sign:394bc096114298e23b8cdf2c795bf5271b8b17a8
do
  op=${pair%%:*}; ref=${pair#*:}
  mkdir "$P/$op"
  git show "$ref:$R/opt_in_${op}_f32/$op.hepta" >"$P/$op/$op.hepta"
  git show "$ref:$R/opt_in_${op}_f32/qualify.py" >"$P/$op/qualify.py"
done
set -- --square-dir "$P/square" --reciprocal-dir "$P/reciprocal" --sign-dir "$P/sign"
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
"$HEPTA_TORCH_PYTHON" -I -B "$D/qualify.py" "$@" --repository "$PWD" \
  --parent "$EXACT_MAIN_ELF" --candidate "$P/build-b/heptatorch" \
  --output "$P/qualification"
```

[Execution evidence](evidence.json) records exact inputs and outputs. These
are three scoped forward routes, not full-schema or subsystem qualification.
