# Ordered first-order F32 square

This independent opt-in projection extends the exact #77 square prototype
at `a8f7be7479702a6b0a7b98e8c1e347b3ee6ee0fd`. It does not modify the default
runtime or award canonical acceptance credit. The public candidate operation
remains `3/16`. A distinct internal node kind records one differentiable
parent and saves its source version. HIA computes the contribution as
`mulf(g, mulf(2, x))`, matching the operation order of eager CPU
`PowBackward` for square. First-contribution storage preserves negative
zero before later additions; the result is treated as a new edge
contribution rather than aliasing the incoming seed.

## Evidence

The pinned CPU PyTorch run and the native candidate agree bit-for-bit on
4,943 forward and 4,943 accumulated gradient words over 32 finite
input/seed fixtures and 46 seeded rounds. These cover ranks 0–4, empty
tensors, finite random bit patterns, signed zero, subnormals and an
overflowing intermediate. At `x = ±2^-74` with seed `2^-76`, the exact
old ELF yields signed zero. The candidate and eager CPU both produce the
correct signed minimum subnormal for one square or twice that subnormal
for two squares feeding one addition. Both native versions refuse
backward with status 6 after a real in-place input version bump, leaving
the owner unchanged on refusal. Eager raises the corresponding saved
version error. A separate multiply/add/ReLU backward replay is byte
identical to the predecessor. All 148 predecessor no-grad square response
frames remain byte identical across 37 cases, including special inputs.

Evidence is limited to the observed first-order CPU F32 fixtures on
owned contiguous zero-offset inputs. Nonfinite gradient payload rules,
arbitrary alias/view or in-place behavior, general graph composition,
higher-order AD, other dtypes and devices, and default admission are
outside scope. The geometric runtime admission itself does not enforce
a finite input/seed domain.

## Reproduce

Use the exact #77 source parent, pinned HIA compiler/linker and locked
CPU wheel with its repository install report. From this draft checkout,
set an absolute fresh scratch directory `P`. Keep each output path new.

```sh
R=tools/current_runtime
D="$R/opt_in_square_f32_ordered"
python3 -I -B "$R/opt_in_square_f32_ad/prepare.py" --output "$P/parent-src"
python3 -I -B "$R/build.py" --manifest "$P/parent-src/manifest.json" \
  --source-root "$P/parent-src" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/parent"
python3 -I -B "$D/prepare.py" --output "$P/native-src"
python3 -I -B "$R/build.py" --manifest "$P/native-src/manifest.json" \
  --source-root "$P/native-src" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/build-a"
python3 -I -B "$D/prepare.py" --output "$P/pinned-src" \
  --expected-build "$P/build-a/build.json"
python3 -I -B "$R/build.py" --manifest "$P/pinned-src/manifest.json" \
  --source-root "$P/pinned-src" --compiler "$COMPILER" --linker "$LINKER" \
  --output "$P/build-b"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$HEPTA_TORCH_PYTHON" -I -B "$D/qualify.py" --repository "$PWD" \
  --parent "$P/parent/heptatorch" --candidate "$P/build-b/heptatorch" \
  --output "$P/qualification"
```

Two cold candidate builds reproduced all 15 object pins and the native
ELF. See [evidence.json](evidence.json) for the pins, scoped replay
and wire frame hashes.
