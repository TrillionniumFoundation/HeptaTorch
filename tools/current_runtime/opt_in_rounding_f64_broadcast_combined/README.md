# Integrated F32 rounding and F64 copy-sign broadcasting

This opt-in image combines the exact rounding aggregate in [#97](https://github.com/TrillionniumFoundation/HeptaTorch/pull/97)
with the single native broadcast body from [#96](https://github.com/TrillionniumFoundation/HeptaTorch/pull/96).
The projector proves the broadcast aggregate differs from its pinned #90
parent in only the named F64 copysign function and that #97 contains the
exact original function. No untouched function is replaced.

## Reproduce

The three aggregate inputs are the `runtime.hepta` outputs from the
source preparation in these exact commits:

- #97 `aa5f900afbd6ac36aaa7723c8079638ebf2f617c` (F32 rounding on #93);
- #90 `69d711dfb53dfab14ca944724ec5293e3938c0fb` (F64 copysign base);
- #96 `373f66db843bc2df5c9414840e74903a85cc35c4` (F64 broadcasting).

Use a compatible checkout with the canonical runtime tools, the pinned
compiler/linker and a Python environment with the locked Torch CPU wheel
and its original pip installation report at
`build/pytorch-install.json` below the repository root. The parent binaries
for the two pinned tests are #93 `5cd57293…` and #90 `f46436d6…`,
respectively, rebuilt with their original instructions.

```sh
R=tools/current_runtime
D=$R/opt_in_rounding_f64_broadcast_combined
P=/absolute/path/to/fresh-combined-replay
set -- --runtime-tools "$R" --rounding-source "$ROUNDING_SOURCE" \
  --copysign-base "$COPY_SIGN_BASE_SOURCE" --broadcast-source "$BROADCAST_SOURCE"
cold() {
  python3 -I -B "$R/build.py" --manifest "$1/manifest.json" --source-root "$1" \
    --compiler "$COMPILER" --linker "$LINKER" --output "$2"
}
python3 -I -B "$D/prepare.py" "$@" --output "$P/source-a"
cold "$P/source-a" "$P/build-a"
python3 -I -B "$D/prepare.py" "$@" --output "$P/source-b" \
  --expected-build "$P/build-a/build.json"
cold "$P/source-b" "$P/build-b"
cmp "$P/build-a/heptatorch" "$P/build-b/heptatorch"
for i in a b; do
  "$HEPTA_TORCH_PYTHON" -I -B "$D/qualify.py" --repository "$PWD" \
    --candidate "$P/build-$i/heptatorch" --output "$P/replay-$i" \
    --rounding-parent "$EXACT_93_ELF" --broadcast-parent "$EXACT_90_ELF" \
    --opt-in-rounding-f32 "$EXACT_97_VERIFIER_DIR" \
    --opt-in-copysign-f64 "$EXACT_90_VERIFIER_DIR" \
    --opt-in-copysign-f64-broadcast "$EXACT_96_VERIFIER_DIR"
done
cmp "$P/replay-a/qualification.json" "$P/replay-b/qualification.json"
```

The qualifier reads the exact verifier bytes and changes only each
expected candidate executable digest; the references and boundary checks
remain unmodified. The F64 broadcast suite supersedes the old intentional
refusal for compatible unequal shapes. One independent native process
also performs F64 broadcast → scalar-tensor copysign interleaved with
F32 floor → square and F32 same-shape copysign. It reads real owner bytes,
checks new Storage and Version handles, verifies inputs are unchanged,
and reads remaining outputs after input and intermediate release.

This candidate is a CPU no-gradient integration for the original scoped
layouts, not a default runtime replacement, F32 broadcasting, F64 rounding,
accelerator qualification, or a complete PyTorch rewrite.
