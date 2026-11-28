# Cold replay of the combined unary/copysign candidate

This command replaces the manual predecessor projections and prebuilt ELF inputs
in the original reproduction recipe. It builds the canonical and F32-copysign
parent images, builds the combined candidate twice from empty directories, and
executes both full real-PyTorch qualification sets. It requires the same pinned
HIA stack compiler/linker and locked PyTorch install as the original recipe.

From the successor checkout (which stacks on #93):

```sh
git fetch --no-tags origin refs/pull/74/head refs/pull/79/head \
  refs/pull/81/head refs/pull/84/head refs/pull/85/head refs/pull/90/head
D=tools/current_runtime/opt_in_unary_copysign
python3 -I -B "$D/replay.py" --repository "$PWD" \
  --compiler "$COMPILER" --linker "$LINKER" \
  --torch-python "$HEPTA_TORCH_PYTHON" \
  --install-report "$HEPTA_PYTORCH_INSTALL_REPORT" \
  --output "$REPLAY_OUTPUT"
```

The PR refs only make the objects available. The runner reads the immutable
commit and path in [replay-inputs.json](replay-inputs.json) and checks every
file's SHA-256 before executing a projector or verifier. For offline replay,
pass `--inputs /path/to/exact/predecessors`; the expected relative destinations
are the `file` fields in the lock. No moving refs, previous object cache, or
prebuilt candidate or parent ELF are accepted in either mode.

`gate.json` is written only after both cold images and all 15 object pins
agree, the two exact parent ELF digests match, and both independent verifier
reports match the byte digest in the original #93 evidence. Per-step stdout,
stderr, build receipts and differential frames are retained under the fresh
output directory. A failed child leaves diagnostics but no passing gate.

The scoped matrix is 1,173 focused cases / 51,857 words plus separate mixed
sessions. No default dispatch, AD, broadcasting, dtype promotion, accelerator,
or whole-declaration coverage claim is made.
