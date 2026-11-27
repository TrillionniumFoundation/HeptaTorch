# Opt-in native F32 average pooling

The exact typed-relu775 Linux x86-64 image refuses ordinary F32
average pooling with requires_grad=False: its shared path reaches
the graph-dependent point loop even when the public output has no AD
parents. This projection selects the HIA no-grad loop for average
pool **only**. It does not change existing gradient or max-pool
branches. The two forward helpers are reused byte-for-byte from the
prior scoped spatial-forward repair; this projection changes just the
one existing pool2d body and has an exact byte-level inverse.

The accompanying opt-in API also lowers evenly tiled
adaptive_avg_pool2d to this native operation. For positive output
extents dividing the input height and width, adaptive bin bounds
[floor(i*I/O), ceil((i+1)*I/O)) become adjacent windows of width
I/O, so kernel and stride coincide with zero padding. The client
calculates geometry only, not tensor values. It dispatches native
forward, optionally native seeded backward, and reads the real
result on the public wire. It checks metadata, distinct owners,
unchanged input/seed, output survival after release, and owner
invariants.

The API is limited to **finite, positive-size, owned contiguous F32
CPU NCHW**, up to 4096 values and 64 per spatial dimension, divisible
bins, and optional first-order F32 gradient. It refuses irregular
or expanding output geometry before launch. This is not a
torch API drop-in, arbitrary adaptive pooling, a default change,
a complete schema/derivative claim, hosted CI, or full acceptance.

## Reproduce

From the repository root, with the same fixed HIA compiler/linker
as in the repository build instructions:

    p=tools/current_runtime/opt_in_adaptive_avg_pool2d
    w=$(mktemp -d)
    python3 -B "$p/project_pool.py" --parent tools/current_runtime/source/runtime.hepta --output "$w/runtime.hepta"
    python3 -B tools/current_runtime/partition.py --runtime "$w/runtime.hepta" --name narrow-avg-pool-no-grad777 --prefix avg_pool777 --output "$w/source"
    python3 -B tools/current_runtime/build.py --manifest "$w/source/manifest.json" --source-root "$w/source" --compiler "$HEPTA_STACK_BUILD/stack-module-compiler" --linker "$HEPTA_STACK_BUILD/stack-linker-session" --output "$w/build"
    (cd "$p" && python3 -B -m unittest -v)
    "$HEPTA_TORCH_PYTHON" -s -E -B "$p/qualify.py" --binary "$w/build/heptatorch" --parent "$PARENT_775_ELF" --wheel "$PINNED_TORCH_WHEEL" --install "$PINNED_TORCH_INSTALL_REPORT"

Two independent cold builds on the dot matched
ecd87c5eba1660def55e9b567a5c0c6ebd020a34297beea04fc5f63eb97f7467.
The qualification executed nine adaptive geometries both with and
without backward, plus two ordinary average-pool cases with overlap
and padding. The exact parent refused all 11 no-grad requests
without changing input; all nine existing backward replays were
unchanged. Actual Torch eager 2.15.0+cpu on AVX2 and native returned
identical F32 bits on the measured fixtures. See
[qualification.json](qualification.json) and
[build-receipt.json](build-receipt.json).
