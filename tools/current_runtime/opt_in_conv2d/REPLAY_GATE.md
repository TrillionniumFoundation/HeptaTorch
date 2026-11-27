# CI-ready replay of the fixed F32 conv2d qualification

This opt-in command reconstructs both the #75 comparison ELF and the #78
candidate from the canonical source. It verifies the original per-unit source
and object hashes, aggregate source, manifest, and ELF, then executes the exact
public-wire differential against pinned eager PyTorch. It does not reuse existing
ELFs or claim default-runtime or full-rewrite qualification.

From the repository root, provision the original pinned compiler and linker and
install the upstream-locked CPython 3.12 Linux x86-64 Torch wheel into a separate
environment. Keep the install report emitted by pip's `--report` (the original
build instructions describe this oracle). Run on an AVX2-capable Linux x86-64
worker:

```sh
python3 -B tools/current_runtime/opt_in_conv2d/replay_gate.py \
  --compiler "$HEPTA_STACK_BUILD/stack-module-compiler" \
  --linker "$HEPTA_STACK_BUILD/stack-linker-session" \
  --torch-python "$HEPTA_TORCH_PYTHON" \
  --wheel "$PINNED_TORCH_WHEEL" \
  --install "$PINNED_TORCH_INSTALL_REPORT" \
  --output "$RUNNER_TEMP/heptatorch-conv2d-replay"
```

Choose an output directory that does not exist, under an existing directory;
the workspace remains available on failure. In CI, rely on the exit code and
upload the output directory for diagnosis if needed. The gate writes
`gate.json` with `PASS_SCOPED_REPLAY` only after both fresh builds, all 32
case reports, both channel refusals and the one-session no-grad spatial chain
match the retained reference. The qualifier verifies the installed Torch
components against the supplied original wheel. Only the SHA-256 of the
machine-specific pip installation receipt may vary relative to the old report;
the new receipt must still name the pinned wheel, and the replay checks its
actual digest before declaring success. Native reports and resulting bytes
are kept separately from compiler subprocess stdout and stderr.

This is an entry point for a runner that already has the pinned inputs. It
does not add or modify a hosted workflow, fetch credentials, or change default
acceptance. Its measured domain remains the fixed bias-free finite dyadic F32
fixtures in the parent qualification.
