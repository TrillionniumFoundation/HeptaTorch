# Native optimizer version correction

This successor is built from the frozen candidate-03 source. The original package,
original native suites and first real PyTorch failure remain replayable.
Production arithmetic executes only in compiled HIA.

## Observed defect and correction

The [first actual strict PyTorch run](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43799969451)
ran all 28 cases, 6,233 frames and 2,590,207 value comparisons. Its two failures
were absolute parameter counters in nonzero-decay AdamW: 128 plus 32 updates
incremented the native counter once where PyTorch incremented twice. All numerical
comparisons completed. The [complete failure receipt](../../validation/hosted-owned-tensor-initial-failure.json)
retains all observations, including 859 separate optimizer-state counter differences.

The fixed [upstream single-Tensor implementation](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/optim/adam.py)
updates parameters with `mul_` for nonzero decoupled decay and then `addcdiv_`;
its second moment uses `mul_` followed by `addcmul_`. The source identity and
selected mode are recorded in `upstream-reference.json`.

`adam_versions.hepta` now performs the matching two parameter increments only
when decoupled decay is nonzero, one first-moment increment and two second-moment
increments. Positive and negative zero decay receive one parameter increment.
Zero learning rate and zero elements still count the actual logical mutating
operations. Numerical kernel instructions are unchanged: only the end of `adam`
passes its configuration into the added metadata function. The other 149
candidate-03 functions remain byte-identical; 125 of the 143 original canonical
function bodies remain byte-identical.

This does not reset an externally supplied momentum tensor's version to mimic a
newly constructed torch.optim state. Such a reset could invalidate saved-reference
reasoning. The explicit-state lifecycle difference remains open until the native
optimizer owns creation and lifetime of its state objects.

## Reproduction and qualification

```sh
make owned-tensors test-owned-tensors owned-tensor-versions test-owned-tensor-versions \
  HEPTA_COMPILER=/absolute/path/to/expanded-hia-native-compiler \
  HEPTA_SOURCE_ROOT=/absolute/path/to/pinned-hepta \
  HEPTA_OWNED_COMPILER=/absolute/path/to/owned-module-build/native/owned-module-compiler \
  HEPTA_OWNED_SESSION_BUILD=/absolute/path/to/owned-session-tools
```

The current service is `build/owned-tensor-versions/owned-tensor-session`.
The native compiler, linker, source, every module and final executable have fixed
SHA-256 identities in `profile.json`. Two independent native builds reproduced
the image. `qualify.py` retains its actual executed test sources. It changes only
two version expectations in the old 11-group suite and only the native image pin
in the independent review; it proves those text edits reverse exactly. Numerical
expectations and tolerances remain unchanged. Six typed-owner checks, the
independent 16-frame review, and ten zero/empty/decay version boundary cases also
execute against the corrected native image.

The strict 28-case real PyTorch oracle remains byte-for-byte unchanged and receives
the explicit new image pin:

```sh
python3 tests/owned_tensor_oracle/pytorch_owned_tensor_oracle.py \
  build/owned-tensor-versions/owned-tensor-session build/pytorch-owned-tensor.json \
  --session-reference tests/pytorch_session_oracle.py \
  --reference tests/pytorch_oracle.py \
  --lock compatibility/pytorch-oracle-lock.json --repo . \
  --expected-native-sha256 57292dfc87acf88e1f94df70bdd4579ace4dbdeecd288f2d60113f7762837575
```

The original fixed numerical expectations, all absolute parameter-counter
assertions, the 2,590-schema denominator and all 17 complete subsystem gates
remain required. CPU binary32/rank 0–4 limits and the missing independent
Storage/Version integration still apply. These gates do not qualify full
PyTorch optimizer objects, saved-data selection or general clone layout.
