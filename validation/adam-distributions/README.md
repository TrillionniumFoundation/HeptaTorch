# Combined Adam/distribution qualification receipts

`reports/qualification.json` records the actual portable 13-suite execution.
Twelve suites meet their scoped checks. `legacy_atomic_controls.json` remains
the real exit1/FAIL from the old atomic profile's unknown-opcode assumption.
The exact inverse test binding and the separate extended admission derivative
are in `reports/test-bindings.json`. `new_command_admission.json` executes all
three profile ELFs and proves the precise86/87 extension and unchanged88/89.

The two183-row reports are replays of the original actual PyTorch observation,
not new live Torch runs. They retain161+4+18 atomic and165+18 explicit outcomes.
The unchanged19-case old optimizer recipe retains18 exposed passes and1partial;
the separately frozen4-case Adam replay passes. The original case20 public
gradient Tensor object remains open. Full completion flags stay false/HOLD.

The archive contains every original request, response, stdout/stderr, generated
source, native object and image from initial and cold qualification, including
the first host-authoring failure and the original atomic negative-test failure.
Every archive member was decoded and compared with the original file by length,
SHA-256 and mode. The independent six-case composition review is a separately
identified report; it is not silently substituted for any frozen test here.

## Execute the portable gate

After rebuilding the two frozen predecessors and this combined profile:

```sh
python tools/adam_distributions/qualify.py \
  --build build/adam-distributions \
  --adam-build build/owned-adam \
  --atomic-build build/cpu-distributions \
  --distribution-build build/cpu-distributions-partial \
  --stack-build build/stack-tools \
  --atomic-tools tools/cpu_distributions \
  --partial-tools tools/cpu_distributions_partial \
  --adam-tools tools/owned_adam \
  --adam-review tests/adam_object_review \
  --tensor-protocol tools/tensor_objects/protocol.py \
  --optimizer-observations validation/optimizer-objects/torch-optimizer-objects.json \
  --output build/adam-distributions-qualified
```

The observation argument must name the exact original24-case file with SHA
`7ad0a2e718300a33c91a9212f14ada3c0c7997be7fc77f64bb557c2ecd858160`;
its repository location may differ. The executable source/image hashes and
all frozen script/reference identities are validated before execution.
