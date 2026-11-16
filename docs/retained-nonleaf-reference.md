# Retained nonleaf gradients: pinned reference preparation

This change adds a reference-only observation plan for `Tensor.retain_grad()`.
The probe is frozen at SHA256
`7cdb53f7a95c48acd1b85d93d90c693d4caa97ada72a54a4021bb616d098fcba`.
Local preparation has run; real PyTorch execution and native compatibility are
pending. It does not add a native opcode or change a production runtime.

The original 24-case Optimizer observation only demonstrated that construction
accepts a retained nonleaf parameter. It did not execute its retained backward,
optimizer step, or lifetime. The new plan preserves that original evidence and
adds eight families with twelve variants:

| Family | Variants | Observations |
| --- | ---: | --- |
| Admission, leaf no-op, idempotence | 1 | Requires-grad rejection, retaining twice, nonleaf Optimizer admission |
| Gradient replacement and aliases | 1 | First materialization, repeated out-of-place replacement, clear, old Saved gradient |
| Strided incoming and assigned gradient | 1 | Logical words, actual retained layout and identity after accumulation |
| Empty retained gradient | 1 | Defined empty gradient versus None and repeated replacement |
| Retained Optimizer parameter | 3 | SGD, Adam and AdamW lazy state, zero/None skip and resume |
| Own Saved-version failure | 1 | Retained gradient effects before the node's failing backward |
| Own released-Saved failure | 1 | Repeated default backward and retained gradient effects |
| Weak lifetime and Optimizer ownership | 3 | Public drop, held gradient, privately owned parameter and final release |

Every snapshot records exact logical binary32 words, shape, strides, storage
offset/size, Version counter and opaque Tensor/Storage identity. None remains
distinct from defined zero. Python wrapper weak lifetime and Storage weak
lifetime are separate observations; neither is an intrusive reference-count
oracle. The observer returns primitive snapshots and installs no custom hooks.
Held seeds, gradients, and the guard using an old gradient are explicit inputs.

## Reproduce preparation

```sh
python3 -I -S -B tests/pytorch_retained_nonleaf_probe.py \
  --reference tests/pytorch_oracle.py \
  --lock compatibility/pytorch-oracle-lock.json \
  --repo . --output build/torch-retained-nonleaf-prepared.json --prepare-only
cmp build/torch-retained-nonleaf-prepared.json \
  validation/retained-nonleaf-reference/prepared.json
```

The result is `PREPARED_NOT_RUN`, with eight planned families, twelve variants,
zero executed cases and no Torch or native execution. The probe refuses changed
oracle or lock bytes. CI also verifies the frozen probe, preparation receipt and
independent source review before invoking it.

## Real reference execution and custody

The reference workflow installs the exact wheel from the existing oracle lock.
Its new execution command is the preparation command without `--prepare-only`,
using `--output build/torch-retained-nonleaf.json`. A real observation requires
all eleven original provenance checks, `OBSERVED`, eight rows with their fixed
variant counts, zero probe errors, and unchanged probe/oracle/lock bytes. No
native-derived expected values are used. Future Adam comparisons must retain
the existing declared numerical tolerance; identities, metadata, Versions and
None remain exact.

The four new workflow steps add two original-byte envelope reports:
`torch-retained-nonleaf-prepared.json` and `torch-retained-nonleaf.json`.
Against reference commit `9a20eadbe20a801c8b26727ae369e0ee0832e423`, declared
reference-job steps increase from 41 to 45 and required envelopes from 24 to 26.
The existing collector discovers both paths without modification and exports
all available JSON files with exact lengths and hashes, including failures.
The native workflow and all previous reference steps remain unchanged.

Pinned primary sources are recorded in
`validation/retained-nonleaf-reference/source-custody.json`:
[retain_grad](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/variable.cpp),
[engine execution](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/csrc/autograd/engine.cpp),
and [Optimizer admission](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/optim/optimizer.py).
Their source-level rationale does not substitute for the pending wheel
observations. The original independent source-only review is preserved at
`validation/retained-nonleaf-reference/independent-probe-review.json` without
claiming that it executed Torch.

This finite CPU binary32 first-order plan does not establish custom hooks,
higher-order differentiation, sparse gradients, other devices/dtypes, graph
rebasing, or full PyTorch compatibility. The full 2,590-schema scope remains.
