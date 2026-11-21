# Native F32 indexed forward correction

The current781 HIA source corrects no-gradient `gather`, `scatter_add` and
`index_add` while retaining the complete existing gradient paths. The new
`indexed_f32` group contributes331 fixed cases to the mandatory32-group,
15,145-case current gate. These finite cases do not close a complete PyTorch
declaration, the whole indexing subsystem or full autograd.

## Reproduced defects

The previous778 image clears AD parent and operation fields when an output does
not require gradients. Three forward paths still read those cleared fields to
recover their inputs and operation kind.

| Operation | Actual old behavior | Required behavior |
| --- | --- | --- |
| `gather([[1,2,3],[4,5,6]], 1, [[2,0],[1,2]])` without gradients | `[[3,1],[2,3]]` | `[[3,1],[5,6]]` |
| `scatter_add` with neither operand requiring gradients | Rejects valid requests with status1 | Computes the indexed accumulation |
| `index_add` with neither operand requiring gradients | Rejects valid requests with status1 | Computes the indexed accumulation |

The correction changes one call in `gather` and one call in `indexed_add`.
Three new HIA helpers accept the live input capabilities and operation kind
explicitly. If the output requires gradients, they call the unchanged original
forward helper. Otherwise, they derive geometry from the live inputs and
output, without writing temporary AD edges. The original `gather_source`,
`indexed_add_forward`, `indexed_add_location` and every backward body remain
byte-identical.

The complete781→778 inverse checks both call sites, all three new bodies,
776 unchanged predecessor functions, definition order, the whole parent source
hash and the exact forward round trip. The combined781→778→775 proof also
retains the signbit and typed-index-selection transition: six changed parent
bodies, six appended helpers and769 unchanged775 functions. Both nested proof
identities remain explicit in the [source review](../validation/current-runtime781/source-review.json).

## Actual scoped observations

| Required observation | Count |
| --- | ---: |
| Gather cases | 81 |
| Scatter-add cases | 125 |
| Index-add cases | 125 |
| Successful reference cases / actual Torch errors | 307 / 24 |
| Actual native sessions | 334 |
| Actual gradient comparisons | 193 |
| Request/response frames | 9,938 |
| Assertions | 37,094 |
| Complete Owner snapshots | 2,918 |
| Complete mapping snapshots | 30 |
| Protocol controls | 56 |
| Successful / failed same-frame transactions | 3 / 12 |
| Real allocation refusal followed by identical-request recovery | 3 |

The cases cover ranks0–4, signed axes, INT64 boundaries, duplicate indices,
scalar and empty cases, transposed/gapped/broadcast/channels-last layouts,
identical self/source tensors, source release and retained output lifetime.
Gather also checks raw NaN and signed-zero bits. Gradient cases use nonuniform
cotangents and compare complete actual gradients with the independent Torch
reference.

The original local runs and pure physical replays passed on the exact781
source and ELF. Their input identities and limitations are recorded in
[native-review.json](../validation/current-runtime781/native-review.json).
The local Torch installation lacks its original pip report and direct_url
metadata, so those runs remain explicitly exploratory. The default runner and
qualifier reject that missing provenance. CI must execute the new group with
all eleven original installation checks, together with every previous group,
and retain the original complete evidence for independent readback.

```sh
python3 -B tests/indexed_f32/run.py \
  --build-directory /absolute/current-build \
  --expected-build-sha256 ACTUAL_BUILD_JSON_SHA256 \
  --torch-python /absolute/pinned-env/bin/python \
  --output /new/indexed-f32-review
```

No old failed result is rewritten. Source inverses describe exactly which
bytes changed; runtime and reference executions provide the numerical evidence.
