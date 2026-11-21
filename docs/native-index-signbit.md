# Native signbit and index_select

The extension introduced in781 and retained through797 in candidate800 adds these two bounded CPU contracts to the one HIA source and static image. HIA performs classification, iteration, allocation, raw element copies, ownership and rollback. Python transports commands and observes results; the independent Torch process produces expected bytes.

The [source review](https://github.com/TrillionniumFoundation/HeptaTorch/blob/683adf048bf083d4f1116a5ef711d18a36f39e24/validation/current-runtime781/source-review.json) fixes the complete source and image. The [qualification plan](../validation/current-runtime800/qualification-plan.json) requires all35 groups/16,365 cases plus the separate [public F64 view gate](native-public-f64-views.md). The recorded predecessor targeted execution and physical replay passed, but the local Torch installation lacks the complete original provenance; those results remain exploratory and do not qualify800; its strict exact-tree CI is still required.

## Signbit

Public opcode126 takes the same unary-predicate packet shape as the existing five classification operations. It admits the seven existing dtype codes: F32, Bool, I64, internal wrapped F64 scalar, public F64, F16 and BF16. It produces an independently owned Bool result with the observed shape/layout. Opcode125 remains reserved and rejected; all older dispatch ranges are unchanged.

The native value function reads sign bits for floating values, including negative zero and signed NaNs, and uses signed comparison for I64. Bool inputs produce false. It reuses the existing native predicate allocation and iteration path. The 97 fixed reference cases include all65,536 encodings for each Half type, special floating values, scalar and empty inputs, layouts and lifetime checks. Six successful same-frame FE transactions and six invalid-suffix transactions check complete rollback; two allocation controls require an actual native refusal followed by the identical request succeeding after resource restoration.

| Required observation | Count |
| --- | ---: |
| Fixed actual Torch cases | 97 |
| Actual native processes | 106 |
| Request/response frames | 2,347 |
| Assertions | 8,843 |
| Owner snapshots | 774 |
| Mapping snapshots | 12 |
| Protocol controls | 13 |
| Successful / failed FE transactions | 6 / 6 |
| Real OOM and recovery controls | 2 |

## Index selection

Opcode30 now handles no-gradient F32, Bool, I64, F64, F16 and BF16 tensors, rank0–4, admitted nonnegative strides and storage offsets. The index payload is signed I64. Negative dimensions normalize against the input rank; negative or out-of-range index values remain errors. Scalar inputs accept exactly one index equal to zero and dimension0 or -1, matching the observed reference behavior. Empty results retain their validated shape and ownership.

The output has independent contiguous storage. Native iteration maps each output element to the selected source coordinate, honors original source strides and offset, and copies the original1/2/4/8-byte element. This preserves wide integers, signed zero and NaN payloads without a lossy F32 conversion. Source tensor bytes, Version and Generator state remain unchanged.

### Corrected F32 behavior

The original no-gradient F32 path selected the first input row repeatedly. For input `[[1,2,3],[4,5,6]]`, dimension0 and indices `[1,0,1]`, it returned three copies of `[1,2,3]`. The corrected raw-copy path returns `[[4,5,6],[1,2,3],[4,5,6]]`. The original failed observation is retained independently; it is not relabeled as a passing baseline.

The complete previous F32 function remains in the actual image as `typed_index_select_previous`. Non-scalar F32 inputs requiring gradients continue through that function. Scalar F32 inputs validate the scalar selection and clone through the existing differentiable path. Five actual reference cases check gradients, including repeated indices, negative axes, empty output and scalar input. Non-F32 AD remains outside this bounded contract.

| Required observation | Count |
| --- | ---: |
| Fixed actual Torch cases | 866 |
| Actual native processes | 869 |
| Request/response frames | 17,153 |
| Assertions | 64,271 |
| Owner snapshots | 5,759 |
| Mapping snapshots | 26 |
| Protocol controls | 78 |
| Actual F32 autograd cases | 5 |
| Real OOM and recovery controls | 2 |
| Original Bool packet numerical successors | 1 |

The Bool successor preserves the original setup allocation order and all three original command fields: logical-not, generator uniform, and index-select. It checks the actual successful prefix and result, then requires an invalid suffix to restore complete tensor, Generator and mapping state. The historical old523 observer is retained unchanged.

## Source and evidence admission

The original signbit/index-select extension changes four775 bodies and adds three helpers. The current781 image also contains the [F32 indexed-forward corrections](native-indexed-forward.md), which change two further call sites and add three helpers. The complete781→778→775 inverse retains both intermediate identities and proves the aggregate six changed functions, six appended functions and769 unchanged parent bodies. It reconstructs the exact775 source and forward-round-trips to the exact781 bytes. All original backward bodies and the retained F32 index-select implementation remain byte-identical. Three existing observer source-admission functions gain this full inverse before the unchanged predecessor chain; every numerical comparison remains unchanged.

Each new strict runner binds the current complete cold build, current source, actual native image, actual Torch reference, all11 original installation provenance checks, exact input bytes/modes, native process receipts and both output EOFs. Its pure qualifier replays every original request/response and Owner snapshot without executing native code or Torch again. The coordinator and sealer reject exploratory local reports.

Standalone strict commands use an actual cold build and its build.json SHA-256:

```sh
python3 -B tests/signbit/run.py \
  --build-directory /absolute/current-build \
  --expected-build-sha256 ACTUAL_BUILD_JSON_SHA256 \
  --torch-python /absolute/pinned-env/bin/python \
  --output /new/signbit-review

python3 -B tests/typed_index_select/run.py \
  --build-directory /absolute/current-build \
  --expected-build-sha256 ACTUAL_BUILD_JSON_SHA256 \
  --torch-python /absolute/pinned-env/bin/python \
  --output /new/index-select-review
```

These finite contracts do not qualify complete PyTorch declarations, arbitrary ranks, devices, all layouts, full typed AD or the complete public frontend. All17 complete system gates and the original declaration denominators remain unchanged.

The current797 successor prepends its complete forward source inverse and reexecutes this group. Its fresh reference also binds the complete official wheel,12,567 original installed files, actual mapped computation libraries and before/after process custody. Historical781 observations retain their own identities; see the [current audit](current-audit.md) and [native forward contract](native-forward.md).
