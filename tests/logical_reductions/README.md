# Native logical reductions

Commands **118 (`any`)** and **119 (`all`)** reduce CPU Bool, Int64 and
binary32 tensors. They also accept the existing rank-zero wrapped binary64
scalar object. Command **115** now accepts these source types when its explicit
result dtype is Bool. All truth conversion, reduction, allocation and Tensor
publication execute in HIA.

The command words are:

| Word | Meaning |
| --- | --- |
| 0 | 118 for `any`, 119 for `all`, or 115 for explicit Bool `sum` |
| 1 | Live public source Tensor handle |
| 2 | Number of dimensions, or unsigned `-1` for all dimensions |
| 3 | Dimension-payload byte offset within the request; zero for absent/empty dimensions |
| 4 | `keepdim`, zero or one |
| 5 | Result dtype, exactly one (Bool) for commands 118/119 |
| 6, 7 | Reserved, exactly zero |

The dimension payload must lie within the declared request payload and contain
all requested 64-bit axis words. Nonaligned byte offsets are accepted by the
existing safe `getr` reader; these commands impose no additional alignment rule.

An explicit empty tuple means **no reduction** for `any` and `all`. For `sum`,
it retains PyTorch's different meaning of **all dimensions**. Negative axes
normalize before duplicate detection; a scalar accepts axis 0 or -1 and
remains rank zero. Empty reductions use False for `any` and True for `all`.
Signed floating zero is False; NaN, infinity and nonzero subnormal values are
True. For `any` and `all`, a floating input may require gradients; each result is a
detached Bool Tensor with independent Storage and Version objects. PyTorch
rejects `sum(dtype=bool)` when the floating input requires gradients; command
115 preserves that explicit gradient error after validating and normalizing the
axes. Duplicate or out-of-range axes take precedence over the gradient error.

These operations retain the current CPU resource limits (rank at most four,
262,144 logical elements and the existing dimension/stride checks). The
wrapped binary64 input remains a scalar-only representation. This increment
does not qualify unavailable dtypes, device backends, `out=` semantics or the
entire PyTorch schema.

## Independent reference

`cases.py` contains literal inputs and operation arguments, never expected
answers. The 147 cases are 49 recipes evaluated independently for `any`, `all`
and explicit Bool `sum`. They cover signed zero, subnormals, NaNs, infinities,
full-width signed integers, positive/negative/empty/duplicate/out-of-range axes,
scalar and empty tensors, rank four, transposes, offsets, holes, overlapping and
zero strides, and floating inputs with gradients. Three additional gradient
recipes combine duplicate or out-of-range axes with each operation to verify
error ordering; the original 138 actual observations remain unchanged.

Run `reference.py` with the exact repository-pinned Torch CPU environment:

```sh
/path/to/pinned/python tests/logical_reductions/reference.py \
  --repository "$PWD" --output /new/reference-directory
```

The original eleven wheel/build/install provenance checks must pass. The
reference runs no HIA program. Binary64 inputs use the genuine saved scalar of
an actual Torch scalar multiplication and retain its exact eight-byte value.

## Native execution

```sh
python3 tests/logical_reductions/native.py \
  --repository "$PWD" --build-directory build/current-runtime \
  --reference /reference-directory/report.json --output /new/native-directory
```

The runner binds the actual source, native executable and complete cold-build
report before execution. It compares output bytes, shape, strides, dtype,
Storage size, offsets, version, gradient and leaf/view state with the actual
reference. It also checks all 84 source descriptor words remain unchanged,
fresh Tensor/Storage/Version identities, absence of gradient/Saved/input edges,
result survival after input release, and the complete 5,056-byte RNG state.

Eighteen additional invalid-packet, missing-payload, stale-handle and failed
second-command controls require complete owner rollback. Each new operation
also runs a real allocation-failure test: a finite address-space limit allows
same-capacity transactions; a Bool-output allocation that needs owner growth
fails, the full old owner remains unchanged, and the identical
request succeeds after restoring the original limit. Actual process exit and
both output EOFs are required. The original request/response streams and each
case report remain in the chosen output directory.

## Standard-Python current-build qualification

The frozen actual reference is stored in
`validation/logical-reductions-reference/`; its report is restored without
rewriting any of its bytes. The CI/local entry is:

```sh
python3 -I -S -B tests/logical_reductions/qualify.py \
  --build-directory build/current-runtime --output /new/qualification
```

This wrapper admits the complete current build and source manifest before
starting the native reader, captures complete runner streams, verifies all
147 cases, 18 packet controls, two allocation/recovery controls, and all 150
native process exits/EOFs, then rechecks every bound input. It needs only
standard Python, the current cold build and Linux own-child observation.
