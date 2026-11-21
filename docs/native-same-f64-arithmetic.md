# Native same public F64 arithmetic

The current HIA candidate routes public F64/F64 Tensor pairs through the native
addition, subtraction, multiplication and true-division kernels. This is command
2, selectors 1–4, within the existing no-grad CPU profile: rank 0–4, admitted
construction and reachable layouts, and the existing resource limits. Addition
and subtraction use alpha=1. The change does not implement F64 AD, out/inplace,
rounding modes, other unary operators, additional architectures or a public
Python/C++ frontend.

The actual reference is the fixed Torch CPU wheel, with AVX2 selected before
import and one Torch thread. Numerical comparisons include every result bit,
including NaN payloads and signed zero. This finite AVX2 result is separate from
complete dtype, operator and subsystem qualification.

## Native change

Only two of the 773 existing HIA function bodies change:

| Function | Change |
| --- | --- |
| `public_f64_binary` | Admit same public F64 pairs for selectors 1–4 through the existing native arithmetic routes. Keep unsupported/private dtype and operation fallback behavior. |
| `half_mul_priority` | For a single-element F64 result with unequal input ranks, select the final reordered axis before reading the two broadcast strides. Preserve the existing branches for other shapes and dtypes. |

The other 771 function bodies and all definition positions are byte-identical to
the mixed-F64 predecessor. Arithmetic value kernels, promotion helpers, AD guards,
allocation, ownership and protocol bodies remain unchanged. The complete source
inverse recovers both the mixed-F64 predecessor and the first failed same-F64
candidate. Source identity complements actual execution; it grants no execution
qualification.

### Why singleton dimension ordering matters

The fixed upstream `TensorIterator` retains a size-one axis's actual stride when
the output axis also has size one. Missing leading axes have stride zero. For
unequal input ranks, the ordinary dimension ordering path starts with reversed
axes and compares each operand's nonzero strides. Coalescing all size-one axes
leaves the stride from the final reordered axis.

The HIA runtime already implements that ordering in
`integer_pointwise_axis_order` and `integer_pointwise_compare_axes`. The successor
reuses those unchanged helpers and `dist_probability_stride`. The additional
branch introduces no allocation or new owner. Unequal ranks and one output
element imply a positive output rank; the existing rank-four limit bounds the
packed permutation shift to 24 bits.

The guard tests the **result** dtype after promotion. It can therefore also
affect an admitted Half/F64 multiplication whose promoted result is F64. The
source proof's `Half_routes_unchanged` field is scoped to the unchanged Half
result branches (dtype 5/6); it does not assert that every request containing a
Half operand follows the same priority path. The retained Half multiplication
and other original groups must execute on this image as part of the full gate.

This follows the fixed [upstream TensorIterator implementation](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/TensorIterator.cpp),
specifically `compute_strides`, `reorder_dimensions`, `coalesce_dimensions` and
the same-shape fast-setup rule. The [independent source review](../validation/current-runtime-same-f64/source-review.json)
binds the exact native source and unchanged helpers.

## Retained failures and expanded cases

The initial 3,288-case execution failed case 2224: multiplying a scalar left NaN
`0x7ff800123456789a` by a rank-one right NaN `0xfff00123456789ab` selected the wrong
payload. Its original reference and failure remain unchanged.

The second candidate passed all original 3,288 cases. An additional 1,800 cases
then exposed nine failures in alternating-stride singleton layouts: indices
4241, 4259, 4277, 4331, 4349, 4367, 4439, 4457 and 4547. A fixed leading-axis
choice was insufficient; the final reordered axis is required. These failures
also remain failures in their original reports.

The final case inventory retains the entire original prefix and appends all
25 ordered rank pairs from zero through four:

| Case set | Cases |
| --- | ---: |
| Original same-F64 recipes, unchanged and in their original order | 3,288 |
| Singleton rank pairs with NaNs | 600 |
| Singleton rank pairs with signed zero | 400 |
| Singleton rank pairs with finite values | 300 |
| Singleton rank pairs with zero, unit, gapped and alternating strides | 500 |
| **Total** | **5,088** |

All original numerical, error, alias, owner, input, RNG and process predicates
remain mandatory. The suite includes 88 protocol controls and four real
allocation refusals with recovery of the identical request. The independent
reader replays the retained physical request/response streams and complete owner
observations without executing Torch or the native image again.

## Current identities and required integration

| Entity | SHA-256 |
| --- | --- |
| Complete HIA source | `846e592beaa7847383cee668bc33d25cd81e79052bd4612252e2f72a83d96d05` |
| Static native image | `5cba6f11269f71efb319aa7368009055bb2e9b9571329daeae9bbc453287c3ad` |
| Ordered 15-unit manifest | `62795753a0e782fe3f75b9086653d41aba743fda356673398c12432d576fac15` |

The complete current gate is **28 groups / 13,075 cases**. It retains the preceding
27 groups / 7,987 cases, including the original F64 48-case entry and its shared
2,541-case mixed arithmetic reference. The original mixed comparison code is
recovered by an exact observer inverse; only source admission and the now-false
"legacy same-F64 route unchanged" marker change. No earlier numerical success is
transferred to this image.

The [qualification plan](../validation/current-runtime775/qualification-plan.json)
and [actual execution review](../validation/current-runtime-same-f64/native-review.json)
record the current scoped results and the requirement for complete new-tree CI,
separately executed Generator images, all other head checks and complete original
artifact readback. The complete PyTorch target remains **HOLD**: no complete
native declaration, derivative declaration or subsystem is closed by this work.
