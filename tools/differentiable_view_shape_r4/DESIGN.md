# R4 concrete view / reshape source checkpoint

This is an isolated successor of the genuine HIA R3 image. No R3 source, object, image, reference or earlier failure is rewritten.

## Runtime geometry
shape_infer_dimensions implements the fixed concrete InferSize.h contract within the existing rank/dimension bounds. shape_compute_strides implements TensorUtils.cpp contiguous-chunk grouping, including scalar all-one strides, unchanged empty strides, changed empty contiguous strides, ignored singleton boundaries and zero-stride chunks. Failure is a real runtime result, not a case-table lookup.

## Construction and ownership
A typed descriptor owns the true element width. The true-view branches bind the same Storage and Version, copy the exact byte storage offset and geometry, and retain the ultimate Tensor base through the existing 640/648/656/664 ViewInfo role. That root is distinct from the immediate AD edge. Operation 45 binds the alias explicitly; operation 16 uses the original alias record path. The original integer view root hook is detected to avoid duplicate ownership.

The reshape fallback creates a real contiguous typed clone using logical loads/stores, including canonical bool conversion, and records the existing CloneBackward operation. It then builds a separate UnsafeViewBackward Tensor with no true ViewInfo base. Its Storage aliases only the temporary clone. The new Version binder creates a distinct zero-valued Version owner; this rule is being independently bound to the fixed upstream unsafe-view wrapper and TensorImpl construction chain before qualification.

All temporary descriptors keep the existing internal public-state value 2 until finish_command consumes their frame lifetime; only the returned output is published. No fake public clone handle or permanent base root is introduced.

## Backward
ViewBackward and ReshapeAliasBackward/UnsafeViewBackward use the existing incoming_reshape_layout and grad.reshape alias-or-copy decision. The new operation identities are separately visible. Numeric accumulation, layout-contract cloning, R3 AsStrided coverage arithmetic and weak/hook machinery remain byte-identical. A new ViewInfo's later real Version change may lazily rebase its current history through the already implemented AsStrided path; that is different from pretending the initial view/reshape backward is AsStrided.

## Evidence
Fixed Torch reference 26 source 4dd374419e4b21a9c814c855d846481730a12ea57a30dcf229a6b962f07021c5 has now actually observed 26 recipes (actual fecf1956790f78a9b1238f535c1c7cce9f1237b211c389ac0736d88687bbb9c9). Its values are immutable observations, not copied into production logic. The first native replay has not run. Previous R3 scopes 14, 4 and 72 remain bound to R3 only.

The independent executor alone performs all native compiler/linker calls after the source is frozen. Every unit is rebuilt from HIA, including the unchanged RNG/owner units. A prepare run is only source emission, call-graph/inverse checking and unit planning.
