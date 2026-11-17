# R4 successor: wrapped float64 Scalar input
This revision starts from the immutable 638-function R4 source67307c29 and imageef877aa6. The initial strict26 run remains FAIL:24 executed PASS, cases22/25 NOT_RUN because a real wrapped float64 Scalar input was not representable.
Command114 now constructs an original rank-zero dtype3 Tensor with eight-byte Storage and real Version0, constructs a separate F32 promotion Tensor, runs the F32 Tensor-times-Scalar computation, and records the original input and Scalar with the existing Saved Tensor ownership. The promotion temporary is not Saved. Backward reads dtype3 via a dedicated software binary64-to-binary32 conversion. The probability conversion and global floating arithmetic functions remain byte-identical.
Conversion covers sign, signed zero, normal/subnormal values, nearest-even rounding, underflow, overflow, infinity and NaN quieting/payload truncation. This source contract is not a claim that the fixed26 reference validates every binary64 value.
The new dtype is internal to wrapped Scalar construction and public original-Saved observation. Generic public97 creation,103 binary, public ToCopy casts and unsupported mutations retain explicit rejection. Read-only typed width/data, Saved access and StorageWeak must carry the real dtype3/8-byte extent. Public Weak/Hold enum checks expand, with private Hold dtype-zero and ownership equations preserved.
The F32 transpose constructor retains TBackward and gains the existing genuine Tensor-level ultimate-base retain. Shape view and reshape remain the actual ViewBackward / ReshapeAliasBackward / Clone→UnsafeView paths; no substitution with AsStrided.
Original638 source,13objects, first24 wire/raw and original reference26 are immutable. New revision qualification requires actual newly compiled units and the same full26 observations. Initial diagnosis may execute only originally blocked22/25. No full PyTorch schema or global acceptance gate is upgraded here.

Explicit public-surface guard additions: legacy command_argument matches only its fixed Tensor roles;55 data_alias,46 save_public and69 rng_strided_view reject dtype3 before any new allocation.97 and110 reject dtype>2,108/109 cast targets>2 or source3 reject,99..102 reject source3. Read-only96/98/13,107 and StorageWeak/drop preserve the real8-byte type.

Original reference26 cases22/25 phase2 explicitly preserve qNaN sign/payload through multiplication by2.0. The unchanged global mulf canonicalizes NaNs; this successor adds a local raw-NaN multiply wrapper for command114 and an exact parent back_binary copy with only its two op3 mul calls rebound. Dispatch requires actual op3 with original right dtype3. Global mulf, old ordinary binary backward and division remain unchanged. The independent32-scalar reference is separate and not yet claimed qualified.

Source05 binds the separate fixed CPU wrapped-float64 reference32 f3ecc9dd7db2ce8fbd0e424557ad5b95ac2e899a1c09cf38dad47e4fecc2b0c6: infinite F32 multiplied by either signed zero produces the observed negative quiet NaN 0xffc00000. The local wrapped_mul_f32 now tests the two symmetric Inf/zero combinations after its general input-NaN propagation and before delegation to the unchanged global mulf. This is a numerical operand-class rule, not case-index or fixture-value selection. Original eight-byte Scalar Saved bits, including signaling NaNs, remain unchanged; only temporary F32 computation quiets NaNs. All source04 runtime and non-math author bytes stay exact except this overlay hash binding and this explanation. No actual32 native qualification is claimed by this source preparation.


## Source06: public-token and internal-node contract correction

This successor preserves source05 and its actual evidence: full reference26 is
FAIL (0/26, report f7a9d470e91faac44055b5db5194a839cc45529970947752d603a3970d0c3d5b);
the independent fixed wrapped-scalar32 passes only its finite arithmetic and Saved
scope (d99df678ed9d256bfb1e26ba3871a0edb0ca8233cd85836d0f640701da5e24c0).
No source05 result is reassigned to this revision.

The source05 command guards incorrectly passed a public typed Tensor token to
tensor_dtype, which expects an internal descriptor index. The public cmd6 path
already resolves its source through command_argument, so transpose must validate
and refresh that internal node without decoding it as a second public token.

The new read-only scalar_peek_resolved follows exactly one original relative-handle
resolution: marker token>>24 == 254, nonzero 24-bit index <= current result count
at state+104, with a nonzero result table at state+96. Invalid resolution returns
zero without writing the error word. scalar_peek_public follows public_node's
actual validation after that resolution: so_tensor_live validates kind, generation,
live row, Storage and Version; node is bounded before dptr; descriptor live+304,
public token+192 and public flag+200 must agree. Neither helper calls terror,
to_view_refresh, allocation, release, or a write operation. They only guard a
valid dtype3 public Tensor; invalid inputs fall through to the unchanged parent
command so its reserved/payload/token precedence remains authoritative.

Typed commands99..102 have one relative resolution through public_node. To commands
108/109/111/112/113 have an outer resolve_handle then the target public_node, so
their guard performs the same two single resolutions. No recursive handle walk or
extra accepted handle depth is introduced. The correct command_argument,114,
to_copy_tensor,save_public,data_alias and rng_strided_view contracts are unchanged.

Transpose now calls valid(node), then to_view_refresh(node), then the exact prior
transpose. Its TBackward operation and real T-level ultimate-base ownership remain.
All scalar math, original F64 Saved Tensor, dtype guards outside these two command
wrappers, shape/reshape algorithms, global mulf, and backward helpers are byte
unchanged from source05. This source preparation is not native qualification.


## Source07: true wrapped-scalar backward layout and retained IEEE addition

Source06 remains an overall FAIL: its unchanged full26 reader reports 24 PASS and
2 FAIL (afe47e394b3334f22f503e606f806e6abc8b3eef32ad5c5884cb3051d0dec598), while
fixed32 passes only that independent domain (21b0d01fc77ffdb7f503aa43575076c047e21a73f2d60b18c219fbbdae92c1f8).
All 26 source06 sessions exited zero; its original numerical/Version failures and
the separately diagnosed raw-address lifetime collision are preserved.

The old op3 incoming description unconditionally declared a fresh contiguous
layout. For an actual wrapped-F64 scalar multiply, grad * scalar is a real
pointwise result. Its layout follows the incoming gradient operand, with the
rank-zero scalar contributing broadcast-zero strides. The new description is
restricted to op3, the original right operand dtype3, and the first AD edge. It
uses the existing signed byte-stride TensorIterator ordering helper on the real
incoming strides and real rank-zero scalar, creates dense output strides, and
sets a fresh zero-offset incoming Storage description. An inverted axis permutation uses the existing contiguous layout; otherwise
compatible strides multiply each actual dimension, including zero. This follows
TensorIterator allocate_or_resize_outputs rather than special-casing all empty
Tensors as contiguous. No recorded reference dimension or expected
Version is used to choose a branch.

The preflight wrapper marks this AD parent for actual private preserve allocation.
Existing incoming_construct materializes the computed logical gradient into that
real physical layout, through the original allocation/trim/scatter/cleanup path.
AccumulateGrad then naturally tests its existing layout contract: a transposed
incoming gradient for a contiguous dense leaf goes through the existing fresh
clone/copy path and its real Version increment. Neither Version values nor
public gradient IDs are directly overwritten by this revision.

The existing retained hook creates a fresh gradient on repeated use, matching
fixed variable.cpp. Its one addition call now uses the already frozen
scalar_add_preserving_bits primitive (b9d272f526920879e66970641f60cfcc0fc1d71aaf3c285e2781a5a909c6a436).
This preserves/quietens IEEE NaN sign and payload, gives the fixed negative NaN
for opposite infinities, and delegates ordinary finite addition to unchanged
addf. The helper and global addf/mulf/divf bodies remain byte-identical; no seed,
case number, or expected output constant controls the retained call site.

The actual reference's released Saved float64 allocations can have their raw
TensorImpl/StorageImpl addresses reused by later float32 gradients. This is an
independent comparison-lifetime issue under root review; production never
recycles an incarnation or changes retained allocation to chase an opaque address.
The source preparation runs no native semantic test or Torch. Public commands,
owner layout, 672-byte descriptors and existing format tags remain unchanged.
