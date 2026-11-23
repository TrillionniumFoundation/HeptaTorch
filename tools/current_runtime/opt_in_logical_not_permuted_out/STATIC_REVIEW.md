# Opcode 128: bounded destination permutation, source-only review

Status: draft, uncompiled and unexecuted. No new opcode, native or Torch
evidence, or default/production change. Only the opt-in next lane is in scope.
No Numbered #803 view development is involved.

## Patch and composition

`logical_out_permuted_f32.hepta` adds one bounded HIA helper.
`prepare_source_review.py` describes the complete branch edit and runs
only static byte/structure checks. It does not emit a runtime or call
a HIA compiler. The only existing body that changes is `logical_out_apply`,
in `tools/current_runtime/source/runtime.hepta` and its
`units/runtime800_12.hepta` partition for this exact source family.

- The exact `f9807c9` predecessor SHA-256 is
  `e43b13aa0d4ad14d297f03b82212e73f0f13b4377dc3067a4e12fed7deaba1f8`.
- The pending numeric-self source SHA-256 is
  `bd811a3b484002467c8f54c67c138f6d8701fc378a389b67422d61db5ca5783b`.
- The composed source SHA-256 is
  `97f5e000faa8bd779807d5b683dbfaff795767a0385b10039ba175a07d4d0f25`;
  the composed unit SHA-256 is
  `385568d42bafa9c73443f4c172088de3e92138f25a82937f7c55f3cd9b682220`.
  The static function inventory is 804. Both patch orders agree byte
  for byte for the monolith and unit, and each permutation patch
  independently reverses to its own exact starting bytes.

The fallback can run only after the existing source layout succeeds and
the pre-existing `ls_disjoint` test proves different handles, nodes,
Storage IDs and Version IDs. Old contiguous exact self is untouched.
This first draft admits only F32 destination, positive strides bounded
by numel, rank 2–4, nonempty, zero offset, genuinely noncontiguous dense
layout, and a retained exact flag-1 no-grad/no-edge view of an ordinary
zero-offset contiguous root with the same element count. The original
source dtype set and exact shape gate are unmodified. Other destination
dtypes, offsets, empty views, singleton zero strides, same-storage
views and autograd are explicitly out of scope.

### Safety and constructor reasoning

The existing `transpose` calls `scalar_previous_transpose` and captures
the ordinary F32 descriptor, unchanged storage start, and permuted
shape/strides. `to_view_root_retain` records the canonical root
token/incarnation, shared Storage and Version, and ViewInfo flag 1
for this ordinary no-grad case. Thus the code contains a candidate
constructor; physical admission is still a later prerequisite.

`to_view_base_node` validates a retained root and matching dtype,
Storage and Version; calling the old contiguous layout checker on that
root rules out live grad/edge and nested/nonordinary roots.
`is_dense` is only used after proving positive dimensions, exact
bounded product with subtraction/division guards, and all strides
positive and bounded. By matching successive running products in
ascending active-axis stride order, dense layout maps logical indices
bijectively to `0..numel-1`. Singleton axes contribute zero offset;
zero strides are intentionally refused even there. Raw `is_dense`
would be insufficient (empty shortcut, unchecked multiplication,
ignored singleton strides); `dist_no_internal_overlap` misses
positive overlapping strides and `dist_dense_layout` changes layout.

The destination's actual tagged byte start must equal tagged storage
base plus zero offset. Its fully checked positive interval fits the
live Storage row and cannot wrap the tag region; `numel*4` is bounded.
Distinct live allocator-owned Storage IDs are assumed disjoint under
the existing owner invariant; the physical reader must verify ranges.
`taddr` maps logical coordinates using stride in elements and the dtype
width, whereas `to_view_logical_offset` is not the general raw-byte
path. Do not call `to_view_refresh`: the new path intentionally has
no AD refresh or descriptor mutation. Resolve and validate all owner,
shape, span and version conditions before reading or writing; keep
`tget`/status, taddr/store per pair, one shared version bump, and
the existing full trial-frame commit/rollback. The existing outer
`to_view_sync_requires` is not used as a prerequisite.

## Physical matrix for a later, authorized replay

| Control | Exact expectation |
| --- | --- |
| Predecessor, F32 root [3,2] and transpose destination [2,3], strides [1,2], offset 0 | Construct a real retained view first with clean owner audit; op128 returns 7, no records; raw owner and data unchanged. If constructor is refused, discover the real semantic collision before relaxing anything. |
| New rank-2 permutation, input [2,3] = [0,2,0,4,5,0] | Existing returned view handle, logical output [1,0,1,0,0,1], base physical order [1,0,0,0,1,1] (little-endian F32 encodings of 0 or 1); input unchanged. |
| New rank-3 permutation, base [2,3,4], permute axes [1,2,0], [3,4,2] strides [4,1,12] | Source zeros only at logical indices 0 and 5; physical ones only at base indices 0 and 14. Construct via existing transposes if no direct permutation op. |
| Singleton, base [2,1,3], permute [2,1,0], [3,1,2] strides [1,3,3] | Two genuine non-singleton axes; source zeros at logical 0 and 3, physical ones at base 0 and 4. No singleton layout rewrite. |
| Alias visibility and sequencing | Same source/result Tensor and storage identities as their pre-state; original base/view/sibling retain their existing shared Version ID and all see +1 once; source Version/RNG/bytes unchanged; destination/base shape/stride/offset/storage span/view token/incarnation/flags unchanged; no fresh owner nodes or accidental AD edges. Verify existing returned handle with same-frame op13/98. |
| Reject non-dense or unsupported destination | A constructible gapped stride, positive overlap [1,1], zero-expanded stride, positive aligned nonzero offset, empty view, rank-1 or merely contiguous true view, and non-F32 permutation refuse opcode128 with no records and byte-exact rollback; record constructor refusal separately if it is not constructible. |
| Reject disallowed relationship | Same Storage or Version with distinct handles/views, source view, equal-count wrong shape, grad/edge on input/view/root, nonzero reserved, stale/future handle. Wrong shape/unsupported layout status 7; retain existing lineage/invalid handle status where applicable. |
| Trial-frame rollback | Successful permutation write followed by malformed op128 (or invalid resolved source) leaves the whole owner, backing, source/destination/shared Version and RNG byte identical. |
| Predecessor compatibility | Exact prior contiguous-disjoint and numeric-self corpus retains results and refusal controls. Include current selected opt-in gate receipts. |
| Independent offline reader | Reparse exact request/response bytes, calculate expected physical offsets independently, inspect raw Tensor/Storage/Version/ViewInfo relationships and complete stream EOFs; no executable or Torch starts during readback. |

Guard-only cases such as forged lying numel, enormous strides, pool
wrap, and saturated Version require a separately authorized internal
fixture; they are not labeled user-wire tests here. Do not weaken a
constructor or owner invariant to produce a test case.

Split-runtime check: export the one new helper in runtime800_12 and
add only its missing external declarations from the exact core
signatures. Verify all 15 units again with the existing build source
checker. Specifically distinguish positive repeated active strides
(e.g. [2,2] at [1,1]) from zero-stride expansion: the former must be
rejected independently by the dense checker. For stale view grad state,
create the view no-grad first, then toggle the root through the original
public API and test both a later frame and same-frame root-toggle then
opcode128. Inspect whether existing per-record synchronization already
sets the view flag; do not label it stale if it is synchronized. If no
public transition can leave a genuinely stale view at the command
boundary, the explicit live root no-AD guard remains defense in depth;
do not forge owner state via public inputs to simulate it.
