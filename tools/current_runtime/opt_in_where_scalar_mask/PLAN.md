# Scalar condition only, atop exact #66 3b5061f

A true rank-0 Bool condition (exactly one element) can be broadcast over
existing equal-shape ordinary value roots. Both value dtypes still must
match exactly within the public set; all three input handles still pass
the exact live Storage/Version, layout, zero-offset, contiguous, no-grad,
rank/count and raw span admission. A length-one vector is not a scalar
unless the old equal-shape path accepts it. Neither value is broadcast.
Use the original command opcode/status/transaction, result allocation and
exact output kind/geometry. No default-visible routing.

One body (`where_self_tensor`, unit12), no ABI or helper changes. Condition
element step is zero for the rank-0/count-1 case, one for all existing
accepted non-scalars; `index * step` is thus always between 0 and the
admitted condition count for every actual load. Empty results still run
no payload iteration, but allocate a distinct fresh typed result. The
existing scalar/scalar input now takes step 0 at index 0, which is exactly
the original load. A trailing singleton axis would require additional
rank/axis checks plus division and empty-axis rules; the scalar path is
smaller and adds an independently useful form without a general planner.

## Later physical acceptance
* Two independent cold native builds on exact #66; start locked official
  CPU Torch in its own process. For each value dtype (six), make true
  rank-0 Bool true and false cases with nontrivial equal rank-2 or rank-3
  roots. Exact selected raw bytes should match the entire corresponding
  branch input, including both signed zeros/subnormals/qNaN/sNaN/Inf,
  I64 extrema and Bool. At least one typed empty with shape [2,0,3],
  bounded scalar packet and repeated x/y input; use matching dtype only.
* Exact #66 physically constructs the same three valid inputs and refuses
  newly admitted calls with status7 (this is a dispatch/admission gap, not
  unknown opcode4 anymore). Read back exact frames, input objects and full
  owner images. Successor no-Torch process + independent raw reader confirm
  fresh Tensor/Storage/Version identities, Version=0, no input byte or
  metadata changes, both input/result release orders, weak expiry and
  result-handle same-frame reference. No pointer comparison for empties.
* Keep live real refusals for a non-scalar [1] or [1,1] condition and a
  trailing singleton coordinate, a numeric scalar condition, incompatible
  x/y shapes (including two equal-numel unequal shapes), mixed x/y types,
  genuine viewed condition/values, stale handles, nonzero padding,
  marked-grad inputs. Check exact before/after owner bytes. One viable
  scalar where followed by known invalid opcode in same frame must roll
  everything back. Real memory pressure only if exact status3 and the
  same packet resumes after removing the local cap; use existing #66 case
  as retained proof otherwise.
* Replay 23 #66 existing case rows, both scalars and all empties included;
  all prior wire traces are expected byte-identical when no scalar
  expansion invokes the new admission. Keep #65’s four retained matrices.
  Distinguish baseline refusals from successor controls. No hosted claim.

Reproduce source check only:
`python3 -I -B prepare_scalar_mask.py EXACT_PINNED_SOURCE EXACT_BUILD_PY`.
The project script keeps all large source buffers in memory, verifies full
split/source equality and reverses both transformed streams exactly.
