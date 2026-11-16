# Binary64 uniform-bound admission for binary32 endpoints

The pinned `aten/src/ATen/native/DistributionTemplates.h` receives `from` and
`to` as `double`. It checks `(to - from) <= numeric_limits<scalar_t>::max()`
before the empty return. For a binary32 Tensor and exactly representable finite
binary32 endpoints, this is a **rounded binary64 subtraction** compared to the
exact binary32 maximum. It is not a real-arithmetic difference test.

The frozen standalone RN03 component implemented a stricter real-difference
contract. Its original source, package, native image, qualification and observed
refusal remain unchanged. In particular `[-1, FLT_MAX]` has an exact difference
slightly greater than `FLT_MAX`, yet the binary64 subtraction rounds back to
`FLT_MAX`. This new common profile explicitly corrects that admission rule.
It does not weaken a numerical comparison or change the MT/FMA implementation.

Let `F = (2 - 2^-23) * 2^127` be the maximum finite binary32 value. At `F`, the
binary64 spacing is `2^(127-52) = 2^75`. `F` is exactly representable in
binary64 and its low significand bits are zero, so it is the even endpoint at
the upper midpoint. For a nonnegative exact difference `d`, therefore:

```text
RNE_binary64(d) <= F  iff  d <= F + 2^74
```

The equality is accepted. Just above the midpoint rounds to a double greater
than `F` and is rejected. This comparison uses no float-encoding ULP-distance
shortcut and no backend output lookup.

Same-sign ordered finite endpoints cannot exceed `F` in real difference. For
opposite signs, write the magnitude sum as `larger + smaller`. If `larger` is
at most `F/2`, the sum is already at most `F`. Above `F/2`, `F-larger` is an
exact binary32 result by Sterbenz's lemma. If this gap is positive, it is at
least `2^104`; the next binary32 value above that gap is separated by at least
`2^81`, larger than the `2^74` binary64 midpoint allowance. Thus the ordinary
binary32 comparison with the exact gap is sufficient. If the gap is zero, the
rule is precisely `smaller <= 2^74` (`0x64800000`). These cases cover every
finite exactly representable binary32 endpoint pair in the declared interface.

The qualification C++ program evaluates the actual double expression directly.
The Python qualification also compares exact signed integers in units of
`2^-149` against `F + 2^74`. Forty-four pairs exercise both signs around the
midpoint and multiple positive gaps, with empty and nonempty output: 88 native
cases. Full output bits, complete Generator state, version increment and failed
frame rollback are checked separately. The real Torch probe uses the same
input corpus but records observations without imposing an acceptance result;
its provenance-bound output is compared only after an actual execution is
retrieved. A local C++ or integer result is never labeled as a Torch run.

The original RN03 negative-control script is preserved and run against the new
profile as a historical contract-change witness. Its two newly valid examples
are replaced explicitly by the next binary32 magnitude above `2^74` in a
separate source-bound replay adapter. Every other RN test and every numerical,
MT-state, ownership, alias, lifetime and rollback comparison is unchanged.
