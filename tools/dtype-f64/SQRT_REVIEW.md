# Binary64 square root: mathematical qualification and backend compatibility

The HIA component remains **PARTIAL**. Its scalar binary64 square root is
qualified for round to nearest, ties to even (RNE), under the stated component
and corpus limits. The frozen PyTorch bitwise oracle remains **FAIL** for the
recorded MKL execution: 4 of 55 groups contain 4,008 non-NaN differences. This
document does not replace that result with a tolerance-based PASS.

## What was observed

The preserved hosted diagnostic identifies Intel oneAPI MKL 2024.2, build
20240605, `USE_MKL=ON`, and PyTorch CPU capability `AVX512`. All 4,008 differences
were adjacent binary64 encodings. Exact rational midpoint tests found zero HIA
RNE errors and 4,008 reference RNE errors in the tested sqrt corpus. Repeating
the first 16 distinct witnesses at lengths 1, 2, 3, 4, 8, 16, 128, and 2,048
produced the same reference result for each witness. These observations do not
support attributing the discrepancy to vector tails. PyTorch's reported CPU
capability also does not identify MKL's exact internal dispatched instruction
sequence.

The fixed PyTorch source routes its contiguous double sqrt through
`vmdSqrt(..., VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE)` when MKL is enabled.
The source and observed configuration are consistent with this route; they do
not reveal the closed library's internal algorithm. The source hashes, URLs,
and actual downloaded documentation hashes are recorded in
`validation/f64-sqrt-primary-sources.json`.

## What the public contracts say

Intel's [VM accuracy overview](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-vector-math-performance-accuracy-data/2021-1/overview.html)
sets the HA design requirement below 1 ulp and permits processor- and
release-dependent results within that bound. HA is consequently insufficient
to establish correct rounding to the nearest representable number. Intel's
[2021.1 measured-accuracy table](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-vector-math-performance-accuracy-data/2021-1/measured-accuracy-of-all-real-vm-functions.html)
lists double sqrt HA errors of 0.52, 0.52, 0.52, and 0.54 ulp for its four CPU
columns. Those historical measurements establish that HA was not universally
RNE; they are not measurements of the 2024.2 library in this run.

The [2024.2 mode reference](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-2/vmlsetmode.html)
documents per-call accuracy and denormal handling. Explicit HA selects the
high-accuracy implementation, and FTZDAZ_OFF disables the fast denormal
handling mode. Changing a global default is not evidence of changing this
explicit per-call mode.

PyTorch's [numerical-accuracy notes](https://docs.pytorch.org/docs/2.15/notes/numerical_accuracy.html)
do not promise bitwise equality across releases, commits, or platforms. That
permits a distinction between API numerical accuracy and exact compatibility
with one fixed backend. It does not close the stronger bitwise compatibility
goal maintained in this repository.

## Review of the HIA algorithm

The reviewed source is `tools/dtype-f64/f64.hepta`, SHA-256
`44a6ff00413829d3cf8f7fbd91fb96a6baf090c3c754882d4361f49ed85a2e56`.
The native executable is
`4225dd162b5c9fcf2a44987839bd410b5e37aef32665683389fb1f4f281f25a2`.
The `sqrt64` body hash is
`2670f6ad1dc9fe7fb69f1fc8e18a8eec7710f45f25aac47dd8aadad9366ca2b4`;
the shared `pack64` body hash is
`1ac3d0eea9643c8e0f8bfc7929dbe73aaa62737a4659872ff1075aa9b2f1c215`.

For a positive finite input, normalization writes the input exactly as
`M * 2**(e-52)`, initially with `2**52 <= M < 2**53`. If `e` is odd,
the code doubles `M` and decrements `e`, preserving the value and making the
exponent even. After that step, `2**52 <= M < 2**54` and
`-1074 <= e <= 1022`.

The code forms the exact 112-bit integer `N = M << 58` as two 64-bit words.
The low-word shift deliberately retains the low 64 bits; `M >> 6` supplies
the high word. Its 56 iterations consume two input bits at a time. If `P` is
the consumed prefix, the loop maintains `P = q*q + r` and `0 <= r < 2*q+1`.
For the next digit `d`, it forms `4*r+d` and tests against `4*q+1`, which
selects between next roots `2*q` and `2*q+1`. Thus the final `q` is exactly
`floor(sqrt(N))`, with exact remainder `r=N-q*q`.

All arithmetic used by this loop fits its unsigned words. Before the last
iteration `q < 2**55`, so the trial is below `2**57`; the shifted remainder
is below `2**58`. A deliberately looser bound of `2**59` also suffices for
every intermediate. No 64-bit overflow can affect the decision. Pair shifts
remain within 0..63 in each selected word. Normalization and biased exponent
arithmetic stay nonnegative.

The root carries three extra bits relative to binary64 precision. Let
`sqrt(N) = q + delta`, where `0 <= delta < 1`. The retained significand is
`q >> 3`; the last three bits determine its position relative to the half-way
point. A nonzero exact remainder means `delta > 0`; OR-ing bit zero records
that fact without crossing the half-way threshold incorrectly. In particular,
a tail of exactly four becomes greater than four only when the exact result
is above the midpoint. `pack64` rounds above-half upward and exact-half to
the even retained significand. Carry into a new binade is handled by its
normalization step.

Positive finite binary64 square roots are finite and normal, including the
smallest subnormal input: `sqrt(2**-1074)=2**-537`. The source handles signed
zero before its sign rejection, preserves positive infinity, and produces a
quiet NaN for negative nonzero and NaN inputs. NaN payload identity and
floating-point exception flags are outside this component's promise.

This is a mathematical source audit plus executable qualification. It is not
a machine-checked proof of the compiler or an exhaustive execution of all
binary64 encodings.

## Independent checks

`tests/f64_sqrt_integer.py` uses a different construction. It selects the exact
result binade, forms the integer whose square root is measured in output
significand units, calls integer `isqrt`, and compares `4*N` with
`(2*q+1)**2`. It does not reuse the HIA restoring loop, sticky-bit packing, a
host floating-point sqrt, or any observed MKL output.

A second integer check constructs the two exact dyadic midpoints surrounding
the candidate answer, squares them, and checks the original input against
those boundaries, including ties-to-even. This independently agrees with the
earlier `Fraction`-based diagnostic's reasoning. Squaring preserves ordering
here because the relevant roots and boundaries are nonnegative.

The reference HA check uses the explicit quantum
`u = 2**(floor(log2(x)/2)-52)` for positive finite `x`. It proves
`abs(y-sqrt(x)) < u` by exact squared bounds around `y-u` and `y+u`.
It rejects equality at one ulp. It uses the exact result's binade, not the
candidate's exponent, and does not substitute distance between encoded words
for a real-error bound. Signed zero, infinity, negative inputs and quiet NaN
classification are checked separately.

For example, take `x=1-2**-53`, encoding `0x3fefffffffffffff`. The midpoint
between `1-2**-53` and `1` is `h=1-2**-54`. Exactly
`h*h-x=2**-108 > 0`, so `sqrt(x)` lies below that midpoint. Returning `1`
is therefore not RNE, even though it satisfies the strict one-ulp accuracy
bound. This argument uses only integers and rational arithmetic.

The independent recorded review ran 63,245 additional native values,
126,490 wrong-neighbor RNE refusals, 50,000 independent Fraction comparisons,
and 229,264 restoring-loop invariant checks over 4,094 inputs. Its 18 fixed
checker controls include exact-one-ulp rejection, binade crossings, signed
zeros, infinities, minimum subnormals and signaling-NaN result rejection.
The original CPython corpus was separately re-executed in full: 55 cases and
236,892 sqrt values, with both independent integer RNE checks passing.

## CI integration and open work

First run both unchanged original 55-case oracles and retain their actual
exit codes and reports. Then run the new qualifier:

```sh
python tests/f64_rne_qualification.py \
  --repo . --native BUILD/f64-native --reference pytorch \
  --strict-report BUILD/pytorch/report.json \
  --cpython-report BUILD/cpython/report.json \
  --output BUILD/rne-and-accuracy
```

The new qualifier re-executes the complete frozen corpus and all 16 invalid
packet probes. It binds every request, response, and reference hash to the
original reports, requires every non-sqrt strict case to pass unchanged,
checks native sqrt RNE, and proves the reference's strict one-ulp bound.
Every bitwise difference is retained in full in both its JSON output and a
separate mismatch artifact. It never accepts a particular mismatch count as
proof, and never changes the original report's status.

Successful execution means only `qualification_status=PASS` for this named
component. The aggregate status is `PARTIAL`, with
`strict_pytorch_bitwise_gate_closed=false`, `full_dtype_gate_closed=false`,
and `whole_pytorch_rewrite_complete=false`.

No MKL-compatible HIA replacement was derived in this review. A generic HA
accuracy bound and a finite collection of output witnesses do not determine
the library's algorithm. Exact backend emulation would require the actual
dispatched algorithm, its operation order and rounding semantics, and any
instruction-specific approximation behavior to be specified and reproduced.
Guessing a reciprocal-sqrt refinement, adding corrections for the observed
inputs, or changing the expected outputs would not establish that property.
The current reusable RNE primitive should retain its mathematical behavior;
any future exact-backend profile needs an independent identity and qualification.
