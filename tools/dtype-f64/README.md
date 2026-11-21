# Native HIA binary64 arithmetic

**Component status: PARTIAL.** The native arithmetic passes the full CPython
bitwise corpus. The recorded unchanged strict PyTorch corpus fails four sqrt
groups: 4,008 observed non-NaN results differ by one encoding step. Integer
rounding-cell checks show native RNE for these inputs and non-RNE results from
the fixed MKL-backed reference. This does not close fixed-backend bitwise parity,
the full dtype gate, or the complete PyTorch rewrite.

`f64.hepta` implements binary64 addition, subtraction, multiplication, division,
square root, exact binary32 widening and rounded binary32 narrowing. Every target
operation uses HIA unsigned integer instructions. The emitted ELF has no Python,
PyTorch, libc floating-point or host arithmetic dependency.

The add/subtract/multiply/divide and conversion bodies are already used by the
current public F64 Tensor implementation, including the bounded no-gradient
mixed-F64 and same-F64 arithmetic paths. The
[current-object packet consumer](../current_f64_consumer/README.md) links all
fifteen objects from the actual current cold build: eleven original function
bodies are byte-identical in the current source, while the packet entry, byte
writer and scalar `sqrt64` adapter retain their original bodies with only
symbol renames checked by an exact inverse. `f64.hepta` and `profile.json`
remain required source witnesses.

Scalar packet sqrt is separate from the public Tensor interface. This consumer
does **not** add a public F64 Tensor sqrt operation or full F64 autograd/device
support. The strict sqrt mismatch remains FAIL; the separately proved RNE
component remains PARTIAL. The800 public-view extension has only local
exploratory execution and complete physical replay so far; its strict view gate,
full35-group run and current-object component CI remain required. See the
[current audit](../../docs/current-audit.md) and
[public-view contract](../../docs/native-public-f64-views.md). The complete
dtype/operator gates remain open and whole-project status remains HOLD.

## Numerical contract

Values are IEEE binary64 encodings. The selected rounding rule is nearest with
ties to even, including gradual underflow and finite overflow. Signed zeros and
infinities are handled explicitly. Invalid operations return a quiet NaN; exact
NaN payload and sign propagation are not claimed. Floating-point exception flags,
alternate rounding modes and device-specific modes are outside this profile.

The core has 14 functions, 191 blocks and 669 HIA instructions. Four helpers
(`getr`, `putw`, `shrjam`, `packf`) are copied byte-for-byte from the canonical
binary32 runtime and individually identified in `profile.json`.

Addition aligns extended significands with a sticky bit. The shared binary64
packer retains three rounding bits, normalizes, handles subnormal results and
rounds once. Multiplication computes both limbs of a 53-by-53-bit product from
32-bit partial products; it preserves discarded information before rounding.
Division uses 56 binary quotient steps and jams a nonzero final remainder.

Square root processes a 112-bit scaled radicand two bits at a time. Its running
root and remainder fit within unsigned 64-bit words. The integer recurrence
retains an exact remainder, so the final rounding distinguishes an exact boundary
from values above it. No approximate reciprocal or approximate square-root seed
is used. Exponents are stored with a positive internal bias to avoid signed
underflow in intermediate exponent arithmetic.

## Packet contract

All words are little endian. A request begins with a 32-bit operation number and
a 32-bit count. Binary operations then carry pairs of 64-bit words; unary and
conversion operations carry one 64-bit word per value. The count limit is 60,000,
and the entire request length must match exactly.

| Operation | Number | Input per value | Output per value |
| --- | ---: | --- | --- |
| Add | 0 | Two binary64 words | One binary64 word |
| Subtract | 1 | Two binary64 words | One binary64 word |
| Multiply | 2 | Two binary64 words | One binary64 word |
| Divide | 3 | Two binary64 words | One binary64 word |
| Square root | 4 | One binary64 word | One binary64 word |
| Widen binary32 | 5 | Binary32 encoding in a zero-extended 64-bit word | One binary64 word |
| Narrow to binary32 | 6 | One binary64 word | Binary32 encoding in a zero-extended 64-bit word |

A successful response begins with an eight-byte zero status and the output words.
Malformed requests return only status 1. A widening input with nonzero high
32 bits returns only status 2; all such inputs are checked before writing output
values. There is no persistent target state.

## Independent qualification

The commands below describe the retained original standalone component and its
fixed image. The old `dtype-f64` workflow remains in this candidate. Use the
[current-object consumer instructions](../current_f64_consumer/README.md) for
the replacement path; its strict execution and original-artifact readback must
complete before retiring the old workflow. Local797 current-object results do
not grant800 qualification.

```sh
python3 -I -S -B tests/f64_oracle.py \
  --compiler /absolute/path/to/pinned-hia-native-compiler \
  --reference cpython --output /absolute/path/to/new-cpython-review

python3 tests/f64_oracle.py \
  --compiler /absolute/path/to/pinned-hia-native-compiler \
  --reference pytorch --output /absolute/path/to/new-pytorch-review
```

The compiler must have SHA-256
`87c247e4d1c3c9a449d2947fd4227fdc186d519bffd203a04cafce7861a4ed54`.
Each run compiles the actual HIA source twice, requires byte-identical native
images and executes them with an empty PATH. The image identity is
`4225dd162b5c9fcf2a44987839bd410b5e37aef32665683389fb1f4f281f25a2`.
The profile, function bodies, source, producer and image are checked against
their recorded identities.

The fixed 55-case campaign includes special-value cross products, every exponent
field with representative mantissas, adjacent binades, 100,000 arbitrary bit pairs,
100,000 close-exponent pairs, halfway inputs and neighbors, multiplication/division
scaling boundaries, square boundaries, binary32 conversion midpoints, all half
encodings widened through binary32, empty packets and 16 refusal cases. Inputs
are deterministic and every request, response and reference stream has a hash.

The CPython campaign passed 1,968,156 non-NaN bit comparisons and 66,910 NaN
classification checks. The PyTorch mode uses actual `torch.float64` operations
and conversions, verifies all 11 existing wheel provenance checks, and compares
the same corpus without numerical tolerances. The relevant pinned upstream
[binary](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp)
and [unary](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp)
kernel source identities are recorded in the profile.

This campaign does not exhaust all 2^64 input encodings or all operand pairs.
Passing it is scoped arithmetic evidence, not full Tensor or upstream schema
closure. The separately required full-project acceptance record is unchanged.

## Correct rounding and the retained PyTorch differences

The actual first strict failures are preserved in
`validation/hosted-native-f64-initial-failure.json`. A separate hosted diagnostic
at commit `22b6329e23faaaead270a3a51dc6a8e5e803d3d7` confirmed MKL 2024.2 and
AVX512 in the fixed CPU wheel, zero native rounding-cell failures and 4,008
reference results outside the exact nearest-even cell. Its full original reports
are retained in `validation/hosted-f64-sqrt-diagnostic.json`.

[The independent mathematical review](SQRT_REVIEW.md) explains the HIA restoring
algorithm and two independent integer qualification paths. It distinguishes
Intel's HA error contract from correctly rounded results. A one-step difference
between encodings is insufficient to prove the HA bound: the new qualifier tests
each reference result against exact squared boundaries and requires error
strictly below one ulp of the exact result's binade.

The original 55-case oracle source, all inputs and exact comparisons remain
unchanged. CI saves its original exit code and full FAIL report. A separately
named component qualification repeats all 55 cases, requires every non-sqrt
strict comparison to pass, proves native sqrt RNE, proves the reference error
bound, and requires both original reports' request/response/reference stream
hashes to match the new execution. Every sqrt mismatch is saved individually.

```sh
python3 tests/f64_rne_qualification.py \
  --repo . --native build/f64-pytorch/f64-native --reference pytorch \
  --strict-report build/f64-pytorch/report.json \
  --cpython-report build/f64-cpython/report.json \
  --output build/f64-rne-accuracy
```

Successful mathematical qualification reports `status: PARTIAL` and a separate
`qualification_status: PASS`. It always retains
`strict_pytorch_bitwise_gate_closed: false`, `full_dtype_gate_closed: false` and
`whole_pytorch_rewrite_complete: false`. The stronger fixed-backend bitwise
target remains open. This component does not insert input-specific corrections
to imitate the observed MKL outputs.

## Executed hosted component qualification

Run [43805617936](https://github.com/TrillionniumFoundation/HeptaTorch/actions/runs/43805617936)
executed all 55 cases on commit `7fb3e34e833a91fc4a8d2eb0cdf5500978cde0a4`.
The full receipt is `validation/hosted-native-f64.json`: 236,892 finite positive
square-root inputs pass both native integer RNE checks and the strict reference
HA bound; all eleven oracle provenance checks pass. All 4,008 mismatches are
retained individually. The original strict oracle still exits 1 with four failing
groups. The separate Native HIA workflow on the same commit passed all 40 steps.
Neither the reference accuracy qualification nor the workflow success closes the
fixed-backend bitwise gate or the complete dtype subsystem.
