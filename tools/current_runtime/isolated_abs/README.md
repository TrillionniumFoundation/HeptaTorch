# Native wide absolute-value component

This isolated Linux x86-64 HIA component implements the raw-word kernels needed
for bounded CPU `abs` on `int64` and `float64`. It does not register a public
Tensor opcode, add an autograd rule, or change the current image, test roster,
capability inventory, or full-contract denominator.

- I64 operates on two's-complement words and wraps at `INT64_MIN`.
- F64 clears the sign bit. This maps negative zero to positive zero and preserves
  the remaining bits of subnormals, infinities, quiet NaNs and signaling NaNs.
  No floating-point conversion occurs.
- The standalone harness uses the existing owned-session wrapper with a private
  9-byte request body (tag 2 for I64 or 4 for F64, one little-endian word).
  A 32-byte response begins with status 0 and the result word, or status 1 and
  no result. All padding bytes are zero. It is not the Tensor command protocol.

## Reproduce

Use the ABI3 module compiler and the entropy session linker identified in
[the current runtime build guide](../../../docs/current-runtime.md).
The verifier checks both tool hashes before executing their snapshot copies.
No Python or C implementation supplies the native results.

```sh
python3 -I -B verify.py \
  --compiler /absolute/path/to/stack-module-compiler \
  --linker /absolute/path/to/generator-session-linker \
  --output /absolute/path/to/fresh-output
```

The verifier cold-compiles and links twice, checks object and ELF reproducibility,
and executes a static ELF against an independent mathematical oracle. It covers
bit extrema and deterministic arbitrary words, refuses invalid tags/body lengths,
then confirms subsequent valid replies in the same session. It also checks
malformed outer requests and clean EOF.

Add `--torch-python /absolute/path/to/pinned-env/bin/python` for a separate
CPU PyTorch differential using the locked version (2.15.0+cpu), CPython 3.12,
AVX2, and single-thread execution. The caller must obtain and verify the
[locked official wheel](../../../compatibility/pytorch-oracle-lock.json)
before invoking it; this optional comparator does not audit installation
provenance. The included dot run had no PyTorch reference interpreter and
records `NOT_RUN` for this differential, not a PyTorch pass.

## Recorded dot run

Two native compiles and links produced the same 12,288-byte ELF. The program
passed 2,186 I64 and 2,070 F64 mathematical comparisons, 264 rejected bodies
with valid replies on both sides, 3 malformed outer controls and clean EOF.
There were 5,048 in-session frames in total. See [receipt](native-receipt.json)
for exact identities. The next integration must separately implement Tensor
dispatch, metadata/layout, resource-failure and same-build differential
qualification; none are claimed here.
