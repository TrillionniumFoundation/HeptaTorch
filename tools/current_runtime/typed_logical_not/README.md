# Native HIA logical-not component

This is a standalone x86-64 HIA component for the **out-of-place** CPU
`logical_not` operation. It is not the existing in-place command and does not
register a new production-runtime opcode or close a complete PyTorch contract.

It evaluates one numeric word at a time using the current runtime's exact
`logical_truth` implementation ([source](https://github.com/TrillionniumFoundation/HeptaTorch/blob/118560581713b2c2ed3e6a12e5ab5770d3ebb385/tools/current_runtime/source/runtime.hepta)).
The component accepts F32, canonical Bool, I64, the two F64 tags, F16 and BF16.
Both floating zeros are false; infinities, NaNs, finite subnormals and every
other nonzero value are true. The output is the Boolean inverse. These are
the scalar semantics of
[PyTorch 2.15 logical_not](https://docs.pytorch.org/docs/2.15/generated/torch.logical_not.html).

The component's test-only binary protocol is a little-endian u64 dtype tag,
u64 count, and exactly that many zero-extended u64 raw words. Status and
results are returned as u64 words. A valid request gets a zero status followed
by one 0/1 word per input. An invalid header, dtype, count or request length
gets status 1, no values. The maximum is 60,000 input words. The component
deliberately does not handle production tensor layout, allocation, aliasing,
versioning, or autograd.

Run the component with the pinned standalone Hepta bootstrap compiler:

```sh
python3 -I -B tools/current_runtime/typed_logical_not/test_component.py \
  --compiler /absolute/path/to/native-compiler \
  --source tools/current_runtime/typed_logical_not/component.hepta
```

The runner requires exact compiler SHA-256
`87c247e4d1c3c9a449d2947fd4227fdc186d519bffd203a04cafce7861a4ed54`.
It compiles twice, checks byte-for-byte ELF equality, and compares every
result with independent Python numerical decoding. Cases include every F16
and BF16 bit pattern, floating signed zeros/NaNs/infinities/subnormals,
I64 extremes, the 60,000-word packet boundary and nine malformed requests.
An existing official CPU Torch 2.15.0+cpu installation can additionally
exercise `--torch`; the local independent run did not run that option and
does not claim strict PyTorch qualification.

Before enabling the public runtime operation, bind it to a new cold-built
runtime image and compare actual typed tensor output, strides, independent
ownership, alias/refusal behavior and original mandatory groups using the
repository's strict same-build reference. This component does not change
the main runtime, the original mandatory groups, or the acceptance ledger.
