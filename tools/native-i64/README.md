# Native HIA contiguous int64 arithmetic (scoped)

This standalone Linux x86-64 executable implements six **actual native HIA**
`int64` lane operations. It accepts a little-endian pair-vector packet and
returns packed signed 64-bit results, without PyTorch, NumPy, libc numerical
routines, or a host arithmetic interpreter in the production path.

| `mode` | Operation | Wrap semantics |
| --- | --- | --- |
| 0 | `add` | two's-complement mod 2^64 |
| 1 | `sub` | two's-complement mod 2^64 |
| 2 | `mul` | two's-complement mod 2^64 |
| 3 | `bitwise_and` | bitwise u64 |
| 4 | `bitwise_or` | bitwise u64 |
| 5 | `bitwise_xor` | bitwise u64 |

Packet: `u32 mode`, `u32 count`, then precisely `count` pairs of little-endian
`u64` bit patterns (`a, b`). The response is `u32 status` (`0` success,
`1` refusal) followed by precisely `count` packed `u64` result words. The
native implementation rejects malformed headers, unknown operations, omitted
or trailing operands, counts over 48,000, and output-capacity exhaustion.

The HIA source is authoritative. The exact compiler is the fixed, self-hosted
Hepta bootstrap, SHA-256
`87c247e4d1c3c9a449d2947fd4227fdc186d519bffd203a04cafce7861a4ed54`.
Its standalone ELF is SHA-256
`fbb6cf1648385be200c94fc14ed73d1539cba3b0239aeea4f7866058c8395ebc`.
The fixed source SHA-256 is
`251705542f4e117e3c9cdaec36cead8ab38252b24ebfef1104d765c546fd0716`.
Two independent compiler invocations must produce byte-identical ELF images.

The independent CPython reference passed 33 of 33 cases, zero failures,
351,174 exact 64-bit result comparisons, including overflow/negative extrema,
10,000 seeded random pairs per operation, all 48,000-pair capacity boundaries,
empty inputs and nine strict refusals. See
`validation/native-i64-cpython.json` for source/ELF/input/response hashes.
The *separate* exact PyTorch 2.15.0+cpu wheel gate is defined by
`.github/workflows/native-i64.yml` and must pass its 11 independent
provenance checks before claiming PyTorch parity. The CPython gate alone does
**not** assert that it has run.

This does **not** implement the upstream Tensor frontend, shape broadcasting,
promotion, stride variants, aliasing, mutation, autodiff or complete `aten`
schemas. All 2,590 upstream native declarations, 688 derivatives and 17
full-project subsystem gates remain in the denominator; whole-project
acceptance remains **HOLD**.
