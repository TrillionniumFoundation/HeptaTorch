# Native HIA packed int8 elementwise arithmetic — scoped component

A new standalone Linux x86-64 native HIA program, not a Python-backed runtime.
Actual source is i8_elementwise.hepta. It performs byte-wise arithmetic,
bitwise operations, exact input-length and mode admission, and packet output.
Both operands and result occupy one byte each, unlike the independent int64 lane
component. Protocol: little-endian u32 mode, u32 count, exactly 2*count operand
bytes; response is u32 status and count result bytes. Modes 0..5 implement
add, subtract, multiply, bitwise and, or, xor. Arithmetic wraps modulo 256.

Reproducibility:
- Source SHA256: 7ca1765e2cede95ec19b64742f69816581733619b5b1b844f57aa72b743ed8d3
- Compiler SHA256: 87c247e4d1c3c9a449d2947fd4227fdc186d519bffd203a04cafce7861a4ed54
- Native ELF SHA256: eba8c05bfe514c38ffa1a5e1ccbc9aacd513531f74becd18f4e3dca73db0a0b1
- Compiler source: Hepta f33d8da316de8fdf27b504371f1a5310997ff548
- Two independent native compiler invocations yielded byte-identical ELF.

Independent actual ROG execution: CPython and the exact pinned official
PyTorch 2.15.0+cpu wheel each pass 28/28 cases, 1,093,216 byte-exact
comparisons, all 65,536 operand pairs for each of six modes, 100,000 seeded
random pairs per mode, one full 100,000-pair admission boundary, empty
responses, and nine malformed-input refusals. Locked PyTorch provenance
requires eleven independent checks, all passed. Complete source-bound
reports are in validation/native-i8-cpython.json and native-i8-pytorch.json.
GitHub hosted CI is separately required and cannot inherit the ROG PASS.

This remains only a bounded contiguous CPU int8 kernel. It has no Tensor
dispatcher, broadcasting, in-place/out forms, views, promotions, autograd,
other dtype/device support, or qualified complete ATen schemas. Preserve
2,590 upstream native declarations, 688 derivative entries, 17 full-system
HOLD gates and full-schema parity zero. Do not merge until the root owner's
PR28 integration is complete and exact-head hosted CI and artifacts pass.
