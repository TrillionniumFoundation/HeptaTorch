# Native int64 Tensor successor

This isolated producer starts from frozen 523-function source `6ffe509a…` and
image `830f0b4e…`. The old producer and observations are immutable. Python only
authors and orchestrates; every Tensor operation executes as native HIA.

The inner packet and outer session framing remain unchanged. Descriptor size
remains 640 bytes. Candidate 06 uses internal tags `HTINT002` and `HTOINT02`;
nonempty previous-profile owners, including the `HTINT001` predecessor, are
rejected. The predecessor's exact documentation is retained as
`PROTOCOL-pre06.md`.

## Typed storage slice

Dtype 2 denotes signed int64, stored in eight actual little-endian bytes per
physical element. Wire scalar words are the complete two's-complement bit
pattern. Dtypes 0 (binary32) and 1 (bool) retain their existing meanings.

Commands 97, 98 and 100 admit code 2 for construction, typed inspection and fill.
Clone, detach, view, transpose, Saved pack/unpack, Weak and Hold retain the real
dtype, byte extent, element strides, offset, Storage and Version rules. Existing
F32 inspectors and numerical kernels still refuse non-F32 data. All non-F32
requires-grad requests are rejected. A rejected ordinary frame preserves the
entire committed owner, Version and RNG state.

The first checkpoint implements the storage slice. Candidate 02 adds commands
103/104; candidate 03 adds 105; candidate 04 adds the actual Scalar operation 106.
Every candidate has its own source, image and evidence identities. This document
describes the current authoring interface, not a transfer of qualification.
Candidate 05 adds actual CPU promotion temporaries before output layout
selection. Candidate 06 adds real ultimate-base Tensor ownership for int64
views, as detailed below.
Candidate 07 changes only three int64 arithmetic bodies to follow the fixed
upstream pairwise overlap admission and overlap/shape/cast error ordering.
All 523 parent function bodies and the candidate-06 state format are unchanged.

## Int64 view ownership

For dtype 2, commands 6 (transpose), 17 (view) and 69 (as_strided) create real
views. A view owns a reference to the ultimate base Tensor, independently of
its own Storage and Version references. Dropping the base's public handle does
not release that Tensor while a view still owns it. A view of a view retains
the same ultimate base. A detached alias owns Storage and Version but has no
base Tensor edge; a clone owns independent Storage and Version. Command 101
rejects an actual view, including an empty view, with status 5.

The existing descriptor fields 584/592 now have a second explicitly checked
role: an int64 view with opcode 0 stores its ultimate-base node and incarnation.
The original F32 native-dropout Saved-mask role is unchanged. The base must be
an older int64 non-view Tensor with matching Storage and Version. Both the
target's field 600 and its actual node reference count at 208 are retained and
released. The audit independently reconstructs these references in temporary
field 616 and clears that field before commit. No second owner or capability
is fabricated from serialized pointer words.

This is a dtype-specific extension of command 69. The existing F32 and bool
command-69 detached-alias contract is unchanged; it does not become a general
F32/bool `as_strided` view API by implication. Command 96 still returns the
existing dtype/width/incarnation/leaf tuple. `_base` is not a new public command;
qualification observes the real private base through read-only owned bytes.

## Arithmetic: commands 103, 104 and 106

Each row below is the complete eight-word command record. All unassigned words
are required to be zero. Tensor arguments accept the existing checked public
handle or preceding-result reference convention.

| Code | Record | Result |
| --- | --- | --- |
| 103 | `[103, op, A, B, 0, 0, 0, 0]` | New Tensor |
| 104 | `[104, op, target, rhs, 0, 0, 0, 0]` | The same target Tensor |
| 106 | `[106, target, scalar_i64_bits, 0, 0, 0, 0, 0]` | The same target Tensor |

Operation IDs are 1 add, 2 subtract, 3 multiply, 4 bitwise AND, 5 bitwise OR,
6 bitwise XOR and 7 true division. At least one binary operand must be int64.
The other operand may be int64, bool or binary32. Bool subtraction and binary32
bitwise operations are rejected. Binary32 input or true division produces
binary32; the other admitted combinations produce int64. Any requires-grad
operand is rejected with status 5: mixed-type autograd is not implemented.
In-place binary output must remain int64 and have exactly the target shape.
Unsupported operation/dtype/reserved fields refuse with status 7; incompatible
shape or admitted overlap checks refuse with status 2.

Scalar command 106 is a real int64 Scalar `add_` with alpha 1. It does not build
a Tensor RHS. Scalar and int64 add/subtract/multiply use all 64 input bits and
wrap modulo 2^64. Successful in-place operations increment the shared Version
once, including empty tensors. Scalar and binary in-place writes reject the
existing zero-stride internal-overlap condition. For pairwise overlap, a
nonempty operand that is not non-overlapping and dense yields the upstream
`TooHard` classification, which the fixed implementation admits. If both
operands are dense and share Storage, disjoint byte intervals are admitted;
intersecting intervals are rejected unless both starts and ends, and the entire
stride vectors, match. The fixed upstream contract permits some genuinely
overlapping nonzero-stride views. The native operation reads and updates their
real shared bytes; it does not silently clone the input to impose a different
result. The older candidate-06 conservative rule and its three actual40
failures remain intact.

In-place memory-overlap checks precede broadcast-shape and output-cast checks.
Bool subtraction retains its earlier operation-specific rejection. Invalid
ordinary frames still roll back the complete native owner and RNG state.

Conversion from signed int64 to binary32 uses integer round-to-nearest,
ties-to-even in HIA, without an intermediate binary64/Python conversion.
The original candidates 03/04 compared unconverted operand strides and retain
their separate evidence. Candidates 05/06 perform actual HIA allocation and
conversion of each operand whose dtype differs from the result dtype before
selecting the output layout. These temporary Tensors use Preserve semantics:
dense inputs keep their element strides, while non-dense inputs use inferred
dense strides. Layout comparison consequently uses the converted operands'
byte strides, including the fixed target's signed wrapped byte-stride rule.
The original inputs' data and Version remain unchanged, and request-local
temporaries are released through the existing real reference-count machinery.
This is internal CPU promotion, not an exposed public `Tensor.to` API.

## Explicit late-error native dropout: command 105

The record is `[105, input, Generator, p_binary64_bits, training, 0, 0, 0]`.
Training is 0=False, 1=True or 2=None. This entry accepts int64 input only;
command 99 and all existing F32 interfaces retain their own contracts.

An admitted response has a 16-byte payload containing two u64 words,
`[runtime_error, mask_handle]`. Success has error 0, the output Tensor in the
record value and the actual mask in the payload. A nonempty training path
performs the real RNG/mask/int64 multiplication work, then reports error 12
for the floating scaling result that cannot be cast into int64. Its record
value and mask handle are both zero. Header status remains zero, the suffix is
not executed, consumed Generator state is committed, and unpublished temporary
mask/output tensors are actually released. Input values and Version remain
unchanged. This is an exposed structured exception, not a successful output.

Empty input returns the same Tensor/Storage/offset and a distinct empty int64
mask with Preserve layout and offset zero, without RNG consumption. Nonempty
False creates an independent int64 output and true one-byte bool mask, without
RNG consumption. The current native dropout element limit is 1,000,000.

Invalid handles, flags, capacity, allocation, or other admission failures keep
the existing nonzero-header whole-frame rollback contract. Output payload
capacity and required allocations are checked before RNG mutation; this
allocation-error policy is not a claim of every PyTorch OOM side effect.

The fixed actual54 report is
`4c454e443c4147c740ea400cb1195bcbcb80e16e5b50debce54c068fa9ae712b`.
It remains the source of expected results; candidate outputs never create those
expectations. Initial equal Version counters do not establish shared identity.
Full PyTorch completion remains false and the schema denominator remains 2590.
