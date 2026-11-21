# Historical core execution protocol (HTORCH01)

This page records the original F32 `HTORCH01` core and its optional interactive
framing, implemented by the [fixed historical core source](../tools/legacy_source143/manifest.json).
Its command definitions are also referenced by later profiles, which specify
their own handles, metadata, transaction and dtype rules.

The default application is the
[current HIA runtime](../tools/current_runtime/source/runtime.hepta). Its
persistent owner/session interface, Generator service and dtype extensions are
qualified through the [current runtime entry](current-runtime.md) and described
by the [capability inventory](../compatibility/capabilities.json). The retained
[Tensor object protocol](../tools/tensor_objects/PROTOCOL.md) documents the owned
Tensor layer; [logical reductions](../tests/logical_reductions/README.md) records
the later reduction commands. The core wire fields and limits below retain
their historical scope.

The original HTORCH01 executable consumes one request from stdin, executes its commands in
order, and writes one response to stdout. Every integer field is an unsigned
64-bit little-endian word. A floating-point field carries the binary32 bits in
its low 32 bits. Tensor payload elements are four-byte little-endian binary32.

This is the original core execution and qualification interface. It is not a
Python `torch` import boundary or the PyTorch serialization format.

## Envelope

The 32-byte request header is:

| Offset | Value |
| ---: | --- |
| 0 | Eight literal bytes `HTORCH01` |
| 8 | Number of commands |
| 16 | Exact total request size in bytes |
| 24 | Reserved; must be zero |

Exactly that many 64-byte command records follow, then optional payload bytes.
Each record has eight words: `opcode, a, b, c, d, e, f, g`. Unused fields should
be zero. Tensor payload offsets are absolute request-byte offsets, four-byte
aligned, and at or after the end of the command records. Optimizer, tensor-list
and index-configuration offsets are similarly bounded and eight-byte aligned.

The 32-byte response header contains literal bytes `HTORSP01`, version `1`, the
number of completed command responses, and the final status. Each command
response has four words followed by its payload:

`opcode, status, result, payload_byte_length`

On the first command error, the response contains that failed command with zero
result and zero payload, and execution stops. A malformed envelope has no
command responses and final status 100. Exhausted output space can terminate
before appending another command response, with final status 3.

## Commands

| Code | Operation | Fields after opcode | Result |
| ---: | --- | --- | --- |
| 1 | Create tensor | rank, requires_grad, dim0, dim1, dim2, dim3, data_offset | Tensor ID |
| 2 | Binary | operation, left, right | Tensor ID |
| 3 | Unary | operation, input | Tensor ID |
| 4 | Full reduction | operation, input | Tensor ID |
| 5 | Matrix multiply | left, right | Tensor ID |
| 6 | Transpose | input, axis0, axis1 | Tensor ID |
| 7 | Backward | scalar_output | 1 |
| 8 | Zero leaf gradient | input | 1 |
| 9 | SGD | parameter, learning_rate_bits | 1 |
| 10 | Read values | input | Element count; binary32 payload |
| 11 | Read leaf gradient | input | Element count; binary32 payload |
| 12 | Set leaf element | input, logical_index, value_bits | 1 |
| 13 | Metadata | input | Tensor ID; metadata payload |
| 14 | Scalar numerical operation | scalar_operation, left_bits, right_bits | Scalar result bits or integer boolean |
| 15 | Linear | input, weight, bias_or_zero | Tensor ID |
| 16 | Mean squared error | input, target | Tensor ID |
| 17 | Contiguous view | input, rank, dim0, dim1, dim2, dim3 | Tensor ID |
| 18 | Clone | input | Tensor ID |
| 19 | Detach | input | Tensor ID |
| 20 | Axis reduction | operation, input, axis, keepdim | Tensor ID |
| 21 | Softmax | operation, input, axis | Tensor ID |
| 22 | Conv2d | input, weight, stride_pack, padding_pack, dilation_pack, groups | Tensor ID |
| 23 | Pool2d | operation, input, kernel_pack, stride_pack, padding_pack, count_include_pad | Tensor ID |
| 24 | Adam/AdamW | parameter, first_moment, second_moment, step, config_offset | 1 |
| 25 | Momentum SGD | parameter, momentum_state, zero, step, config_offset | 1 |
| 26 | LayerNorm | input, weight_or_zero, bias_or_zero, epsilon_bits | Tensor ID |
| 27 | Probability-target cross entropy | logits, target | Tensor ID |
| 28 | BatchNorm evaluation | input, running_mean, running_variance, weight_or_zero, bias_or_zero, epsilon_bits | Tensor ID |
| 29 | Seeded backward | output, seed_or_zero | 1 |
| 30 | Index select | input, axis, index_count, index_payload_offset | Tensor ID |
| 31 | Gather | input, axis, index_config_offset | Tensor ID |
| 32 | General matrix multiply | left, right | Tensor ID |
| 33 | Reclaim training graph | retained_prefix_count | 1 |
| 34 | Concatenate | axis, tensor_list_config_offset | Tensor ID |
| 35 | Stack | axis, tensor_list_config_offset | Tensor ID |
| 36 | Scatter add | self, source, axis, index_config_offset | Tensor ID |
| 37 | Index add | self, source, axis, index_config_offset, alpha_bits | Tensor ID |

Binary operation IDs are 1 add, 2 sub, 3 mul, 4 div. Unary IDs are 5 ReLU,
6 neg, 11 sqrt, 12 exp, 13 log, 14 sigmoid, 15 tanh. Full reductions are 7 sum
and 8 mean; axis reductions are 18 sum and 19 mean. Softmax operations are
20 softmax and 21 log-softmax. Pooling operations are 23 average and 24 maximum.

Scalar numerical operation IDs are 0 add, 1 sub, 2 mul, 3 div, 4 less-than,
5 unsigned-integer conversion, 6 neg, 7 abs, 8 sqrt, 9 exp, 10 log,
11 sigmoid, 12 tanh. Operation 5 treats its first input as a full unsigned
64-bit integer; comparison returns integer 0 or 1.

Axis parameters accept negative axes encoded as two's-complement u64. Scalar
axis operations accept 0 and -1. View dimensions are explicit nonnegative
dimensions; inferred `-1` dimensions are not part of this interface.

Spatial packs use `(height << 32) | width`. Convolution is NCHW/OIHW, supports
positive groups, and omits an internal bias; bias can be composed through a
reshaped broadcast tensor. Pooling uses floor output sizes, unit dilation,
padding at most floor(kernel/2), and includes/excludes padding for average
pooling as selected. It does not return argmax indices to the caller.

## Tensor identity and graph composition

Before the first graph reset, IDs begin at 1 and follow allocation order.
After a reset, retained-prefix IDs stay unchanged and transient IDs carry a
generation, as specified below. Zero is an optional sentinel only in commands
documenting it. All other uses of zero as a tensor ID fail.
Most tensor-producing commands allocate one descriptor. Composite commands
also allocate their internal graph nodes:

| Command | Allocated descriptors |
| --- | ---: |
| Linear | 2 without bias; 3 with bias |
| MSE | 3 |
| LayerNorm | 8, plus 1 for weight and 1 for bias |
| Probability-target cross entropy | 5 |
| BatchNorm evaluation | 7, plus 2 for weight and 2 for bias |

The returned ID is the last allocated tensor. A command-stream producer must
account for these allocations and reset generations when referring to later tensors. The HIA library
functions in the source return IDs directly and can be called from an edited
native entry program. The fixture packets document concrete examples.

Metadata payloads contain 13 words: rank, numel, requires_grad, four dimensions,
four strides, storage-owner ID, and that owner's current version. Values are
read in logical row-major order even when a tensor uses transposed or
channels-last strides. Commands 34–37 each allocate one output descriptor.

All dimensions are checked against the per-dimension limit before computing
numel. If any dimension is zero, numel is zero; a large prefix before that zero
does not cause a false element-count refusal. Nonempty tensors still obey the
262,144-element limit. Ordinary contiguous strides are constructed from the
last dimension using `stride *= max(dimension, 1)`. Thus shape `[2,0,3]` has
strides `[3,3,1]`, and `[262144,262144,0,262144]` is a legal empty shape with
strides `[68719476736,262144,262144,1]`. An illegal dimension remains an error
even in an otherwise empty shape. At rank at most four, these empty stride
products are bounded by `2^54` and fit in the metadata words.

## Autograd and storage

Command 7 requires a gradient-tracked output with one element and seeds it with
1. Command 29 accepts a gradient-tracked output of any supported shape and an
explicit binary32 seed tensor with exactly the same rank and dimensions. Equal
element counts with different shapes fail with status 2. The seed is read in
logical order, so a transposed seed is accepted. Zero as the seed ID selects the
same implicit-unit rule as command 7. Empty outputs accept exactly shaped empty
seeds. A seed can require gradients or alias the output; it supplies cotangent
values without adding an edge to the graph.

The graph remains available for repeated backward calls; leaf gradients
accumulate until explicitly cleared. Only gradient-tracked leaf tensors support
command 11. This eligibility is checked even for empty tensors. ID, seed-shape,
tracking and saved-version checks precede gradient changes. There is no distinct
`grad=None`, forward-mode differentiation or higher-order graph.

Transpose and contiguous view share storage and the storage owner's version.
Detach shares storage and removes the autograd connection. Clone copies logical
values into new storage while preserving a differentiable connection. A leaf
write or optimizer update increments the shared owner's version. Backward
preflights the reachable graph, checking saved parent versions and newly owned
output versions before altering gradients. This validation is conservative. The packet payload-import path uses
`tset` for each element, so its owner counter includes those initialization
writes. Its absolute value is not qualified as PyTorch `Tensor._version`;
alias sharing, explicit mutation increments and stale-saved-value rejection
are the current contract. Absolute counter parity remains an open compatibility
item for the general tensor/storage interface.
For concatenation and stacking, every saved input occurrence is checked,
including repeated inputs, skipped `(0,)` tensors and inputs without gradients.
The indexed-addition checks below can also reject a source derivative during
preflight, even when its forward operation succeeded. Such a backward refusal
does not clear work buffers or change existing leaf gradients.

## Training graph reclamation

Command 33 preserves the initial `keep` descriptors and reclaims every later
descriptor and its arena storage. The first successful reset fixes this retained
prefix for the remainder of the request; later resets must use exactly the same
`keep`, including when it is zero. Allocate parameters, optimizer states and any
retained inputs before creating transient graph nodes. Each retained descriptor
must be an owned leaf: its graph operation is zero and its storage-owner ID is
itself. A detached alias, view or operation output cannot be retained in this
prefix. The legal range is `0 <= keep < next_slot`, at most 255.

Retained values, leaf gradients, work buffers, versions and optimizer states are
preserved. Reset does not clear gradients. A typical iteration explicitly zeros
parameter gradients, builds the forward graph, runs backward, updates parameters,
and then resets to the same retained prefix. Read any transient output before
resetting. Once discarded, its ID is invalid even if a later tensor reuses the
same slot.

The workspace generation starts at zero and increments after every successful
reset. Slots are in 1..255. Retained slots use their original IDs, while a
transient slot uses `(generation << 8) | slot`. Consumers must preserve the full
64-bit ID. A slot with the wrong generation, a generated ID for a retained slot,
or slot zero returns status 1. The maximum generation is `2^56 - 1`; a further
reset returns status 3 before modifying any graph state. Resetting with all 255
slots retained is legal, but leaves no slot available for new tensors.

Invalid prefix ranges or changes return status 7. Retaining a nonleaf or shared
owner returns status 6. These preflight checks precede reclamation; on refusal,
the workspace is unchanged except for its sticky error status. The production
protocol still stops at the first failed command.

The 255-descriptor limit now applies to tensors live between resets. Input,
output and workspace limits remain 1 MiB each, and a request can still exhaust
command or response capacity. This mechanism reuses memory within one request;
the executable still initializes a fresh workspace for each invocation.

## Modules and optimizers

Linear accepts rank 1–4 input `[..., in_features]` and rank-two
`[out_features, in_features]` weights, then computes `input @ weight.T + bias`.
An optional bias must be a vector of `out_features` elements. LayerNorm normalizes the last dimension;
affine weights and biases must be vectors of that dimension. BatchNorm evaluation
accepts rank 2–4 with channels at dimension 1 and frozen, non-gradient running
mean/variance vectors. Probability-target cross entropy accepts rank-two logits
and equally shaped probability targets, and computes a batch mean. Index labels,
ignore_index, label smoothing and class weights are separate unimplemented API
variants.

An Adam configuration is six u64 words: learning_rate_bits, beta1_bits,
beta2_bits, epsilon_bits, weight_decay_bits, decoupled_weight_decay. The last
word is 0 for Adam and 1 for AdamW. A momentum configuration is five words:
learning_rate_bits, momentum_bits, dampening_bits, weight_decay_bits, nesterov.

Optimizer states must be distinct, owned, non-gradient leaf tensors matching
the parameter shape. Hyperparameters are checked before any parameter/state
write. Steps range from 1 through 2^32-1 and are supplied by the caller;
monotonic step succession is not maintained internally. Momentum initializes
its buffer from the first gradient, with dampening only on later steps.
Nesterov requires positive momentum and zero dampening. Gradients are not
automatically cleared. Because `grad=None` is not represented, a zero gradient
buffer does not suppress an explicitly requested weight-decay update.

## General matrix multiplication

Command 32 accepts rank 1–4 inputs. Two vectors produce a scalar dot product;
matrix-vector and vector-matrix products omit the temporary singleton matrix
dimension. Higher ranks broadcast the leading batch dimensions and contract the
last left dimension with the second-last right dimension. Scalar inputs and
incompatible contraction/batch dimensions fail with status 2. Empty batches,
zero output dimensions and zero contraction dimensions are supported. Strided
inputs produced by transpose are read in their logical order.

Backward accumulates both operands' derivatives, including shared operands and
reductions across broadcast batch axes. Command 5 retains its rank-two contract
and uses the same matrix kernel. Every product checks the 16,777,216-term budget
before allocating its result: `output_numel * contraction_size`. The check uses
division to avoid overflow. Zero contraction size has zero work and produces
zeros subject to the ordinary result-capacity limits.

## Integer indexing payloads

Index values are exact signed 64-bit little-endian integers in the request
payload. They are not binary32 tensor elements. The runtime still has only
binary32 tensors; this interface does not add a general integer tensor dtype.
All payload offsets must be eight-byte aligned, at or after the command records,
and fully contained in the request. Admitted indices are copied into private
workspace storage and retained for the reverse scatter-add.

Command 30 accepts rank 1–4 input and a one-dimensional index list of
`index_count` integers at `index_payload_offset`. Its output replaces the chosen
axis length by that count. Negative axes are normalized; negative or out-of-range
indices fail with status 7. Empty lists and empty dimensions are supported.
Rank-zero input is outside this command's contract and fails with status 2.

Command 31 reads a configuration at `index_config_offset`: a u64 rank, that many
u64 dimensions, then the product of those dimensions in signed-i64 index values.
For non-scalar tensors, index rank must equal input rank. Nonempty results
require every non-gather dimension to be no larger than the corresponding input
dimension; no broadcasting occurs. A rank-zero input accepts a rank-zero index
of value zero and axis 0 or -1. Empty index results retain the empty-index fast
path, including non-gather dimensions larger than the input. Rank, axis,
dimension capacity and payload framing are still checked.

Both operations produce new storage. Repeated indices add their gradients to
the same input location, and transposed inputs propagate through their original
view graph. Mutating a saved source or output still fails the backward version
preflight.

## Concatenation and stacking

Commands 34 and 35 read a tensor-list configuration: one u64 input count,
followed by exactly that many full u64 tensor IDs. The count and the complete
list must fit the request. A zero count returns status 7. The list may repeat
an ID and must retain the generation bits used after command 33. Invalid or
discarded IDs return status 1; malformed alignment, offsets or truncated lists
return status 100. The saved-list and result-storage requirements are checked
together before allocation.

Concatenation accepts rank 1–4 tensors of equal rank and equal dimensions away
from the selected axis, adding their extents along that axis. A tensor whose
exact shape is `(0,)` is skipped for rank/shape and axis-length matching. Other
empty shapes still participate in those checks. If every input has shape
`(0,)`, the result has shape `(0,)` and any signed 64-bit axis is accepted,
including `INT64_MIN` and `INT64_MAX`. Otherwise, the axis is normalized against
the first non-skipped tensor. Scalar inputs are rejected with status 2.

Stacking requires exactly equal input shapes and inserts the input count as a
new dimension. It accepts input ranks 0–3, so its output remains within rank
four. The axis is normalized against `input_rank + 1`; scalar stacking accepts
0 or -1. Stack does not apply concatenation's `(0,)` skip rule. Both commands
check every output dimension and use the zero-aware numel rule above.

Input values are read through their actual strides. Cat selects channels-last
output strides for rank-four results only when every input suggests that
format under the pinned PyTorch stride-selection rule. Singleton dimensions
can make this distinction depend on exact strides even when both layouts are
contiguous. A skipped `(0,)` input or an input whose suggested format is ordinary
contiguous makes the mixed output ordinary contiguous. Stack's
last-axis insertion for rank-three inputs applies that same rule to the virtual
unsqueezed inputs; other supported stack paths produce ordinary contiguous
output. These strides describe the actual output storage: forward writes use
physical tensor addresses. They are not metadata-only changes, and they do not
add a general memory-format conversion command.

Each occurrence contributes its slice or selected cotangent back to its input.
Repeated IDs and shared graph ancestors accumulate through the existing work
buffers. Backward buffers use logical element order independently of output
layout. The private saved-parent list is reclaimed with the operation during
graph reset; it cannot keep a discarded generation alive. In-place and `out=`
variants, dtype promotion and higher-order differentiation remain outside this
protocol's scope.

## Scatter add and index add

Commands 36 and 37 take binary32 `self` and `source` tensors and return new
owned storage with self's shape. Both inputs remain available. Their index
configuration uses the same framing as command 31: a u64 rank, that many u64
dimensions, and the resulting number of signed-i64 index values in logical
row-major order. Rank zero contains one index. No float-to-index conversion or
integer tensor descriptor is introduced. Saved indices are copied into the
graph's arena and reclaimed with that graph.

### Scatter add

Command 36 accepts index ranks 0–4. For nonempty indices, self, source and index
have equal effective ranks; scalars have effective rank one and size one.
Every index dimension must fit the corresponding source dimension, and every
nonselected index dimension must fit self. Negative indices and indices at or
above self's selected extent return status 7. Empty indices skip these shape
checks after axis, local rank, dimension-capacity and payload checks. A rank
above four is still outside this local profile.

The output begins as a copy of self, then adds source values at the selected
index coordinates. Repeated destinations accumulate. The self derivative is
the incoming cotangent. The source derivative gathers from that cotangent and,
where permitted, reduces to the original source shape.

Consequently, some source/index shape pairs are valid for forward but not for
source backward. If source requires gradients, its rank must not exceed the
index rank, and each right-aligned source dimension must equal the index
dimension or be one. An incompatible returned source derivative produces
status 5 during graph preflight. If source does not require gradients, self's
identity derivative remains available. This includes empty inputs and the
observed scalar/vector distinction; forward success does not promise that an
incompatible source derivative will be accepted.

### Index add

Command 37 accepts an index configuration of rank zero or one. Self and source
must have equal literal ranks, equal dimensions away from the selected axis,
and an index count equal to source's selected extent. Scalar/scalar inputs
accept a scalar or length-one vector index; a scalar/non-scalar pair is
rejected. Empty indices do not bypass these shape checks.

The `alpha_bits` field must contain a binary32 encoding with its upper 32 bits
zero; nonzero upper bits return status 7. To request the default mathematical
coefficient one, encode `0x3f800000` explicitly. Zero, signed zero, infinities
and NaNs are admitted coefficients. The forward update adds
`alpha * source` at each selected destination. The pinned CPU profile uses
software fused multiply-add for rank above one and separately rounded
multiply/add for rank zero or one, as qualified in [Numerics](numerics.md).

For an empty multidimensional self, the pinned CPU forward path returns before
checking index values. This operation may therefore succeed with saved
out-of-range indices. If source requires gradients, its subsequent index-select
derivative checks those values, and the HIA graph preflight returns status 7
before gradient changes. If only self requires gradients, its empty identity
derivative succeeds. Nonempty updates reject invalid indices before execution.

Both operations save the full self/source IDs and owner versions. Shared
inputs receive both identity and source contributions, and repeated backward
calls accumulate them. A mutation still triggers the existing saved-version
refusal. The indexed-addition work check is
`self.numel + max(self.rank, 1) * actual_updates <= 16,777,216`; scatter uses
index count for updates, and index_add uses source numel unless its empty-self
path skips the updates. Checked division precedes multiplication. The current
tensor and index limits already impose a smaller attainable workload.

## Limits and errors

| Status | Meaning |
| ---: | --- |
| 0 | Success |
| 1 | Invalid tensor ID |
| 2 | Shape, rank, axis or view mismatch |
| 3 | Capacity or supported work-budget exhaustion |
| 4 | Unsupported operation |
| 5 | Gradient or optimizer-state contract mismatch |
| 6 | Invalid mutation or stale saved version |
| 7 | Invalid value or operation argument |
| 100 | Malformed packet, size or payload offset |

The native input/output/workspace buffers are each 1 MiB. There are at most 255
tensor descriptors and rank is at most four. Each dimension and descriptor's
numel is bounded by 262,144; actual available storage is smaller because the
53,248-byte metadata region and all data/gradient/work arrays share the workspace.
Convolution, matrix multiplication and pooling each admit at most 16,777,216
kernel terms per operation. For pooling the conservative term count is
`output_numel * kernel_height * kernel_width`, including padded positions. The
forward admission bound also bounds that node's reverse work. These are
per-operation bounds; they do not impose a wall-clock deadline on an entire
multi-command request.

The outer native ELF wrapper rejects input larger than 1 MiB with process exit
3 and no response. Unsupported command-line arguments exit 64. OS-level I/O or
allocation failure exits 4. Those process-level failures are distinct from the
structured statuses above. No background service, network listener, GPU context,
or distributed process group is created by this executable.

## Optional interactive framing

The ordinary executable still consumes one `HTORCH01` packet. The optional
`heptatorch-session` process accepts a sequence of `HTOSES01` frames and emits a
`HTOSRP01` response before reading the next frame, without requiring stdin EOF.
The complete byte-level contract, canonical failed response, atomic state
commit, EOF behavior and native error exits are specified in
[tools/session/CONTRACT.md](../tools/session/CONTRACT.md).

This entry keeps state across requests in one process. It uses two one-MiB
arenas and can commit or discard the complete packet, including graph-reset
generations, gradients and optimizer states. It has no owner outside those
arenas and does not establish general storage allocation or persistence after
process exit. The same numerical kernels serve both entry modes.

## Separate typed int64 owner profile

The successor eleven-module profile uses `HTINT002` / `HTOINT02` private state
tags and 640-byte descriptors. Its exact wire contract is separately versioned
in [the int64 protocol](../tools/integer_tensors/PROTOCOL.md). Commands97/98/100
provide qualified dtype2 construction, raw reads and fill. Command103 performs
typed binary arithmetic,104 its admitted int64 Tensor-RHS in-place variants,
105 explicit late-error int64 dropout, and106 actual Scalar add_ with alpha1.

Command105 commits the observed completed prefix and RNG advancement before
its runtime error, then stops the suffix. The ordinary full-frame rollback
contract remains separate. Public Python exception classes/messages, arbitrary
dtype/mode combinations and mixed F32/int64 autograd remain unimplemented.
This profile does not change the earlier canonical binary32 packet contract.
