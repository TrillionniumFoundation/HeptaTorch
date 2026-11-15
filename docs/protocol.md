# Native execution protocol

The HIA executable consumes one request from stdin, executes its commands in
order, and writes one response to stdout. Every integer field is an unsigned
64-bit little-endian word. A floating-point field carries the binary32 bits in
its low 32 bits. Tensor payload elements are four-byte little-endian binary32.

This protocol is the native execution and qualification interface. It is not a
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
aligned, and at or after the end of the command records. Optimizer configuration
offsets are similarly bounded and eight-byte aligned.

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

IDs begin at 1 and follow allocation order. Zero is an optional-affine sentinel
only in commands documenting it. All other uses of zero as a tensor ID fail.
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
account for these allocations when referring to later tensors. The HIA library
functions in the source return IDs directly and can be called from an edited
native entry program. The fixture packets document concrete examples.

Metadata payloads contain 13 words: rank, numel, requires_grad, four dimensions,
four strides, storage-owner ID, and that owner's current version. Values are
read in logical row-major order even when a tensor uses transposed strides.

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
output versions before altering gradients. This validation is conservative.

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
