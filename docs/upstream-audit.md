# PyTorch upstream audit and completion criteria

HeptaTorch implements a native HIA tensor and training runtime. The compatibility
target is the pinned PyTorch release below. Completion is tracked against that
release's behavior, including runtime services and backend semantics, as well as
its operator declarations.

## Pinned baseline

| Field | Value |
| --- | --- |
| Upstream | `pytorch/pytorch` |
| Stable release | `v2.15.0` |
| Release date | 2026-10-28 |
| Commit | `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457` |
| Audit date | 2026-11-15 |

The GitHub release API and tag reference were read independently. The tag points
directly to the commit above. The release is neither a draft nor a prerelease.
The lock file preserves the source blob and subtree identities used for the
inventory. Future work must update that lock deliberately instead of silently
comparing against a moving `main` branch.

- [Release](https://github.com/pytorch/pytorch/releases/tag/v2.15.0)
- [Pinned source](https://github.com/pytorch/pytorch/tree/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457)
- [Local upstream lock](../compatibility/upstream-lock.json)
- [Complete native schema inventory](../compatibility/upstream-schemas.json)

## Measured inventory

| Source | Count | Meaning |
| --- | ---: | --- |
| `aten/src/ATen/native/native_functions.yaml` | 2,590 | Exact lines beginning `- func:` |
| Distinct names before `(` | 2,590 | Includes overload suffixes |
| Distinct names before `.` | 1,558 | Base names, not a public API count |
| `tools/autograd/derivatives.yaml` | 688 | Exact lines beginning `- name:` |
| `torch/` | 4,761 | Git blobs |
| `aten/` | 2,757 | Git blobs |
| `c10/` | 441 | Git blobs |
| `test/` | 11,920 | Git blobs, including fixtures |
| `torchgen/` | 89 | Git blobs |
| `functorch/` | 83 | Git blobs; excludes `torch/_functorch` |
| `caffe2/` | 49 | Git blobs |
| `third_party/` | 225 | Git blobs, plus 35 submodule gitlinks |

Every queried subtree returned `truncated=false`. These counts do not include
the contents of submodule gitlinks. They do not measure implementation effort,
executed tests, public API coverage, or equivalent behavior.

The schema inventory contains every one of the 2,590 exact declaration strings.
Its initial `unimplemented` status means that the complete upstream schema has
not been qualified. A restricted native implementation can be useful and tested
without covering all of that declaration's dtype, device, layout, shape, aliasing,
mutation, and differentiation behavior. Scoped capability evidence is tracked
separately. The full-schema status must not be promoted by counting a function
name or by passing only a contiguous-vector example.

The denominator also excludes generated schemas, custom and other namespaces,
higher-level modules, Python and C++ frontend behavior, compiler facilities,
distributed services, serialization, and backend-specific kernels. Therefore
even qualifying all 2,590 records would not by itself complete the whole project.

Sources:
[native declarations](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/aten/src/ATen/native/native_functions.yaml),
[derivative declarations](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/tools/autograd/derivatives.yaml).

## Native implementation boundary

The initial executable target is an HIA CPU runtime using binary32 values,
software floating-point primitives, tensors of rank at most four, reverse-mode
automatic differentiation, and SGD. The current HIA toolchain's lack of native
floating-point instructions makes numerical primitive qualification part of the
runtime work. This target establishes an executable training path and leaves
the complete upstream target in the inventory.

HIA sources must perform the actual tensor operations, gradient propagation,
and parameter updates. A host PyTorch call, a precomputed answer, or an unchecked
constant success result is not an implementation. Upstream tools may be used
outside the runtime to produce independent reference data. Compiler bootstrap
requirements and test-oracle dependencies must be stated independently from
runtime dependencies.

For every supported operation, evidence must identify:

- The HIA entry point and tested source revision.
- Device, dtype, rank, layout, shape and capacity constraints.
- Forward, backward, aliasing and mutation behavior.
- Unsupported inputs and the resulting explicit error.
- Reference input and output values, tolerances, and how the reference was made.
- The command and observed result of the native executable.

A native HIA API and unchanged execution of existing `import torch` programs are
different compatibility surfaces. Full frontend compatibility also requires a
specified language boundary and the corresponding object, error, override, and
extension protocols.

## Semantic requirements for the CPU training path

### Tensor storage and shape

A tensor descriptor must distinguish logical shape from physical strides and
storage offset. Multiplication when computing sizes and address ranges must be
checked before allocation or access. Zero-element tensors and scalar rank-zero
tensors need separate test cases. Invalid dimensions, incompatible shapes,
capacity exhaustion, and out-of-bounds views must return an error before data
access.

Transpose and view operations share underlying storage in PyTorch. A copied
transpose can implement the numerical transformation but cannot qualify aliasing
semantics. `view` requires compatible strides; `reshape` may copy when required.
Basic indexing returns a view and advanced indexing returns a copy. Test writes
through a view against its base to verify sharing.

Source: [Tensor views](https://docs.pytorch.org/docs/2.15/tensor_view.html).

### Broadcasting and reductions

Broadcast trailing dimensions, including missing dimensions and singleton
dimensions. Test `[2, 1, 3]` with `[1, 4, 1]`, rank-zero scalar inputs, incompatible
shapes, and empty axes. In particular, a size-zero dimension broadcast with a
singleton remains size zero; a plain maximum of the two dimension sizes is
incorrect for this case. In-place operations must not grow the target's shape.

Backward must sum contributions over every expanded or missing input axis.
Reduction tests must cover empty input, explicit axes, negative axes, duplicate
axes, keepdim, NaNs, and dtype-dependent behavior. An empty sum and an empty mean
have different results. A full upstream schema also includes dtype promotion;
the binary32 slice must state its narrower dtype contract.

Source: [Broadcasting semantics](https://docs.pytorch.org/docs/2.15/notes/broadcasting.html).

### Software binary32 arithmetic

Qualify the actual software arithmetic independently before relying on training
loss as evidence. Include positive and negative zero, both signs, subnormals,
minimum normal values, overflow, underflow, infinities, NaNs, cancellation,
division by zero, halfway rounding, and guard/round/sticky-bit cases. Record the
rounding mode and distinguish numerical equality from exact bit-pattern
requirements. NaN payload propagation needs a stated contract if it is tested.

Floating-point reductions and matrix products can have different rounding from
other valid implementations because operation order affects the result. The
test oracle should compare elementwise primitives at the appropriate bit or
numerical level and use justified tolerances for composite calculations.

Source: [Numerical accuracy](https://docs.pytorch.org/docs/2.15/notes/numerical_accuracy.html).

### Automatic differentiation

The runtime must record actual executed operations and propagate gradients in
reverse dependency order. Multiple paths to the same input must accumulate:
`x * x + x` is a small decisive case. Broadcasting requires reduction to each
input's original shape. Matrix multiplication requires both input derivatives.
Test ReLU's selected derivative at zero independently of smooth finite-difference
tests.

Saved values must remain valid until their backward operations run. A mutation
policy must either retain snapshots or detect invalidated saved tensors.
Upstream uses version counters and rejects changed saved values when necessary.
Gradient accumulation across calls, graph retention, no-grad mode, detach, and
higher derivatives are separate behaviors and must be accounted for separately.

Source: [Autograd mechanics](https://docs.pytorch.org/docs/2.15/notes/autograd.html).

### Optimization and training

An end-to-end test must execute forward computation, scalar loss, backward,
gradient clearing, and a parameter update. Use deterministic affine regression
and a small network with at least one nonlinear operation. Assert gradients and
updated parameters against independent values, then recompute the new loss.
Falling loss alone does not establish correct gradients.

Vanilla SGD can close independently of momentum variants. If momentum is added,
PyTorch initializes the buffer to the first gradient; dampening applies starting
with later steps. Nesterov, weight decay, maximize, skipped gradients, parameter
groups and optimizer state restoration require distinct qualification.

Source: [Pinned SGD](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/optim/sgd.py).

## Whole-project completion matrix

| Component | Required implementation and evidence |
| --- | --- |
| Runtime and dispatcher | Storage ownership, allocators, dtype promotion, schemas, dispatch, meta execution, errors and extension registration |
| Tensor operations | All declared operations and generated variants across supported shapes, dtypes, devices and layouts |
| Views and mutation | Strides, offsets, aliasing, overlap rules, in-place operations and saved-value correctness |
| Sparse, nested and quantized tensors | Representations, invariant checks, operators, conversions and differentiation |
| Autodiff and transforms | Reverse and forward mode, higher derivatives, hooks, custom functions, checkpointing and batching transforms |
| Neural networks | Parameters and buffers, recursive module state, train/eval semantics, convolution, pooling, normalization, attention and recurrent layers |
| Optimizers | Parameter groups, optimizer state, adaptive and momentum methods, schedules and checkpoints |
| Accelerators | Device discovery, transfer, allocation, streams/events, execution, numerical modes, and CUDA/ROCm/MPS/XPU kernels |
| Distributed systems | Rendezvous, process groups, collectives, asynchronous work, timeouts, failures, DDP, FSDP, DTensor and distributed checkpoints |
| Compiler and export | Graph capture, symbolic shapes, decompositions, transformations, lowering, fusion, AOT execution and export formats |
| Data and reproducibility | Data loading, worker lifecycle, RNG state, deterministic modes and checkpoint resumption |
| Ecosystem and packaging | Serialization, profiling, frontend APIs, extension protocols, supported platforms and reproducible builds |

The completion matrix is an engineering work inventory. No row is marked
complete merely because the repository contains a corresponding directory or
symbol. In particular, CPU results cannot establish GPU or distributed behavior.

## Upstream test sources

The pinned upstream files below are semantic references for independent tests:

- `test/test_binary_ufuncs.py`, `test/test_unary_ufuncs.py`
- `test/test_reductions.py`, `test/test_tensor_creation_ops.py`
- `test/test_view_ops.py`, `test/test_shape_ops.py`, `test/test_indexing.py`
- `test/test_autograd.py`, `test/test_ops.py`
- `test/test_nn.py`, `test/test_optim.py`, `test/test_serialization.py`
- `torch/testing/_internal/common_methods_invocations.py`
- `torch/testing/_internal/opinfo/definitions/`

Link: [Pinned test tree](https://github.com/pytorch/pytorch/tree/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/test).

Tests must record which of these behaviors were reproduced. An exact arithmetic
fixture, finite-difference check, independently implemented reference, and live
PyTorch differential run are different evidence types. They must not be reported
interchangeably. The initial audit environment's default Python could not import
`torch`; reference data must state its actual provenance.

## Attribution and repository cleanup

The extracted native schema strings are upstream reference material. The pinned
PyTorch `LICENSE` and `NOTICE` are preserved under `third_party/pytorch/` with
their original contents. This directory supplies attribution for reference data;
it does not install or embed the upstream runtime or all dependencies named in
the original notice.

Keep one implementation path for each supported behavior. Remove generated
build products, temporary probes, abandoned implementations and duplicate
reports from the tracked source tree after their replacement is verified.
Retain source, reproducible build and test entry points, required notices,
the pinned compatibility inventory, and evidence necessary to reproduce the
reported results. Missing features stay visible in the inventory rather than
being removed from the target during cleanup.
