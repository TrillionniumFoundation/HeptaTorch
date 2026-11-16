# Native CPU Bernoulli and dropout, profile 03

This is a **partial** distribution component over the frozen seven-kind Tensor/Generator runtime. The new ELF executes all numerical work, Generator transitions, storage writes and automatic differentiation in native HIA. Python authors HIA, invokes the native compiler/linker, or observes qualification runs; it is not a production numerical runtime.

The frozen production source is `a72a98232c7b7f10efa67ce29b21e417c58201383bef9aacc266fe247ff356ce`, and the native ELF is `515790a191b3db1ecbfcb3f2afab644ef980b51ad85c5d6daf1b8c5a9a5a6e28`. Its 294 core functions, 3,718 blocks and 14,091 instructions are partitioned into seven real native units including the unchanged RNG and owner units. The pinned ABI 3 compiler admits every unit; the native linker checks actual interfaces, relocations, call graph and stack budget. No compiler capacity was increased.

## What is implemented

- In-place float32 Bernoulli with an exact binary64 scalar probability. Every nonempty element consumes two MT19937 words, including probabilities 0 and 1. Signed zero is accepted; invalid ranges, infinities and NaNs are rejected.
- In-place float32 Bernoulli with float32 Tensor probabilities, broadcasting and actual TensorIterator iteration order. The output operand is considered before the probability operand. Equal-stride ambiguities proceed to the next operand. Nonzero overlapping strides and legal probability aliases retain per-element read/write feedback.
- Functional float32 Bernoulli with a real graph node and the exact zero derivative. Negative zero, NaN and infinite incoming gradients produce positive zero; the derivative does not read saved probability values.
- Ordinary CPU float32 dropout, including identity for probability zero, evaluation mode and empty tensors; the probability-one multiply path; the native double complement, float32 scaling, dense noise layout and multiply path for intermediate probabilities.
- A real Saved noise tensor for differentiable functional dropout, sharing Storage and Version through the explicit saved-noise accessor. Retain/default backward, input mutation, mask Version rejection, released Saved state, nullable gradients and the existing partial backward entry remain usable.

The profile keeps the existing explicit-Generator wire convention. Public PyTorch dropout uses its default Generator; the reference probe binds that Generator's complete state to the corresponding explicit native Generator. In-place operations on tracked tensors are qualified under the explicit `no_grad` flag. Full normal-gradient-mode in-place graph rewriting remains outside this profile.

## Actual reference evidence and open differences

The unchanged probe `pytorch_bernoulli_dropout_probe.py` has SHA `db00efdf0cca14b3b3f25545e8a59a27308ca128d861e04086c18bd1edb6d3a4`. Its actual pinned-wheel report contains 183 observations, no setup errors and all 11 provenance checks. All 31 unique, complete 5,056-byte Generator states have their own length and hash checks. The original report is included unchanged as `evidence/torch-bernoulli-dropout.json` with SHA `c624f5c4762b1c71e7053aba9af55aaa3d8bbaae5aa100b0a6df5f44f58da467`.

The native replay reports **161 PASS, 4 OPEN_TRANSACTION_DIFFERENCE and 18 OPEN_UNREPRESENTED**, with no unexplained failures. It executes 1,894 live frames, compares 49,069 float32 words exactly, compares 182 complete Generator states, and observes 10,861,632 bytes of whole-owner rollback. This is a partial qualification, not an assertion that all 183 are equivalent.

The four transaction differences are preserved explicitly. Three Tensor-probability cases encounter an invalid value after two successful elements: PyTorch has already written/advanced its Generator and has not incremented the Tensor Version. A tracked-leaf in-place dropout error occurs after the mask consumes eight MT words. This profile's existing application-frame contract rolls back the complete owned state on these errors. The four immutable observations remain open and are not reclassified as passes. A separate successor can expose explicit partial-commit commands without silently changing commands 80–85.

The 18 unrepresented observations consist of eight double-probability Tensor cases, two integer-probability Tensor cases, and eight `native_dropout` contrasts. A double scalar probability is not a double Tensor dtype. Ordinary dropout and `native_dropout` have different fast paths, mask dtype and scaling order; this profile does not fabricate a bool mask using a float payload.

Other explicit limits remain: CPU float32 tensors; rank at most four; nonnegative strides; finite bounded object tables/pool; no default-Generator singleton API; no device generators, normal/distribution suite, feature/alpha dropout, complex dtype, higher-order differentiation or full PyTorch completion. All full-RNG, full-dtype and full-rewrite completion flags remain false.

## Independent native controls

`test_probability_math.py` links a qualification-only borrowed entry to the **unchanged production objects** and exercises 106,488 probability bit patterns. An independent arbitrary-precision integer checker verifies all 425,952 returned words: validity, correctly rounded binary64 `1-p`, the rounded binary32 divisor and the binary32 reciprocal. It includes binade boundaries, midpoint neighborhoods, all exponent ranges, subnormals, signed zero, NaNs, infinities and deterministic random values. No production opcode or helper is replaced.

`test_controls.py` executes 98 live frames across four groups. It verifies reserved fields, bad/stale typed handles, unimplemented reserved opcodes, exact one-million-element execution, maximum-plus-one refusal, two million MT32 consumptions at scalar probability zero, empty no-consumption semantics, actual IEEE multiply behavior, and composition with commands 57, 58 and 70. It compares 1,000,023 values and 44,193,488 rollback bytes. The existing command 58 commits a successful random prefix and stops its suffix on a backward runtime error; command 57 still rolls back both Tensor and Generator.

`test_entry.py` executes 86 owned-entry/output/real-allocation-refusal controls, including all output capacities 0–31, full-owner preservation, Version overflow, actual Linux `RLIMIT_AS` allocation failure and a response-overflow failure after prior random work. Faulted byte-state is confined to the private qualification harness, which creates a genuine owned buffer; it is not a production raw-pointer or arbitrary-owned-state import API.

`test_handoffs.py` performs real owned moves between the frozen nullable RNG predecessor and this profile. Both cross-profile directions reject status 7, retain the exact owner pointer/length/bytes, allow the original profile to resume, and close/unmap the owner. The new-profile state contains an actual Bernoulli graph and cleared nullable leaf. Symbol-prefix renaming is invertible to each original HIA source byte sequence; it does not substitute external bodies.

## Authoring capacity boundary

The first aggregate authoring attempt exceeded the pinned host Core queued-node budget. That failure is retained. The new adapter uses the same per-function scalar-parameter pruning and forwarding-substitution rules, bounded-decodes each actual function, and checks its body against interfaces derived from the actual complete source functions before and after transformation. It introduces no synthetic callee body.

The old entry, `gradient_direct` and `partial_apply_node` are independently re-emitted with both the ordinary aggregate compactor and the bounded compactor and required to match the frozen predecessor HIA bodies byte for byte. The per-function check is **not** an aggregate Core-program qualification. All actual emitted production units still undergo the unchanged native compiler, and the linked application undergoes the unchanged ABI 3 linker and stack admission. `body-compaction.json` and the unit/source/body maps preserve this distinction.

## Reproduction

Required completed predecessors are the frozen `tensor_rng_nullable` build, the common Tensor/RNG build's `overlay-authoring.py`, the gradient-materialization build's `overlay-authoring.py`, the pinned external Hepta authoring checkout, and the frozen ABI 3 stack build. The upstream compiler is not vendored here. Exact input hashes are checked before and after authoring; all 32 actually imported upstream modules must belong to the verified pinned source closure.

```sh
python -B tools/cpu_distributions/rebuild.py \
  --baseline build/tensor-rng-nullable \
  --common-authoring build/tensor-rng-common/native/overlay-authoring.py \
  --gradient-authoring build/gradient-materialization/overlay-authoring.py \
  --upstream /absolute/pinned-Hepta-checkout \
  --stack-build build/stack-budget \
  --output build/cpu-distributions

python -B tools/cpu_distributions/qualify.py \
  --build build/cpu-distributions \
  --baseline build/tensor-rng-nullable \
  --stack-build build/stack-budget \
  --output build/cpu-distributions-qualification
```

The rebuild asserts every HIA unit, native object, complete source and ELF identity in `profile.json`. Qualification returns success only when the scoped native checks pass and the unchanged actual report still produces precisely the declared PASS/open partition; its overall compatibility status stays PARTIAL.

The evidence archive contains the original production HIA/objects/ELF, qualification-only images and requests, complete response traces, initial rejected attempts and replay mapping corrections. Its manifest is checked by reading every archived member and comparing length and SHA. No raw unique observation was discarded.

## Source attribution

`primary-sources.json` identifies exact official PyTorch source at commit `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`. The default CPU scalar Bernoulli implementation uses the generic MT template unless both MKL and `FBCODE_CAFFE2` are enabled; `USE_MKL=ON` alone does not select VSL. The actual wheel's complete-state observations independently confirm the generic consumption path. Its API behavior is source-bound and observation-bound, not inferred only from that compile-time condition.

The 17 RNG engine functions and the owned entry module are reused byte for byte. `THIRD_PARTY_NOTICES.txt` retains the predecessor's original MT19937 notice. New mathematical/loop implementations do not call PyTorch, NumPy, MKL, libc numerical routines or a host tensor interpreter.
