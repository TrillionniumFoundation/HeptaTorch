# Native I64 and public F64 negation

## Contract

Command3, operation6 accepts no-grad I64 (dtype2) and public F64 (dtype4) tensors in the existing admitted rank0..4 construction and layout contracts. It returns a fresh independently owned Tensor/Storage/Version with the input dtype and shape. It never retains the source through an AD, Saved, view or owner edge. Other typed unary operations retain their existing refusals.

HIA performs I64 wrapping subtraction from zero; the INT64_MIN result retains its64-bit encoding. F64 flips only bit63, preserving all exponent/fraction bits, including signaling and quiet NaN payloads. The output layout uses the existing native typed iterator/allocation contract. No Python, Torch, NumPy, libc or libm executes application arithmetic.

F64 direct as_strided and F64 AD remain separate unimplemented contracts. Reachable non-contiguous F64 layouts in this suite are created by the existing F32 view-to-F64 conversion path; literal F64 payloads are then assigned without changing that geometry.

## Source and build

The negation increment was introduced in767 functions and15 ordered compilation units. In that increment, only predecessor `unary` and `command_argument` change;763 original functions remain byte-identical and two native helpers are appended. [wide_neg_source_transition.py](../tools/current_runtime/wide_neg_source_transition.py) restores the complete765 source, verifies every old function and exact forward round trip, and composes with all existing source inverses. The original767 identities and observed results remain in [its source-bound review](../validation/current-runtime767/native-review.json). [The current audit](current-audit.md) gives the current successor source, ELF and manifest hashes.

Every application object is freshly compiled by the pinned HIA ABI3 compiler and linked by the pinned native entropy linker. The unchanged build admission rejects an unpinned or different manifest even when a preliminary build happens to produce the same ELF.

## Independent reference and actual physical evidence

The405 literal recipes in [cases.py](../tests/wide_neg/cases.py) have SHA-256 `22063936e50a4e5da53be543fecc446a0fef529712da140a9afdb863b69da73a`. Expected results come from the actual pinned Torch process. The observed2.15.0+cpu build, wheel SHA, interpreter, pip report, NumPy2.3.5, requested AVX2 path and single thread are bound with all source/input/output files and both stream EOFs.

Coverage includes dense lengths0..70, signed zeros, infinities, all2048 F64 exponents with selected fractions and both signs, every individual NaN payload bit, full-width integer boundaries, deterministic literal hash words, all24 rank-four dense permutations, zero/overlapping strides, offsets, empty dimensions and independent ownership. The suite checks output bytes and metadata, input immutability, release independence, RNG, real process mappings,46 protocol controls and two real allocation failures followed by the identical successful request.

The actual405 run and independent replay passed10,484 frames,37,314 assertions,287,275 exact result elements,2,959 owner snapshots and four mapping reads. The original F6448 suite also passed on the same767 image with the same actual405 reference:2,867 frames and15,421 assertions. Its complete87-request entry recipe is unchanged:86 refusals and one F64-negation numeric successor. The original48 sub-sessions, both OOM/recovery pairs and original exact/non-NaN plus NaN-classification predicates remain.

The current778 successor requires a new complete31-group/14,814-case run. The405-case negation group must share the same current ReLU776 reference as Half negation371 and ReLU776; the two original wide ReLU control requests now require exact numeric successors while all other predicates remain. The original767 focused successes retain their original identities. See [the current qualification plan](../validation/current-runtime778/qualification-plan.json) and [ReLU contract](native-typed-relu.md).
