# One native runtime for Tensor RNG and nullable gradients

This successor combines all of the frozen common Tensor/Generator runtime with
the complete nullable-gradient command70 behavior. Its production execution is
one guarded-stack, syscall-backed HIA ELF and one genuine owned state buffer.
Tensor values, gradient Holds, Generator state and graph metadata participate in
the same frame transaction. No host process computes random values or gradients.

The core SHA256 is
`490fc144160f6e4440aed001363f86ccff56c335f9781e38b058fc6106b1be09`;
the ELF SHA256 is
`5baa9cd5e5625f0e2324941486982b714dcdedfe897f5c16686a3ac8882ce242`.
The executable is 1,392,640 bytes. Its 275 core functions, 3,442 blocks and
13,097 instructions compile as five bounded core modules plus the unchanged
17-function RNG module and owned-entry module. Every existing native compiler,
linker, effects, resource-lifetime and whole-program stack check remains active.

## Exact composition

The builder compares three complete, pinned HIA sources: materialization05,
the common RNG successor, and the nullable-gradient successor. It proves that
the two deltas overlap only in `init_state` and `state_header`, where the
nullable source changes only its state magic. It then copies six nullable
functions and the single new `zero_grad_none` function byte for byte. All 264
other common function bodies are byte-identical. Beyond the six copied
replacements, only four explicitly identified state-tag functions change. The common command dispatcher entry, including
command58's suffix-stop and prefix-commit path, is byte-identical. The separate
RNG and owned-entry modules are also byte-identical.

The six copied replacements are `descriptor`, `tgrad`, `zero_grad`,
`ensure_backward_work`, `release_backward_scratch` and `object_command`.
The new function is `zero_grad_none`. The builder generates no numerical
operations; it partitions these authoritative HIA bodies and submits each unit
to the actual native ABI3 compiler, then uses the actual native stack linker.
Python is used only for source composition and qualification.

## State and wire contract

The new prefix magic is **HTRNGN01** and the backing magic is **HTORNGN1**.
The existing seven-kind table layout is unchanged. Generator remains kind7;
its private kind6 Hold contributes exactly once to Storage ownership. The
global numerical prefix is 139,264 bytes. Within the backing object, records
start at 128, the free table at `128+448*slots`, audit storage at
`144+464*slots`, and the data pool at `144+488*slots`.

The external frames remain HTOWNR01/HTOWNS01 and the inner packets remain
HTORCH03/HTORSP03. Commands60..69 retain their common uniform/Generator/view
meaning. Command70 accepts one differentiable leaf Tensor and zero reserved
arguments; it drops a defined gradient's real private Hold/Storage and resets
the gradient address and presence. Repeating it on None succeeds without
allocating. A later backward reserves a real gradient again, including a
zero-length Storage for an empty defined gradient. Command8 zeroes only a
defined gradient; it does not materialize None. Value readout of None remains
compatibility zeroes while command59 reports its actual absence.

Presence remains node field360; fields336,368,376 remain temporary staging and
must be zero at every returned state. No new persistent node field is used.
The global112 word remains reserved-zero and global120 remains unused by this
profile; neither is silently assigned to the Generator.

Ordinary command and admission failures, including command57 failures, restore
the entire old owner. Command58 runtime errors retain completed prefix effects
and skip the suffix. Thus `[70, uniform, 58-runtime-error, uniform]` can commit
the gradient becoming None and exactly the first fill's Tensor/Version/RNG
changes. The same prefix followed by a failing57 or ordinary command rolls all
of them back. A capacity replay restarts from the original owner and advances
the committed Generator only once.

## Executed qualification

| Suite on this exact image | Observed result |
| --- | --- |
| Nullable storage's six native groups | 179 frames; 131,101 exact gradient words; 10,671,232 full-owner rollback bytes; 261 node invariant checks |
| Ten actual Torch gradient-clear observations | 780 frames; 2,669 exact comparisons; 288 comparisons under the unchanged Adam policy |
| Previous common joint tests | 3 groups; 155 frames; Generator, partial-commit and rollback assertions retained |
| New nullable/RNG joint tests | 4 groups; 56 frames; 393 exact words; 34 complete MT states; 2,500,896 rollback bytes; 62 node checks |
| Real owned-state handoffs to both predecessors | 4 directions; 17 frames; refusal preserves pointer, length and every byte; correct runtime resumes |

The new joint tests perform two independent 12-update training sequences using
57 and58, respectively: clear-to-None, uniform, forward, backward, SGD and graph
reset in the same process. Each update produces a distinct gradient Hold while
parameter/gradient bits, Version and full MT state match independent exact
models. Another case grows the object table with 100 Generators after clearing
a gradient and drawing a prefix, then encounters a real Saved-version error.
It checks prefix commit, suffix non-consumption, correct seed states, recovery
and the complete-rollback controls. Empty fills increment Version while
consuming no random word and still permit genuine zero-length gradient
rematerialization.

The two predecessor handoff executables contain both complete runtimes and a
checked HIA owner controller. The new state first defines and then actually
clears a gradient before the wrong-profile call. Neither host-created owner
descriptors nor serialized native pointers enter production. Both ownership
directions are rejected and recovered; closing the live allocation makes its
mapping absent while the test process remains alive.

The nullable tests' private read-only observers originally knew HTSTATE6 and
the six-kind pool offset. The portable adapter changes only those tags and
`424*slots` to `488*slots`; its exact edits and before/after hashes are recorded.
The common joint observer changes only its magic. No input packet, expected
numerical result, tolerance or error/commit expectation changes in these three
replays. The stronger handoff test's additional defined-to-None sequence is
separately recorded as a new test, not called an unchanged predecessor replay.

## Remaining scope

The ten-case actual-reference replay retains all prior object-model gaps:
11 momentum-state Version observations, 9 explicit/lazy state-creation
observations and 30 absent native step-Tensor observations. Six Adam words
differ in raw bits while satisfying the unchanged previous tolerance; they
are not called bit-exact. The actual Torch process ran in the recorded hosted
reference job; this package performs native replay of its fixed observations.

This component does not implement an Optimizer-owned state object, public
gradient-Tensor identity/Version or independently held gradient aliases. It
does not broaden uniform to tracked nonleaf autograd, normal/dropout, other RNG
dtypes, default concurrent Generators, higher-order differentiation or full
PyTorch. All previous fixed capacity limits remain. Gradient release returns
an extent to the real owned pool; it does not promise one OS unmap per gradient.
The full rewrite and full RNG gates remain open.

## Reproduce

The only native-build inputs are the three frozen HIA build directories and
the frozen ABI3 tools. No new upstream checkout or host compiler is needed to
compose and compile this successor; the predecessors' own build procedures
remain responsible for producing their exact pinned sources.

```sh
python -B tools/tensor_rng_nullable/rebuild.py \
  --common-build build/tensor-rng-composition/native \
  --materialization-build build/gradient-materialization \
  --nullable-build build/gradient-none \
  --stack-build build/stack-tools \
  --output build/tensor-rng-nullable
```

The image is `build/tensor-rng-nullable/tensor-rng-session`. Reproduction checks
all seven unit sources, objects and final image, plus before/after input custody.

```sh
python -B tools/tensor_rng_nullable/adapt_tests.py \
  --nullable-package tools/gradient_none \
  --common-package tools/tensor_rng_composition \
  --output build/nullable-rng-test-sources
g++ -std=c++17 -O2 tools/tensor_rng/reference.cpp -o build/common-mt-reference
python -B tools/tensor_rng_nullable/test_combination.py \
  --executable build/tensor-rng-nullable/tensor-rng-session \
  --rng-package tools/tensor_rng --cpp-reference build/common-mt-reference \
  --output build/nullable-rng-joint
```

The adapted `test_nullable.py` and `replay_grad_clear.py` use their predecessor
CLI with the new explicit executable and source hashes. `test_handoffs.py`
takes `--build`, `--rng-package`, `--stack-build`, `--peer-pins` and repeated
`--peer common05=... --peer nullable70=...`; the supplied peer pins bind every
original source unit. Replays never modify the frozen predecessor packages.
