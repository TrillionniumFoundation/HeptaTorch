# Common native Tensor, Generator and backward runtime

This profile combines the complete frozen gradient-materialization05 core with
native Tensor uniform and the frozen CPU MT19937 engine. It is one HIA-linked
ELF and one actual persistent owned buffer, not two communicating numerical
services. It retains commands 57/58/59, first-gradient bit preservation,
gradient presence, per-node Saved release, the default/retain flag and all
existing numerical kernels. The Generator tables and Tensor storage share one
transactional state owner; no native owner descriptor is copied into that pool.

The qualified executable SHA-256 is
`cdcaebb2785639585de7fd2a4e1f98d41a56e885f1657c3485ce5f187f89d726`;
the complete numerical core source is
`7447c169d3b2cad1e55cb7b1b8970d67f4103c5866f6db0a60c0b4d05d02cc0b`.
Seven bounded native units produce a 1,372,160-byte executable. Native compiler,
owned-resource ABI, provider and guarded-stack linker profiles are unchanged.

## Exact source composition

The parent is the entire 261-function materialization05 HIA source, SHA
`9feab47afea6723284e15a2ad6c9071dc8c5bdc27b572d2eef4ab4628ed1aea9`.
Of those bodies, 237 are byte-identical in the common core. Exactly the 24
already declared Tensor-RNG layout/entry functions change. No backward,
gradient materialization, gradient-presence, optimizer, dispatch or numeric
body is replaced by the older Tensor09 implementation.

The adaptation checks each affected layout function against both parent
sources. It accepts only the explicit state-magic difference in `init_state`
and `state_header`. The partial entry comes from its hash-bound authoring
source; after removing only the added scratch argument and the 60–69 dispatch
branch, its AST must be exactly identical to the original partial entry.
In particular its command58 suffix-stop branch remains present. The original
owned step's commit/refusal logic is preserved, with one additional genuinely
owned temporary RNG buffer supplied as a checked mutable loan.

The complete 17-function frozen RNG03 non-entry source is still byte-identical
in its own module. Its byte helpers remain private. Commands 60–69 retain the
standalone Tensor-RNG wire meaning, with the separately documented binary64
admission correction in `RANGE_ADMISSION.md`. Source generation and tests may
use Python; production numerical/state execution never does.

## Shared state and failure semantics

Outer framing and application packets remain `HTOWNR01`/`HTOWNS01` and
`HTORCH03`/`HTORSP03`. The internal prefix magic is **`HTRNGP01`**, and the
seven-kind backing magic is **`HTORNGP1`**. The table/pool formulas remain the
new Tensor-RNG seven-kind layout. Node field 360 remains persistent gradient
presence. Fields 336, 368 and 376 remain exclusively the materialization05
temporary staging fields and are cleared before normal success or runtime
error return. This profile does not allocate new persistent node fields.

Every ordinary command/admission failure and every command57 failure rolls the
entire frame back, including any earlier Tensor fill and Generator advance.
Command58's runtime error instead remains a successful response envelope whose
last record reports the backward error: the earlier prefix and completed nodes
are committed, and later commands are skipped. For `[uniform,58-error,uniform]`,
exactly the first fill's Tensor bytes, shared Version and Generator draws commit.
The second fill consumes nothing. A capacity-growth replay starts from the
original owner, so a uniform prefix is never committed twice. If 58 succeeds
and a later ordinary command fails, the existing whole-frame rollback applies.

The actual native qualification includes both Saved-version error6 and spent
Saved error10, atomic57 controls, a 100-Generator growth replay before a partial
error, later ordinary failure after successful58, complete MT-state comparison,
gradient presence, and cleanup of the transient staging fields. Separate 12-step
training sequences for57 and58 execute native uniform, forward, backward, SGD
and graph reset in the same process, with exact parameter/gradient bits and MT
state after every update.

## Qualification and retained limits

- The common image passes the unmodified, hash-bound replay of all 13 real
  Torch signed-zero cases, all 16 partial-backward cases and all 8 gradient-
  presence/optimizer cases. The last suite retains its existing
  `QUALIFIED_WITH_OPEN_STATE_LIFECYCLE` result; it is not relabeled as complete
  Optimizer object compatibility.
- All 88 actual Torch double-bound observations match exactly: admission,
  Version, complete MT-state hash and every returned binary32 word. The native
  program also matches independent C++ double and exact-integer checks.
- The declared RN replay passes eight groups / 729 frames, 33,193 exact values,
  160 full MT states and 42 complete owner rollbacks. The original frozen RN03
  script's stricter range refusal is preserved as a genuine historical
  contract-change failure; two explicitly identified negative-control inputs
  are adapted in a separate file. No numerical tolerance changes.
- The unchanged RN entry/work suite passes 88 cases, including all short output
  capacities, real allocation refusal and 1,000,000 checked native fill values.
- Eight real owned-state cross-profile directions pass against the complete
  HTSTATE3, HTSTATE4, HTSTATE5 and HTSTRNG1 runtimes: incompatible state is
  rejected with its pointer, length and bytes unchanged, then the correct
  runtime resumes. This uses real HIA-created owners and native linkage of
  both full runtimes, with reversible source-symbol renaming only.

This remains a bounded CPU binary32 component. It does not implement normal,
dropout, other Tensor RNG dtypes, a global/default concurrent Generator,
tracked-nonleaf uniform autograd, arbitrary Python dispatch, general hooks,
higher-order AD or a real lazy Optimizer object. Commands68 and69 retain the
explicit no-grad/untracked scope in the standalone protocol. The state audit
relies on checked producer/consumer invariants; it is not an arbitrary foreign
owner-byte authentication mechanism. Full PyTorch and all-RNG gates remain open.

## Rebuild

Minimum native build inputs are: the frozen Tensor09 build, materialization05
build, partial04 entry authoring, standalone Tensor-RNG package, frozen RNG03
source, pinned Hepta authoring checkout and ABI3 stack tools. The older release04
build is needed only for the optional four-predecessor handoff qualification.

```sh
python tools/tensor_rng_composition/rebuild.py \
  --tensor09 build/tensor-objects \
  --partial-build build/gradient-materialization \
  --entry-authoring build/backward-partial/overlay-authoring.py \
  --rng-package tools/tensor_rng \
  --rng-source tools/rng-cpu/rng.hepta \
  --upstream /path/to/pinned-hepta \
  --stack-build build/stack-tools \
  --output build/tensor-rng-composition
```

The output image is `build/tensor-rng-composition/native/tensor-rng-session`.
The rebuild verifies exact source, every native object and final image pins;
`composition.json` records the whole parent body map and reversible entry AST.
The source-only adapter does not vendor the full upstream compiler. All input
source hashes are checked before use and again after execution.

The portable native runners are `test_joint.py`, `test_range.py`, `replay_rng.py`
and `test_profile_handoffs.py`. The standalone package's `test_entry.py` and
`test_composition.py` also run directly against this build; the latter accepts
`--backward-code 57 --presence` or `--backward-code 58 --presence`. Every report binds the actual executable.
The fixed Torch observations are included as evidence; replay does not claim a
new Torch process was executed locally. An independent same-run live-Torch
joint oracle is a separate qualification and is not implied by these replays.


## Minimal native qualification commands

The C++ executable is an independent MT19937 seed/draw reference used only by
tests. It is never linked into or called by the production runtime.

```sh
g++ -std=c++17 -O2 tools/tensor_rng/reference.cpp -o build/common-mt-reference
python -B tools/tensor_rng_composition/test_joint.py \
  --executable build/tensor-rng-composition/native/tensor-rng-session \
  --rng-package tools/tensor_rng \
  --cpp-reference build/common-mt-reference \
  --output build/common-joint-qualification
python -B tools/tensor_rng_composition/test_range.py \
  --executable build/tensor-rng-composition/native/tensor-rng-session \
  --rng-package tools/tensor_rng \
  --cpp-reference build/common-mt-reference \
  --torch-observation tools/tensor_rng_composition/evidence/pytorch-uniform-range-probe.json \
  --output build/common-range-qualification
python -B tools/tensor_rng_composition/replay_rng.py \
  --build build/tensor-rng-composition/native \
  --rng-package tools/tensor_rng \
  --output build/common-rng-replay
```

These three commands exercise 3 joint cases, 88 exact range cases and the
8-group native RNG model. The broader entry, handoff, training and unchanged
backward replays are separately recorded in `evidence/`; the minimal commands
do not claim to rerun those other suites. For `test_entry.py` and handoff tests,
pass `--build build/tensor-rng-composition/native` (the inner native directory).
The four-predecessor script takes repeated `--peer label=build-directory` and
the supplied `handoff-peer-pins.json`, without rewriting any predecessor.
