# Public gradient objects and the combined native owner

Two new reproducible profiles extend the existing owned Tensor implementation.
The first gives public gradients real Tensor, Storage, Version and lifetime
identity. The second composes that implementation with the existing owned
Adam/AdamW and CPU distribution runtime. Numerical execution, gradient
propagation, state mutation, allocation and reference release execute in native
HIA. Python tools generate exact HIA source, invoke the native compiler, transport
packets and compare results.

| Profile | Core functions / native units | Source SHA-256 | Executable SHA-256 |
| --- | --- | --- | --- |
| Public gradient objects | 355 / 8 | `1b18da2cf9791e1b20982f9706434c1359ea194dd983dc6c66eb005f152e77b5` | `be18ab877f38a4027a527783db80a1418a137c7d6da91e5bc39e2af8c044cb6b` |
| Public gradients, Adam and distributions | 428 / 9 | `1fed8ea7ba5bf755a9df9ca1e61d34cfd4712cc2fd66e0728493091c81c24dae` | `6773485896d961b6c2e62784ebb66320290df47128bd1331aecc0c8b7f6611fe` |

The combined profile has 5,738 core blocks and 21,348 core instructions. Its
576-byte descriptors use `HTGAD002` / `HTOGAD02`; the gradient-only parent uses
`HTGRAD02` / `HTOGRAD2`. Wrong-producer nonempty states are rejected with their
owner identity, length and complete bytes intact. These internal format tags
are not a checkpoint or a general public Tensor ABI.

## Reproduce the native implementation

Use the pinned HIA checkout and ABI3 toolchain described in [build.md](build.md).
`HEPTA_SOURCE_ROOT` must identify the pinned source checkout;
`HEPTA_STACK_BUILD` must identify the reproduced compiler/linker build.

```sh
make test-gradient-objects \
  HEPTA_SOURCE_ROOT=/absolute/path/to/pinned-hepta \
  HEPTA_STACK_BUILD=/absolute/path/to/reproduced-stack-tools

make test-gradient-adam-distributions \
  HEPTA_SOURCE_ROOT=/absolute/path/to/pinned-hepta \
  HEPTA_STACK_BUILD=/absolute/path/to/reproduced-stack-tools
```

The first target builds the exact owned-SGD prerequisite and executes the frozen
16-step gradient qualifier. Both a cold HIA overlay and the full authoring lineage
must produce the same runtime source and executable. The aggregate reader checks
124 frozen package inputs, the eight actual object files, every original child
report, the reference provenance, the unchanged legacy differences and both
predecessor handoffs. The executable is published to
`build/gradient-objects/tensor-gradient-object-session` only after qualification.

The second target builds the exact gradient and Adam/distribution parents and
executes the frozen 17-step combined qualifier. The root orchestration also
executes the independent compiler/entry/handoff gates, the unchanged four
autograd cross-feature runners, and the strict extreme-stride observer/comparator.
Every gate receives the same actual native image and source. The resulting image
is `build/gradient-adam-distributions/tensor-gradient-adam-distributions-session`.

Root aggregate reports are written as
`build/hosted-gradient-objects-native.json` and
`build/hosted-gradient-adam-distributions-native.json`. They retain complete
decoded child reports plus each original report's byte length and SHA-256.
Original request/response records and build artifacts remain in the review
directories and the frozen validation archives. JSON integers are handled by
Python without truncating u64 payloads or extreme strides.

The lower-level `gradient-objects` and `gradient-adam-distributions` Make targets
build the images without claiming that they have executed all qualification.
Their portable CLIs are documented in the corresponding `tools/` READMEs.

## Behavior qualified by this increment

Public `.grad` returns the actual bound Tensor. Assignment and clearing manage
real private references; a held gradient, an alias, a Saved record or another
checked owner can keep its Storage alive after the parameter's binding changes.
Incoming gradient creation, propagation and accumulation preserve the observed
layout and Version relationships within the fixed cases. Typed request-local
references cannot reinterpret a setter scalar or a getter's None as a Tensor.

The combined owner keeps physical Preserve-layout incoming Storage separate from
the logical gradient work array. This fixes a real wrong-order gradient when a
transposed Bernoulli zero contribution arrives before a nonuniform contribution.
Capacity admission reserves required resources before partial backward, and
success or a committed operation error releases unused transient references.
Four explicit resource controls include real address-space exhaustion and recovery.

Pointwise layout selection writes through the actual selected Tensor strides.
For the observed extreme size-one strides, the comparator converts element
strides to wrapped signed byte strides before ordering. This is the sole function
change from candidate04 to05; all other 427 core functions remain byte-identical.

Command95 uses a narrow binary32 helper for the fixed CPU scalar NaN observations.
It preserves the observed payload and sign, quiets signaling NaNs and retains the
observed scalar operand priority. Its finite path and the original global
arithmetic helper are unchanged. This is not an all-operator or all-ISA bitwise claim.

| Separate reference or native suite | Qualified scope / explicit boundary |
| --- | --- |
| Original public-gradient 10 | 9 qualified; 1 partially represented |
| Incoming-gradient 25 | 22 qualified; 3 hook/create-graph cases unrepresented |
| Alias/edge 12 | 12 qualified |
| Bernoulli gradient/layout 8 | 8 qualified; 2,112 native frames |
| Original Optimizer 24 | Disjoint original SGD, Adam21–24 and newly mapped case20 yield 23 qualified and 1 partial retained-nonleaf case |
| Distribution 183 | Atomic: 161 pass / 4 transaction differences / 18 unrepresented; explicit operation-error path: 165 pass / 18 unrepresented |
| Independent scalar NaN 48 | 48 pass on the actual fixed CPU observations |
| Independent mixed-layout 2 | 2 pass, including actual forward strides and backward gradient layout |
| Independent cross-feature 6 / math 2 | Separately counted lifetime, rollback, Generator and wrong-order controls pass |
| Extreme-stride 8 | Same actual reference and comparator: candidate04 has 4 failures; candidate05 has 8 passes |

Suite counts overlap. They are not added to the 2,590-schema inventory or called
unique upstream API coverage. The legacy presence8 and clear10 reports retain
their original state-Version, lazy-creation, missing-step and tolerated-word
differences. An owned Optimizer result does not rewrite the old explicit-state API.

## Hosted qualification and evidence

Each new image has three distinct Native CI steps: native qualification,
unchanged live32 and unchanged frozen35. The live32 source remains
`554671168c7a4a408dc1e1df89a19db15c3efcefcf4ae1a300af1b6dcae40160`;
the frozen35 source remains
`149b41c455444edeb8d25d0ad98045e29683e780f37f7e210e7985242d359597`.
Their original inputs, expected values, tolerances and absolute Version checks
remain intact. Preparation-only CLI checks are recorded as `NOT_RUN` and do not
count as actual PyTorch execution. The complete new hosted results remain a
separate admission gate.

The workflow now requires 30 report envelopes. It retains the original 24 reports
and all earlier steps in order, adds six new steps, and increases the job timeout
from 20 to 40 minutes to accommodate the actual extra cold builds and regressions.
The original full reports are exported without JSON reserialization in the CI ZIP.

The 378-function predecessor already passed both exact-head 68-step jobs and its
own live32/frozen35. Its original 50-report artifact, root333 checks, independent
7,944 audit assertions and verified merge receipt are retained under
`validation/hosted-adam-distributions-3e75b3a/`. Those are audit assertions, not
additional PyTorch tests, and that predecessor's result is not inherited by355/428.

The frozen producer and independent packages are under
`validation/gradient-adam-distributions/`,
`validation/gradient-composition-review/` and
`validation/adam-distributions-gradient-review/`.
Entire candidate04 peer packages remain under their `history/frozen-package-04/`
directories. Original native failures, the compiled but unqualified candidate03,
host observer errors and their narrow corrections retain separate identities.
The root's new cold executions are under `validation/gradient-objects-root/` and
`validation/gradient-adam-distributions-root/`.

## Work that remains open

Nonleaf `retain_grad` needs actual contiguous first-gradient materialization,
out-of-place subsequent accumulation, hook timing on a node's own exception,
weak lifetime and Optimizer acceptance. Current command slots93/94 remain reserved.
Hooks, forward/higher-order AD, arbitrary incoming use-count and layout paths,
general gradient dtype/device assignment, arbitrary real Adam step exponents,
the 18 missing distribution dtype/mask cases and all other full-system
requirements remain in scope.

The target is still PyTorch v2.15.0 at its original pinned commit. Full-schema
parity remains0, all17 full-system gates remain open or in progress, and
whole-project acceptance remains **HOLD**.
