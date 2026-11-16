# Explicit partial Bernoulli and dropout errors

This is a separate native HIA profile extending the frozen atomic distribution
profile03. It preserves every one of the predecessor's 19 `dist_*` function
bodies and adds explicit commands86/87 for the four late-error side effects
observed in the fixed CPU PyTorch wheel. Commands80–85 retain whole-frame
rollback. The observation, its probabilities, all Tensor words, Versions, and
complete 5056-byte Generator states are unchanged.

The component remains **PARTIAL**. Of the original183 actual observations,
165 are represented and strictly pass with the new explicit interfaces;
18 require probability Tensor dtypes or the distinct `native_dropout` boolean
mask interface that this float32 Tensor profile does not implement. The same
new executable, using the unchanged old interfaces, still reports exactly
161 PASS,4 transaction differences,18 unrepresented. The old differences have
not been deleted, relabelled, or made green by weakening their checker.

## Identity and implementation

| Item | Identity |
| --- | --- |
| Runtime source | `55ed0c6e16317add03974c01ffd85e38684fc535a3e5cb6423ed043f4217f05b` |
| Native ELF | `bb4c25e35a170f08276d464db6b630cf03765ff816e6a9ca9aa1651360a1ccf4` |
| Atomic predecessor source | `a72a98232c7b7f10efa67ce29b21e417c58201383bef9aacc266fe247ff356ce` |
| Frozen actual report | `c624f5c4762b1c71e7053aba9af55aaa3d8bbaae5aa100b0a6df5f44f58da467` |
| State and object tags | `HTDIST02` and `HTODIST2` |
| Source size | 300 core functions,3814 blocks,14441 instructions across7 native units |

The six new functions are real HIA code. Python authors and verifies source,
invokes the pinned native compiler, and assembles a request for the native
ABI3 linker. Tensor values, probability comparisons, MT advancement,
noise creation, writes, ownership, and cleanup run in the resulting ELF.
The RNG source and owned `step`/`close` module are byte-identical to the
predecessor. No numerical host library is called by the target.

`profile.json` binds every unit source/object hash and each of the294 predecessor
function bodies. Only `entry` and four state-tag functions change; the old
gradient, RNG, distribution and storage bodies remain identical. The internal
pool remains seven kinds at offset `144+488*slots`; the tags distinguish the new
operation-error contract. Real checked owner handoffs prove both profiles reject
the other's state, retain its exact pointer/length/bytes, and allow the proper
profile to resume and finally release the owner.

## Rebuild and qualify

The inputs are the frozen profile03 build, its source-only tool package, the
pinned Hepta authoring checkout and the frozen ABI3 stack tool build. The
predecessor build must contain its exact `overlay-authoring.py`, runtime,
RNG source and owner source. Do not substitute the earlier nullable-RNG build.

```bash
python -B tools/cpu_distributions_partial/rebuild.py \
  --baseline build/cpu-distributions \
  --upstream /path/to/pinned-hepta \
  --stack-build build/native-stack-tools \
  --output build/cpu-distributions-partial

python -B tools/cpu_distributions_partial/qualify.py \
  --build build/cpu-distributions-partial \
  --baseline build/cpu-distributions \
  --atomic-tools tools/cpu_distributions \
  --stack-build build/native-stack-tools \
  --output build/cpu-distributions-partial-qualification
```

`rebuild.py` verifies this package before executing authoring, and checks exact
source, object and ELF reproduction afterward. `build.py` verifies upstream
source custody before import and all actually loaded source files afterward.
`qualify.py` verifies the predecessor's exact source manifest, runs its original
atomic replay unchanged, then runs explicit replay, four late-error cases,
independent native controls, owned-entry guards and typed handoffs. It returns
success only for the documented165/18 and161/4/18 partitions plus every native
control. Its aggregate JSON status deliberately stays `PARTIAL` and all three
completion flags stay false.

The inherited host authoring step checks one actual function at a time against
interfaces extracted from all actual functions. This avoids an aggregate JSON
document-node limit; it does **not** claim aggregate Core admission. Every
emitted unit still passes the unmodified pinned native compiler, including
effects, ownership, loans, CFG and limits. The native linker validates exact
cross-unit contracts and the producer-bound whole-program stack metadata.
No old compiler check or limit is removed.

## Qualification meaning

The fixed actual replay compares values and layouts, storage aliases, Versions,
complete MT states including normal caches, dropout Saved-noise behavior,
gradients and graph release. The four previously different late errors are
three invalid Tensor probabilities after a valid prefix and one tracked-leaf
inplace dropout error after generating noise. New records preserve the observed
prefix and stop subsequent commands.

Additional native controls use CPython's independent MT engine only as an
oracle. They cover invalid probabilities at0,1,623,624,625 and1249; retained
normal caches; repeated-stride probability feedback; repeated-stride dropout;
noise generation before tracked-leaf and expanded-view errors; actual pool
growth without double consumption; prefix Tensor writes and Generator advances;
skipped suffixes; and saved aliases whose Version remains unchanged by a failed
inplace operation. These are labelled source/model lifecycle checks, not new
hosted PyTorch observations.

The owned-entry suite links a real HIA probe to the unchanged production
objects. It checks output capacities0/31/32/63/64/95/96, exact full-owner equality
with interactive execution, later ordinary-error rollback, Version overflow,
real address-space allocation refusal and subsequent recovery. Its raw owner
bytes are qualification-only: the public session never accepts a process
pointer or arbitrary serialized native owner. `/proc` observes only the test's
own child process; it does not execute target mathematics.

## Preserved failures and limits

The first explicit replay's test harness compared owner identity *after* a
successful state-query frame, which correctly commits a new clone. Its
incomplete report contains15 false harness differences in143 completed rows.
The complete raw90336851-byte report is preserved losslessly compressed with
length/SHA verification, together with the original script and the exact diff.
The semantic correction moves the comparison before the query. Two diagnostic
changes bound byte rendering and update the module description; no expected
values or target bytes change. The first whole-report in-memory compression
attempt was killed with exit137; the verified streaming archival succeeded.

An additional native lifecycle fixture initially used the reduction opcode,
then the wrong binary argument order. Native admission correctly rejected those
test inputs with statuses4 and1. Both scripts, transcripts and corrections are
preserved. The correct public wire form is `(2,3,x,x)` for `x*x`. The final
gradient expectation is unchanged; no production fix was needed.

Limits remain rank4, per-axis262144, Tensor numel16777216, and at most1000000
elements per active distribution. A float32 Tensor-probability draw consumes
one raw MT32 word per valid element. Scalar-double Bernoulli used by dropout
consumes two. Normal-gradient-mode inplace operations on tracked nonleaf
Tensors remain outside this profile. Optimizer kind8 and public-gradient object
successors require a separate real composition; they are not silently replaced
by this seven-kind build. The full RNG, dtype, autograd and PyTorch rewrite
targets remain open.

`primary-sources.json` retains the exact17 official PyTorch source identities
used by the predecessor, including TensorIterator, overlap, Bernoulli, dropout,
distribution transforms, `empty_like`, dense-stride inference and derivatives.
The fixed actual report and probe remain in the frozen predecessor package.
This package's evidence binds them by SHA and preserves its own native source,
objects, ELF, wire transcripts and all failed attempts.
