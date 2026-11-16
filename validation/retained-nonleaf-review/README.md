# Independent retained-nonleaf native qualification

The fixed real-Torch reference contains **8 families / 12 variants**. The first
native candidate passed all twelve without a replay or expected-value change:

| Bound input or observation | Identity or result |
| --- | --- |
| Native HIA source | `666230f05cc6942470006d1f184135321d8cbc26a6c6ba0bf7e3cf9857fe515b` |
| Native ELF | `7568c0a4ec40d82d3a0b46d1e8a27499dd96e8fe3fd15433523131a196da89b5` |
| Real reference | `ca8b5d3a7c679dcc5ad9a3d7b2bbd38f664c90e6474cf3e3c28496d5b716d1b3` |
| Fixed reference probe | `7cdb53f7a95c48acd1b85d93d90c693d4caa97ada72a54a4021bb616d098fcba` |
| Independent native replay | `c925e33872dc87ead29ecc2125b1799763a87b13c8ae2eba85e966b4202f1d8e` |
| First native report | `6540097fd11f9f1155813309e1237101567a4555636af4678a2c2858a3156e08` |
| Executed interactive frames | 3,849; every response before stdin EOF |
| Assertions | 38,529 |
| Exact binary32 word observations | 1,352 |
| Adam parameter/moment positions under unchanged tolerance | 188; zero raw-bit differences |
| Full-owner bytes checked after ordinary admission refusals | 9,087,312 |

The primitive snapshots compare the actual gradients, None, strides, offsets,
Storage sizes, absolute Versions, retain flags and opaque Tensor/Storage
identity relations. They cover replacement rather than mutation of retained
G, old held/Saved aliases, empty and strided G, lazy SGD/Adam/AdamW state,
the retain hook's committed effects before errors 6 and 10, weak nonleaf roots,
and Optimizer-only strong roots. `SEMANTICS.md` identifies each observation
in the original JSON.

The original Adam tolerance remains `rtol=4e-5, atol=5e-6` only for parameter
and moment numerical values. The report preserves every raw word. Gradient
and step Tensor words, identity, metadata and Versions are exact. Python
warning objects, wrapper weakrefs and class-name spelling are diagnostic
only. Completed-node/leaf fields are recorded but are not assigned invented
Torch expectations: this reference did not register node hooks.

## Portable strict replay

From a repository containing this handoff, with a genuinely rebuilt retained
candidate, use the exact source and ELF pins above:

```sh
python3 tests/retained_nonleaf_review/replay_retained_nonleaf.py \
  --observations validation/retained-nonleaf-review/reference/torch-retained-nonleaf.json \
  --probe validation/retained-nonleaf-review/reference/pytorch_retained_nonleaf_probe.py \
  --protocol tests/retained_nonleaf_review/protocol.py \
  --source build/retained-gradients/native/runtime.hepta \
  --binary build/retained-gradients/native/tensor-retained-session \
  --expected-source-sha256 666230f05cc6942470006d1f184135321d8cbc26a6c6ba0bf7e3cf9857fe515b \
  --expected-sha256 7568c0a4ec40d82d3a0b46d1e8a27499dd96e8fe3fd15433523131a196da89b5 \
  --output build/retained-gradients-independent-12
```

The `build/...` names are explicit example destinations, not assumptions made
by the runner. All paths are arguments. An output directory must be new;
previous results cannot be overwritten. `--prepare-only` validates the fixed
reference, probe, codec and transport without importing Torch or executing a
native binary. Every real execution snapshots its runner and the two pure
transport files and checks all input hashes again after completion.

The expected committed state tags are `HTRET001` and `HTORET01`, with a
576-byte descriptor and the existing eight-kind owned layout. The runner only
reads its own child's memory to observe stable Tensor incarnation and whether
a public getter token is temporary. It never writes that memory. Temporary
getter tokens are closed after primitive snapshots, preserving the recipe's
actual ownership obligations.

## Additive link to original Optimizer row 14

The original `7ad0a2e7...` 24-case oracle and `ea748d41...` replay remain
unchanged. A second small script executes precisely the earlier unexecuted
retained-nonleaf constructor observation with the original zero-momentum,
zero-weight-decay SGD defaults. It first rejects the nonleaf without retain,
then accepts the **same Tensor** after command 93. Its additional native
identity, no-state-construction, rollback and audit assertions are labeled
interface controls, not additional Torch observations.

That additive run passed in 15 interactive frames. It does not reclassify the
old partial report or rerun the other 23 original recipes. A combined ledger
may cite this exact sub-observation together with separately bound executions
of the remaining original inputs.

```sh
python3 tests/retained_nonleaf_review/replay_optimizer_retained_admission.py \
  --observations validation/retained-nonleaf-review/reference/original-optimizer-observations.json \
  --probe validation/retained-nonleaf-review/reference/pytorch_optimizer_object_probe.py \
  --old-replay validation/retained-nonleaf-review/reference/replay_optimizer_objects_v2.py \
  --protocol tests/retained_nonleaf_review/protocol.py \
  --source build/retained-gradients/native/runtime.hepta \
  --binary build/retained-gradients/native/tensor-retained-session \
  --expected-source-sha256 666230f05cc6942470006d1f184135321d8cbc26a6c6ba0bf7e3cf9857fe515b \
  --expected-sha256 7568c0a4ec40d82d3a0b46d1e8a27499dd96e8fe3fd15433523131a196da89b5 \
  --output build/retained-original-optimizer-subcase
```

## Evidence and limits

The handoff includes the unmodified real reference and source probe, the
reader's semantic extraction, both actual native reports, and a raw archive
with every original request, response, observed snapshot and executed-source
copy. Archive members are checked individually against the original bytes.
Producer source generation and separate OOM/resource/typed-owner gates are
other agents' responsibilities; this report does not claim to have rerun them.

This is evidence for the stated first-order CPU binary32 scope. It does not
close custom hooks, higher-order differentiation, sparse or other-dtype/device
behavior, arbitrary in-place history rebasing, every optimizer mode, all 2,590
schemas, or the full PyTorch rewrite.
