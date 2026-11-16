# Numeric Tensor objects v3: real PyTorch oracle entry

This package prepares and runs the frozen strict 28-case numeric Tensor oracle
through the actual v3 object protocol. It adds seven object-lifetime cases in a
separate report group. The new entry, bridge and lifecycle file belong together
in one directory. They contain no target numeric kernel and do not replace the
native HIA runtime with Python.

## Execution

Provide the three unchanged reference files explicitly. This example uses
repository-relative names for an integration; the receiving repository can choose
another directory while retaining the exact dependency hashes.

```bash
python3 tests/tensor_object_oracle/pytorch_tensor_objects_oracle.py \
  build/tensor-object-session build/tensor-object-pytorch.json \
  --owned-reference tests/pytorch_owned_tensor_oracle.py \
  --session-reference tests/pytorch_session_oracle.py \
  --reference tests/pytorch_oracle.py \
  --lock compatibility/pytorch-oracle-lock.json \
  --repo . \
  --expected-native-sha256 4bd22ab115a5428be6ee1ab77fd31a1e20baf14a80756eb38f000574cbad531f
```

`--prepare-only` performs source, case-plan and reversible AST checks without
importing Torch or executing a native test case. `--trace-directory DIR` also
saves every actual outer request and response frame; the directory must not
contain an earlier run's case directories. Normal reports include ordered wire
hashes without requiring these potentially large raw traces.

The normal entry loads the unchanged reference support. Its eleven provenance
checks, deterministic setup, numeric comparators and direct parameter absolute
version assertions all remain active. If Torch is absent or does not match the
lock, the entry writes `NOT_RUN` and exits 2. It cannot fall back to a different
Torch version or synthesize a reference.

## Frozen dependency custody

| Dependency | SHA-256 |
| --- | --- |
| 28-case owned Tensor oracle | `ee3cf72154ac9dfa31055557b3facab8fdd3e14aad03ef6000ced55ccd75ed88` |
| 21-case session oracle | `35fa5fa3e1b740f901c11d46dca484079b77805004586c03ae42b35be2265577` |
| Reference support | `ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae` |

The complete 28-case file is executed unchanged as a definition provider; its
guarded main does not run. Its original session adaptation is then used without
changing any test function or registration. Six v3 changes are applied to
`Session.exchange`: response magic, response version, begin/trace hooks, the
record projection hook, and the successful-commit hook. Reversing these changes
must reproduce the entire original v2 definition AST exactly. The inherited
21-to-v2 adaptation independently performs the same inverse check against the
original session definitions.

## Handle and metadata adapter

The frozen test's ordinal and `generation << 16` expressions remain a logical
client ledger. They are not assertions about v3 native handle encoding. Native
Tensor IDs come only from successful response records. The bridge checks their
kind, generation, slot range, uniqueness and increasing generation on slot reuse.
The final native node-sequence count must equal the logical descriptor creation
count, including the original composite-command counts.

Only Tensor-ID operand words and the Tensor IDs inside concatenate/stack payload
lists are relocated. Same-frame operands use the checked native earlier-result
reference. An inverse patch must restore every original encoded byte, including
numeric payloads and scalar arguments. Command 14 is a scalar operation and is
excluded from both ID relocation and Tensor-result classification. Bindings are
committed only after a successful whole-frame response; a failed frame exposes
no trial ID and changes no committed binding.

The v3 Storage/Version identity, counter and storage-bound fields are retained
in the report; optional raw traces preserve each full 16-word record. For the
unchanged 13-word observation methods, the bridge returns actual words 0–10 followed by
the actual Storage ID and actual version counter (words 11 and 13). Actual
Version IDs remain independently checked. No numerical value, shape, stride,
counter or reference result is synthesized. The original alias test is
additionally required to share both Storage and Version for detach, and to
have independent Storage and Version for clone.

## Qualification boundaries

The report separates the unchanged 28 numeric cases from seven new lifetime
cases: `.data` version separation, dense/empty clone layout, public Saved
retention across reset, unsaved nonleaf expiry, AccumulateGrad leaf retention,
explicit graph-retention policy, and nongrad coefficient Saved retention.

The graph-retention case deliberately observes a remaining difference. Torch
default backward releases Saved data, whereas this native protocol retains
it until the public graph roots are dropped or reset. That case verifies the
declared native contract and records `OPEN`; a passing qualification does not
mean default-backward API parity. Public Saved unpack is likewise an explicitly
non-grad extension and does not claim full SavedVariable autograd restoration.
All observed optimizer state construction differences remain visible, and the
original direct parameter absolute counters are still strict assertions.

`full_rewrite_complete` remains false and the PyTorch API denominator remains
2590. These cases qualify bounded CPU binary32/rank-0-to-4 behavior. They do not
close all Tensor types, arbitrary strides, foreign storage, devices, hooks,
Python object APIs, higher-order gradients, or complete PyTorch.

## Local evidence

`native-smoke-09/report.json` is six real native bridge checks, 19 interactive
frames and 18 exact scalar/float comparisons. It does not import Torch and is
not counted as an upstream numerical differential. It covers same-frame
arithmetic, failed-frame binding rollback, reset generation, concatenate-list
references, composite hidden-node bookkeeping and the scalar command.

The first smoke report and raw frames are retained in `native-smoke-09-first/`.
Its sole failure was a test input mistake: scalar operation 1 is subtraction,
but the smoke expected addition. The original request returned the correct
`2 - 3 = -1`; the corrected request uses scalar operation 0. The expected
addition result and assertion were unchanged. This did not require a native
change. The separate candidate08 native classification failure remains in the
runtime author's own frozen evidence; this report does not relabel that run.

`prepare-09.json` records successful reversible preparation with zero executed
cases. `local-reference-attempt-09.json` and its provenance report preserve the
actual local result `No module named 'torch'`, with zero native differential
cases. The 35-case real pinned Torch run is pending CI. Test sources and raw
artifacts are listed by content hash in `handoff-manifest.json`.
